"""
Downstream VLM experiment: does frame selection change SEMANTIC spatial
estimation the same way it changes geometric reconstruction?

Same frame subsets, same captures, same reference. The VLM sees the selected
frames directly and estimates room dimensions. We also ask for its own
confidence, giving a SECOND miscalibration probe alongside MapAnything's
multi-view consistency score.

Three outcomes, all publishable:
  A) VLM degrades like geometry  -> selection is a shared upstream bottleneck
  B) VLM stays accurate          -> VLMs could flag geometric failure
  C) VLM wrong AND confident     -> silent failure propagates through the
                                    whole stack, which is the deployment result

Cached per (capture, strategy, seed): re-runs cost nothing.

    export ANTHROPIC_API_KEY=...
    python -u run_vlm_estimate.py
"""
import base64, glob, io, json, os, time
import numpy as np
import pandas as pd
from PIL import Image

import strategies_v3
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline
from harness import STRATEGIES, FACTORIAL, GEOMETRY

MODEL = os.environ.get("VLM_MODEL", "claude-sonnet-5")
SEEDS = (0, 1, 2)
K = 16
MAX_PX = 768                      # long side; keeps token cost sane
CACHE = "cache/vlm"
OUT = "results/vlm_estimates.csv"

# Core cells only: the 2x2 plus the three strategies the paper names.
WANTED = ["uniform", "sharpness", "coverage",
          "spread_high", "spread_low", "clustered_high", "clustered_low"]
ALL = {**STRATEGIES, **FACTORIAL}
ALL = {k: v for k, v in ALL.items() if k in WANTED}

PROMPT = """These are {n} frames sampled from a single continuous video of ONE room, in temporal order.

Estimate the room's floor dimensions.

Reply with ONLY a JSON object, no other text:
{{"length_m": <number>, "width_m": <number>, "ceiling_m": <number>, "confidence": <0-1>, "reasoning": "<one sentence>"}}

length_m is the longer horizontal dimension, width_m the shorter, both in metres. confidence is how sure you are that your estimate is within 15% of the true value."""


def encode(path):
    im = Image.open(path).convert("RGB")
    s = MAX_PX / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ask_vlm(paths):
    import anthropic
    client = anthropic.Anthropic()
    content = [{"type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": encode(p)}} for p in paths]
    content.append({"type": "text", "text": PROMPT.format(n=len(paths))})
    for attempt in range(4):
        try:
            r = client.messages.create(model=MODEL, max_tokens=300,
                                       messages=[{"role": "user", "content": content}])
            txt = "".join(b.text for b in r.content if b.type == "text")
            txt = txt.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            return json.loads(txt.strip())
        except Exception as e:
            if attempt == 3:
                return {"error": f"{type(e).__name__}: {e}"}
            time.sleep(2 ** attempt * 3)


os.makedirs(CACHE, exist_ok=True)
os.makedirs("results", exist_ok=True)
pipe = MapAnythingPipeline()
rows, n_calls = [], 0

for vid in sorted(glob.glob("data/*.MOV") + glob.glob("data/*.mp4")):
    cap = os.path.splitext(os.path.basename(vid))[0]
    feats = build_feature_table(vid, f"cache/{cap}", pool_size=200,
                                normalise_yaw=False)
    ref = pipe.reconstruct(feats, list(feats.index))
    if not (ref and ref.get("success")):
        print(f"{cap}: reference failed"); continue
    gt = sorted(ref["dimensions"], reverse=True)
    print(f"\n{cap}: ref {gt[0]:.2f} x {gt[1]:.2f}")

    for name, fn in ALL.items():
        for seed in SEEDS:
            key = f"{CACHE}/{cap}__{name}__{seed}.json"
            idx = sorted(fn(feats, K, seed=seed))
            if os.path.exists(key):
                res = json.load(open(key))
            else:
                res = ask_vlm([feats.loc[i, "path"] for i in idx])
                if "error" not in res:          # never cache a failure
                    json.dump(res, open(key, "w"))
                n_calls += 1
                time.sleep(1.0)

            row = dict(capture=cap, strategy=name, seed=seed,
                       ref_L=gt[0], ref_W=gt[1])
            if "error" in res:
                row["ok"] = False
                row["err_msg"] = res["error"]
            else:
                try:
                    d = sorted([float(res["length_m"]), float(res["width_m"])],
                               reverse=True)
                    row.update(
                        ok=True, vlm_L=d[0], vlm_W=d[1],
                        vlm_ceiling=float(res.get("ceiling_m", np.nan)),
                        vlm_conf=float(res.get("confidence", np.nan)),
                        vlm_err=float(np.mean([abs(a - b) / b
                                               for a, b in zip(d, gt)])),
                        reasoning=str(res.get("reasoning", ""))[:200])
                except Exception as e:
                    row["ok"] = False; row["err_msg"] = str(e)
            for g, f in GEOMETRY.items():
                row[g] = f(feats, idx)
            rows.append(row)
        print(f"  {name:15s} done")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\n{n_calls} new API calls -> {OUT}")

ok = df[df.ok]
print("\n=== VLM room-dimension error by strategy ===")
print(ok.groupby("strategy").agg(
    vlm_err=("vlm_err", "mean"), sd=("vlm_err", "std"),
    vlm_conf=("vlm_conf", "mean"), n=("vlm_err", "size")
).round(3).sort_values("vlm_err").to_string())

fac = ok[ok.strategy.isin(FACTORIAL)]
if len(fac):
    fac = fac.assign(spread=fac.strategy.str.startswith("spread"),
                     hiq=fac.strategy.str.endswith("high"))
    print("\n=== VLM 2x2 ===")
    print(fac.pivot_table(index="spread", columns="hiq",
                          values="vlm_err").round(3).to_string())

print("\n=== does the VLM know when it is wrong? ===")
from scipy import stats
if ok.vlm_conf.notna().sum() > 5:
    r, p = stats.spearmanr(ok.vlm_conf, ok.vlm_err)
    print(f"spearman(vlm_conf, vlm_err) = {r:+.3f}  p={p:.2e}")
    print("negative => calibrated; ~zero or positive => it does not know")

geo = ok.dropna(subset=["vlm_err"])
if len(geo) > 5:
    r, p = stats.spearmanr(geo.angular_coverage, geo.vlm_err)
    print(f"spearman(angular_coverage, vlm_err) = {r:+.3f}  p={p:.2e}")
    r, p = stats.spearmanr(geo.mean_sharpness, geo.vlm_err)
    print(f"spearman(mean_sharpness,   vlm_err) = {r:+.3f}  p={p:.2e}")
