"""
Budget sweep: does the spread effect collapse as K grows?

PREDICTION FROM OUR OWN MECHANISM: angular_coverage is capped at K/36 for
K<36 (36 bins of 10 deg). At K>=36 every spread strategy saturates at 1.0,
so coverage stops being a discriminator and the spread-vs-clustered gap must
shrink toward zero. If it does, the mechanism confirms itself and the
crossover point is the figure. If it doesn't, our explanation is wrong.

Resumable: re-running skips (capture, strategy, K, seed) rows already in the
CSV, so a dropped SSH session costs nothing.

    python run_budget_sweep.py 2>&1 | grep -v "Checking\|frustum\|triangle"
"""
import glob, os, time
import numpy as np
import pandas as pd

import strategies_v3                      # registers coverage_q*
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline
from harness import STRATEGIES, FACTORIAL, GEOMETRY

BUDGETS = [8, 12, 16, 24, 32, 48]
SEEDS = (0, 1, 2)
OUT = "results/budget_sweep.csv"

for k in ("coverage_q50", "mean_consecutive_overlap"):   # dupe / degenerate
    STRATEGIES.pop(k, None)
    GEOMETRY.pop(k, None)
ALL = {**STRATEGIES, **FACTORIAL}

os.makedirs("results", exist_ok=True)
done = set()
if os.path.exists(OUT):
    prev = pd.read_csv(OUT)
    done = set(zip(prev.capture, prev.strategy, prev.K, prev.seed))
    print(f"resuming: {len(done)} rows already done")

pipe = MapAnythingPipeline()
rows = []
t_start = time.time()


def flush():
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    rows.clear()


for vid in sorted(glob.glob("data/*.MOV") + glob.glob("data/*.mp4")):
    cap = os.path.splitext(os.path.basename(vid))[0]
    feats = build_feature_table(vid, f"cache/{cap}", pool_size=200,
                                normalise_yaw=False)

    # One reference per capture: the full 200-frame pool.
    ref = pipe.reconstruct(feats, list(feats.index))
    if not (ref and ref.get("success")):
        print(f"{cap}: reference FAILED, skipping")
        continue
    gt = sorted(ref["dimensions"], reverse=True)
    yaw_span = float(np.degrees(np.ptp(feats.yaw)))
    print(f"\n{cap}: yaw={yaw_span:.0f}deg ref={gt[0]:.2f}x{gt[1]:.2f}")

    for K in BUDGETS:
        t0 = time.time()
        n_new = 0
        for name, fn in ALL.items():
            for seed in SEEDS:
                if (cap, name, K, seed) in done:
                    continue
                idx = fn(feats, K, seed=seed)
                r = pipe.reconstruct(feats, idx)
                ok = bool(r and r.get("success"))
                row = dict(capture=cap, strategy=name, K=K, seed=seed, ok=ok,
                           yaw_span_deg=yaw_span, n_selected=len(idx))
                if ok:
                    d = sorted(r["dimensions"], reverse=True)
                    row.update(
                        L=d[0], W=d[1],
                        err=float(np.mean([abs(a - b) / b for a, b in zip(d, gt)])),
                        degenerate=bool(r.get("degenerate")),
                        aspect=r.get("aspect"),
                        h=r.get("ceiling_height_m"),
                        conf=r.get("mean_conf"),
                        tilt=r.get("gravity_tilt_deg"),
                    )
                for g, f in GEOMETRY.items():
                    row[g] = f(feats, idx)
                rows.append(row)
                n_new += 1
        flush()
        if n_new:
            print(f"  K={K:>2}: {n_new} runs in {time.time()-t0:.0f}s")

flush()
print(f"\ntotal {time.time()-t_start:.0f}s -> {OUT}")

d = pd.read_csv(OUT)
d = d[d.ok]
full = d[d.yaw_span_deg > 300]
print("\n=== spread vs clustered by budget (full sweeps only) ===")
fac = full[full.strategy.isin(FACTORIAL)]
if len(fac):
    fac = fac.assign(spread=fac.strategy.str.startswith("spread"))
    piv = fac.pivot_table(index="K", columns="spread", values="err")
    piv.columns = ["clustered", "spread"]
    piv["gap"] = piv["clustered"] - piv["spread"]
    print(piv.round(3).to_string())
print("\n=== coverage achieved vs K ===")
print(full.pivot_table(index="K", columns="strategy",
                       values="angular_coverage").round(3).to_string())
print("\n=== degeneracy rate vs K ===")
print(full.pivot_table(index="K", columns="strategy",
                       values="degenerate").round(3).to_string())
