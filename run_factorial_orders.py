"""
The definitive 2x2: spread x quality, MARGINALISED OVER VIEW ORDER.

WHY: the order ablation showed that reordering the SAME 16 frames moves the
long dimension by ~19% on average (up to 54%). That is the same magnitude as
the strategy effect. A single-ordering measurement therefore carries variance
comparable to the effect being measured. Averaging over orderings puts that
variance inside the error bars.

4 cells x 5 captures x 3 seeds x 3 orderings, K=16 -> 180 reconstructions.

    python -u run_factorial_orders.py 2>&1 | grep -v "Checking"
"""
import glob, os, time
import numpy as np
import pandas as pd
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

import strategies_v3
from pipeline_mapanything_v2 import build_feature_table, align_from_gravity
from harness import FACTORIAL, GEOMETRY, select_uniform

N_ORDERS = 3
SEEDS = (0, 1, 2)
K = 16

model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
torch.set_grad_enabled(False)


def recon(feats, idx_ordered):
    """Metric identical to the pipeline, but honours the given view ORDER."""
    paths = [feats.loc[i, "path"] for i in idx_ordered]
    preds = model.infer(load_images(paths), memory_efficient_inference=True,
                        minibatch_size=1, use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True,
                        use_multiview_confidence=True)
    pts, poses, confs = [], [], []
    for p in preds:
        xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        m = p["mask"][0].float().cpu().numpy().reshape(-1) > 0.5
        pts.append(xyz[m])
        poses.append(p["camera_poses"][0].float().cpu().numpy())
        if "conf" in p:
            confs.append(p["conf"][0].float().cpu().numpy().reshape(-1)[m])
    A = np.concatenate(pts)
    if len(A) < 5000:
        return None
    P = np.stack(poses)
    ups = -P[:, :3, 1]
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    up = ups.mean(0); up /= np.linalg.norm(up)
    q = A @ align_from_gravity(up, A[::20]).T
    lo, hi = np.percentile(q, [5, 95], axis=0)
    e = hi - lo
    L, W = sorted([float(e[0]), float(e[1])], reverse=True)
    return dict(L=L, W=W, h=float(e[2]),
                degenerate=bool(W < 1.2 or L / max(W, 1e-6) > 4.0),
                aspect=float(L / max(W, 1e-6)),
                conf=float(np.mean(np.concatenate(confs))) if confs else np.nan)


rows = []
t0 = time.time()
for vid in sorted(glob.glob("data/*.MOV") + glob.glob("data/*.mp4")):
    cap = os.path.splitext(os.path.basename(vid))[0]
    feats = build_feature_table(vid, f"cache/{cap}", pool_size=200,
                                normalise_yaw=False)

    ref = recon(feats, list(feats.index))
    if ref is None:
        print(f"{cap}: reference failed, skipping"); continue
    gt = sorted([ref["L"], ref["W"]], reverse=True)
    unwrapped = float(np.degrees(np.ptp(np.unwrap(feats.yaw.to_numpy()))))
    print(f"{cap}: ref {gt[0]:.2f}x{gt[1]:.2f}  unwrapped yaw {unwrapped:.0f} deg")

    for name, fn in FACTORIAL.items():
        for seed in SEEDS:
            base = sorted(fn(feats, K, seed=seed))
            geo = {g: f(feats, base) for g, f in GEOMETRY.items()}
            for o in range(N_ORDERS):
                idx = list(base) if o == 0 else list(base)
                if o > 0:
                    np.random.default_rng(1000 * seed + o).shuffle(idx)
                r = recon(feats, idx)
                row = dict(capture=cap, strategy=name, seed=seed, order=o,
                           ok=r is not None, unwrapped_yaw_deg=unwrapped)
                if r:
                    d = sorted([r["L"], r["W"]], reverse=True)
                    row.update(L=d[0], W=d[1], h=r["h"], conf=r["conf"],
                               degenerate=r["degenerate"], aspect=r["aspect"],
                               err=float(np.mean([abs(a - b) / b
                                                  for a, b in zip(d, gt)])))
                row.update(geo)
                rows.append(row)
    print(f"  {cap} done ({time.time()-t0:.0f}s)")

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/factorial_orders.csv", index=False)

ok = df[df.ok]
print("\n=== 2x2, marginalised over view order (all captures) ===")
ok = ok.assign(spread=ok.strategy.str.startswith("spread"),
               hiq=ok.strategy.str.endswith("high"))
print(ok.pivot_table(index="spread", columns="hiq", values="err",
                     aggfunc=["mean", "std"]).round(3).to_string())
print("\nper cell:")
print(ok.groupby("strategy").agg(
    err=("err", "mean"), sd=("err", "std"),
    degen=("degenerate", "mean"), conf=("conf", "mean"),
    n=("err", "size")).round(3).to_string())

print("\n=== order variance WITHIN a fixed selection ===")
w = ok.groupby(["capture", "strategy", "seed"])["L"].agg(["mean", "std", "min", "max"])
w["range_pct"] = ((w["max"] - w["min"]) / w["mean"] * 100)
print(f"mean within-selection range from ordering alone: {w.range_pct.mean():.1f}%")
print(f"between-strategy spread in err: "
      f"{ok.groupby('strategy')['err'].mean().max() - ok.groupby('strategy')['err'].mean().min():.3f}")
