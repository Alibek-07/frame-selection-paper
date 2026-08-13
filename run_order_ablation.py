"""
View-order ablation. MapAnything anchors its world frame to a reference view,
so multi-view feed-forward models are order-sensitive. We fix temporal order
everywhere; this measures how much that choice matters.

If dimension spread across orderings is comparable to the strategy effect,
order is a confound we control rather than a nuisance we ignore.

NOTE: MapAnythingPipeline.reconstruct() sorts indices internally (by design),
so this script calls the model directly to preserve ordering.

    python run_order_ablation.py 2>&1 | grep -v "Checking"
"""
import glob, os
import numpy as np
import pandas as pd
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

import strategies_v3
from pipeline_mapanything_v2 import build_feature_table, align_from_gravity
from harness import select_uniform, select_coverage

model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
torch.set_grad_enabled(False)


def reconstruct_ordered(feats, idx):
    """Same metric as the pipeline, but honours the given view ORDER."""
    paths = [feats.loc[i, "path"] for i in idx]      # NOT sorted
    preds = model.infer(load_images(paths), memory_efficient_inference=True,
                        minibatch_size=1, use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)
    pts, poses = [], []
    for p in preds:
        xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        m = p["mask"][0].float().cpu().numpy().reshape(-1) > 0.5
        pts.append(xyz[m])
        poses.append(p["camera_poses"][0].float().cpu().numpy())
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
    return dict(L=L, W=W, h=float(e[2]))


rows = []
for vid in sorted(glob.glob("data/*.MOV") + glob.glob("data/*.mp4")):
    cap = os.path.splitext(os.path.basename(vid))[0]
    feats = build_feature_table(vid, f"cache/{cap}", pool_size=200,
                                normalise_yaw=False)

    for sname, fn in (("uniform", select_uniform), ("coverage", select_coverage)):
        base = sorted(fn(feats, 16, seed=0))
        orders = {"temporal": list(base), "reverse": base[::-1]}
        for s in (0, 1, 2):
            p = list(base)
            np.random.default_rng(s).shuffle(p)
            orders[f"shuffled{s}"] = p

        for oname, idx in orders.items():
            r = reconstruct_ordered(feats, idx)
            if r is None:
                rows.append(dict(capture=cap, strategy=sname, order=oname, ok=False))
                continue
            rows.append(dict(capture=cap, strategy=sname, order=oname, ok=True, **r))
        print(f"{cap}/{sname} done")

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/order_ablation.csv", index=False)
ok = df[df.ok]
print("\nL (m) by ordering:")
print(ok.pivot_table(index=["capture", "strategy"], columns="order",
                     values="L").round(2).to_string())
print("\nspread across orderings:")
g = ok.groupby(["capture", "strategy"])["L"].agg(["mean", "std", "min", "max"])
g["range_pct"] = ((g["max"] - g["min"]) / g["mean"] * 100).round(1)
print(g.round(3).to_string())
print("\nIf range_pct is comparable to the strategy effect (~20%), view order "
      "is a first-class confound and must be reported.")
