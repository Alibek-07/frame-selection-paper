import numpy as np, pandas as pd
import strategies_v3  # registers coverage_q*
import strategies_v3  # registers coverage_q*
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline
from harness import STRATEGIES, FACTORIAL, GEOMETRY

feats = build_feature_table("data/hikmatollah_apt.MOV", "cache/cap01",
                            pool_size=200, normalise_yaw=False)
pipe = MapAnythingPipeline()
ref = pipe.reconstruct(feats, list(list(feats.index)))
gt = sorted(ref["dimensions"], reverse=True)
print(f"reference {gt[0]:.2f} x {gt[1]:.2f}  (NOT ground truth)")

rows = []
for name, fn in {**STRATEGIES, **FACTORIAL}.items():
    for seed in (0, 1, 2):
        idx = fn(feats, 16, seed=seed)
        r = pipe.reconstruct(feats, idx)
        if not r or not r.get("success"):
            rows.append(dict(strategy=name, seed=seed, ok=False)); continue
        d = sorted(r["dimensions"], reverse=True)
        row = dict(strategy=name, seed=seed, ok=True,
                   L=round(d[0],2), W=round(d[1],2),
                   h=round(r["ceiling_height_m"],2),
                   err=round(float(np.mean([abs(a-b)/b for a,b in zip(d,gt)])),4),
                   conf=round(r["mean_conf"],3))
        for g,f in GEOMETRY.items():
            row[g] = round(f(feats, idx), 3)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results/factorial.csv", index=False)
print(df[df.ok].groupby("strategy")[
    ["err","conf","angular_coverage","mean_sharpness",
     "mean_consecutive_overlap","max_angular_gap"]].mean().round(3).to_string())
