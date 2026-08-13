import time, numpy as np, pandas as pd
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline
from harness import STRATEGIES, GEOMETRY

feats = build_feature_table("data/hikmatollah_apt.MOV", "cache/cap01",
                            pool_size=200, normalise_yaw=False)
pipe = MapAnythingPipeline()

# dense reference: every 3rd frame of the pool. Not ground truth, but the
# best estimate available and a legitimate consistency target.
t0 = time.time()
ref = pipe.reconstruct(feats, list(feats.index[::3]))
print(f"REFERENCE (K={len(feats.index[::3])}) {time.time()-t0:.0f}s -> "
      f"{ref['dimensions'][0]:.2f} x {ref['dimensions'][1]:.2f} m, "
      f"h={ref['ceiling_height_m']:.2f}, conf={ref['mean_conf']:.3f}")
gt = ref["dimensions"]

rows = []
for name, fn in STRATEGIES.items():
    idx = fn(feats, 16, seed=0)
    t0 = time.time()
    r = pipe.reconstruct(feats, idx)
    dt = time.time() - t0
    if not r or not r.get("success"):
        print(f"{name:14s} FAILED {r}")
        continue
    L, W = r["dimensions"]
    err = np.mean([abs(a-b)/b for a, b in zip(sorted((L, W), reverse=True),
                                              sorted(gt, reverse=True))])
    row = dict(strategy=name, L=round(L, 2), W=round(W, 2),
               h=round(r["ceiling_height_m"], 2),
               err_vs_ref=round(float(err), 4),
               conf=round(r["mean_conf"], 3), secs=round(dt))
    for g, f in GEOMETRY.items():
        row[g] = round(f(feats, idx), 3)
    rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("results/first_signal.csv", index=False)
