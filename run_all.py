import glob, os, numpy as np, pandas as pd
import strategies_v3
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline
from harness import STRATEGIES, FACTORIAL, GEOMETRY

for k in ("coverage_q50", "mean_consecutive_overlap"):   # dupes / degenerate
    STRATEGIES.pop(k, None); GEOMETRY.pop(k, None)

pipe = MapAnythingPipeline()
rows = []
for vid in sorted(glob.glob("data/*.MOV") + glob.glob("data/*.mp4")):
    cap = os.path.splitext(os.path.basename(vid))[0]
    feats = build_feature_table(vid, f"cache/{cap}", pool_size=200,
                                normalise_yaw=False)
    ref = pipe.reconstruct(feats, list(feats.index))
    if not (ref and ref.get("success")):
        print(f"{cap}: reference FAILED"); continue
    gt = sorted(ref["dimensions"], reverse=True)
    print(f"{cap}: pool={len(feats)} yaw={np.degrees(np.ptp(feats.yaw)):.0f}deg "
          f"ref={gt[0]:.2f}x{gt[1]:.2f}")
    for name, fn in {**STRATEGIES, **FACTORIAL}.items():
        for seed in (0, 1, 2):
            idx = fn(feats, 16, seed=seed)
            r = pipe.reconstruct(feats, idx)
            ok = bool(r and r.get("success"))
            row = dict(capture=cap, strategy=name, seed=seed, ok=ok,
                       degenerate=bool(r.get("degenerate")) if ok else None,
                       aspect=r.get("aspect") if ok else None)
            if ok:
                d = sorted(r["dimensions"], reverse=True)
                row.update(err=float(np.mean([abs(a-b)/b for a, b in zip(d, gt)])),
                           conf=r["mean_conf"], tilt=r["gravity_tilt_deg"],
                           traj=float(np.linalg.norm(r["traj_extent_m"])))
                for g, f in GEOMETRY.items():
                    row[g] = f(feats, idx)
            rows.append(row)

df = pd.DataFrame(rows); df.to_csv("results/all_captures.csv", index=False)
ok = df[df.ok]
print("\n", ok.groupby("strategy")["err"].agg(["mean","std","count"]).round(3).sort_values("mean").to_string())
print("\nper-capture err:\n", ok.pivot_table(index="strategy", columns="capture", values="err").round(3).to_string())
