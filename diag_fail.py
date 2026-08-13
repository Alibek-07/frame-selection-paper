import strategies_v3
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline
from harness import FACTORIAL, GEOMETRY
pipe = MapAnythingPipeline()
for cap, seeds in [("hikmatollah_apt",[2]),("home-office-portrait",[0,2]),
                   ("mokhi_office",[0,2])]:
    ext = "mp4" if "portrait" in cap else "MOV"
    feats = build_feature_table(f"data/{cap}.{ext}", f"cache/{cap}",
                                pool_size=200, normalise_yaw=False)
    for s in seeds:
        idx = FACTORIAL["clustered_low"](feats, 16, seed=s)
        r = pipe.reconstruct(feats, idx)
        g = {k: round(f(feats, idx), 3) for k, f in GEOMETRY.items()}
        print(f"{cap} seed{s}: {r.get('error') if r else 'None'}")
        print(f"   cov={g['angular_coverage']} gap={g['max_angular_gap']} sharp={g['mean_sharpness']}")
