import numpy as np
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline
from harness import select_uniform

VIDEO = "data/hikmatollah_apt.MOV"

feats = build_feature_table(VIDEO, "cache/cap01", pool_size=200,
                            normalise_yaw=False)
print(feats.head())
print(f"pool={len(feats)}  raw yaw span={np.degrees(np.ptp(feats.yaw)):.1f} deg")

idx = select_uniform(feats, 16)
print("selected:", idx)
print(MapAnythingPipeline().reconstruct(feats, idx))
