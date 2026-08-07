import numpy as np, torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from pipeline_mapanything_v2 import build_feature_table
from harness import select_uniform

feats = build_feature_table("data/hikmatollah_apt.MOV", "cache/cap01",
                            pool_size=200, normalise_yaw=False)
idx = select_uniform(feats, 16)
paths = [feats.loc[i, "path"] for i in idx]

m = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
torch.set_grad_enabled(False)
preds = m.infer(load_images(paths), memory_efficient_inference=True,
                minibatch_size=1, use_amp=True, amp_dtype="bf16",
                apply_mask=True, mask_edges=True)

cams = np.stack([p["cam_trans"][0].float().cpu().numpy() for p in preds])
print("camera positions:\n", cams.round(3))
print("cam std per axis:", cams.std(0).round(3))
up = int(np.argmin(cams.std(0)))
print(f"--> UP AXIS = {up}  (min camera variance)")

pts, masked = [], 0
for p in preds:
    xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
    mk = p["mask"][0].float().cpu().numpy().reshape(-1) > 0.5
    masked += int(mk.sum()); pts.append(xyz[mk])
A = np.concatenate(pts)
print(f"mask pass rate: {masked/(len(preds)*p['mask'][0].numel()):.1%}")
print("raw extents  :", (A.max(0) - A.min(0)).round(2))
lo, hi = np.percentile(A, [5, 95], axis=0)
print("5-95 extents :", (hi - lo).round(2))
horiz = [a for a in range(3) if a != up]
print(f"--> floor {(hi-lo)[horiz[0]]:.2f} x {(hi-lo)[horiz[1]]:.2f} m, "
      f"ceiling {(hi-lo)[up]:.2f} m")
np.save("cache/cap01/cloud.npy", A[::20])
