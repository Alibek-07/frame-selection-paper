import numpy as np, torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from pipeline_mapanything_v2 import build_feature_table
from harness import select_uniform

feats = build_feature_table("data/hikmatollah_apt.MOV", "cache/cap01",
                            pool_size=200, normalise_yaw=False)
paths = [feats.loc[i, "path"] for i in select_uniform(feats, 16)]

m = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
torch.set_grad_enabled(False)
preds = m.infer(load_images(paths), memory_efficient_inference=True,
                minibatch_size=1, use_amp=True, amp_dtype="bf16",
                apply_mask=True, mask_edges=True)

P = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
cams = P[:, :3, 3]
print("camera trajectory extent (m):", (cams.max(0)-cams.min(0)).round(2))

# (A) gravity from camera orientation: OpenCV cams are y-down, so up = -R[:,1]
ups = -P[:, :3, 1]
ups /= np.linalg.norm(ups, axis=1, keepdims=True)
g_cam = ups.mean(0); g_cam /= np.linalg.norm(g_cam)
spread = np.degrees(np.arccos(np.clip(ups @ g_cam, -1, 1)))
print(f"(A) camera-up  {g_cam.round(3)}  spread {spread.mean():.1f}+-{spread.std():.1f} deg")

pts = []
for p in preds:
    xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
    mk = p["mask"][0].float().cpu().numpy().reshape(-1) > 0.5
    pts.append(xyz[mk])
A = np.concatenate(pts); A = A[::10]

# (B) gravity from layeredness: true up gives sharp floor+ceiling peaks
def peakiness(d):
    h, _ = np.histogram(A @ d, bins=np.arange((A@d).min(), (A@d).max()+0.05, 0.05))
    return float(np.sort(h)[-6:].sum() / max(h.sum(), 1))
best, bs = None, -1
for th in np.linspace(0, np.pi, 60):
    for ph in np.linspace(0, 2*np.pi, 120, endpoint=False):
        d = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        s = peakiness(d)
        if s > bs: best, bs = d, s
if best @ g_cam < 0: best = -best
print(f"(B) layered-up {best.round(3)}  score {bs:.3f}")
print(f"    disagreement: {np.degrees(np.arccos(np.clip(best@g_cam,-1,1))):.1f} deg")

for name, up in (("camera-up", g_cam), ("layered-up", best)):
    a = up / np.linalg.norm(up)
    t = np.array([1.0,0,0]); t = t - a*(a@t); t /= np.linalg.norm(t)
    R = np.stack([t, np.cross(a, t), a])
    q = A @ R.T
    best_a, best_ar = 0, np.inf
    for ang in np.arange(0, 90, 0.5):
        r = np.radians(ang); Rz = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
        lo, hi = np.percentile(q[:, :2] @ Rz.T, [5, 95], axis=0)
        if np.prod(hi-lo) < best_ar: best_a, best_ar = ang, float(np.prod(hi-lo))
    r = np.radians(best_a); Rz = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
    lo2, hi2 = np.percentile(q[:, :2] @ Rz.T, [5, 95], axis=0)
    loz, hiz = np.percentile(q[:, 2], [5, 95])
    L, W = sorted(hi2-lo2, reverse=True)
    print(f"[{name:10s}] floor {L:.2f} x {W:.2f} m   ceiling {hiz-loz:.2f} m   yaw {best_a:.1f} deg")
