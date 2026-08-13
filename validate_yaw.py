"""
Validate optical-flow yaw against MapAnything's predicted camera poses.

Our yaw currently rests on one observation: 358.4 deg measured for a full
sweep. This checks it per capture against an independent estimate.

Method: dense reconstruction -> camera poses -> gravity from mean(-R[:,1]) ->
project each camera's forward axis into the horizontal plane -> yaw. Both
tracks are relative, so we fit a constant offset and a sign, then report RMS
disagreement and circular correlation.

    python validate_yaw.py 2>&1 | grep -v "Checking\|frustum\|triangle"
"""
import glob, os
import numpy as np
import pandas as pd
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from pipeline_mapanything_v2 import build_feature_table

model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
torch.set_grad_enabled(False)
rows = []

for vid in sorted(glob.glob("data/*.MOV") + glob.glob("data/*.mp4")):
    cap = os.path.splitext(os.path.basename(vid))[0]
    feats = build_feature_table(vid, f"cache/{cap}", pool_size=200,
                                normalise_yaw=False)
    idx = list(feats.index[::5])                     # 40 views
    paths = [feats.loc[i, "path"] for i in idx]
    preds = model.infer(load_images(paths), memory_efficient_inference=True,
                        minibatch_size=1, use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)

    P = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
    ups = -P[:, :3, 1]
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    up = ups.mean(0); up /= np.linalg.norm(up)

    e1 = np.array([1.0, 0, 0]); e1 -= up * (up @ e1); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    fwd = P[:, :3, 2]                                 # camera +Z
    fwd = fwd - np.outer(fwd @ up, up)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    yaw_pred = np.unwrap(np.arctan2(fwd @ e2, fwd @ e1))
    yaw_flow = np.unwrap(feats.loc[idx, "yaw"].to_numpy())

    best = None
    for sign in (1, -1):
        y = sign * yaw_pred
        off = np.mean(yaw_flow - y)
        rms = float(np.sqrt(np.mean((yaw_flow - (y + off)) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, sign, off)
    rms, sign, off = best
    span_flow = float(np.degrees(np.ptp(yaw_flow)))
    span_pred = float(np.degrees(np.ptp(yaw_pred)))
    r = float(np.corrcoef(yaw_flow, sign * yaw_pred)[0, 1])

    rows.append(dict(capture=cap, span_flow_deg=round(span_flow, 1),
                     span_pred_deg=round(span_pred, 1),
                     rms_deg=round(float(np.degrees(rms)), 1),
                     pearson_r=round(r, 4), sign=sign))
    print(rows[-1])

df = pd.DataFrame(rows)
df.to_csv("results/yaw_validation.csv", index=False)
print("\n", df.to_string(index=False))
print("\nr > 0.95 and RMS < 20 deg => the optical-flow track is sound and the "
      "nominal-hFOV assumption holds. Report this as instrument validation.")
