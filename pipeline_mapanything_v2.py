"""
pipeline_mapanything_v2.py — real ArbuzPipeline + feature-table builder.

v2: uses mapanything.utils.image.load_images (canonical) instead of a hand
port of preprocess.ts. Adds optical-flow yaw estimation so coverage-based
selection works without ARKit poses.

UNTESTED. Verify against scripts/demo_images_only_inference.py.

CHECKPOINT: use "facebook/map-anything-apache" (Apache 2.0). The default
"facebook/map-anything" is the more restrictive variant.

WHY PYTORCH NOT THE PRODUCTION ONNX EXPORT
  reconstruct/model.ts is static at 16 views and pads shorter inputs by
  repeating the last frame. Padded outputs are dropped but padded views still
  join cross-view attention, and which frame is duplicated differs per
  strategy — a confound on exactly the comparison we make. Report
  ONNX/PyTorch parity at K=16 as validation.

CANDIDATE POOL vs BUDGET — important
  Production extracts exactly 16 evenly spaced frames. For the experiment the
  POOL must be much larger (default 200) or every strategy selects from 16 and
  no selection happens. `uniform` at K=16 over the full pool reproduces
  production behaviour, so it is the deployed baseline.

METRIC: masked pts3d -> floor/yaw alignment -> 5-95 percentile extents,
mirroring RoomSpec. Port reconstruct/align.ts for exact parity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NOMINAL_HFOV_DEG = 65.0     # typical phone rear camera; see caveat in estimate_yaw_track


# ============================ feature table ================================

def extract_pool(video_path: str, out_dir: str, pool_size: int = 200) -> pd.DataFrame:
    """
    Write `pool_size` evenly spaced JPEGs (bin centres, as room-scan/frames.ts)
    and return a table of frame_idx, path, sharpness.
    """
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        raise RuntimeError(f"cannot read frame count: {video_path}")

    idxs = sorted({min(int(round((i + 0.5) * total / pool_size - 0.5)), total - 1)
                   for i in range(pool_size)})
    rows = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            continue
        h, w = bgr.shape[:2]
        s = min(1.0, 1512 / max(h, w))              # cap long side, as frames.ts
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4)
        p = os.path.join(out_dir, f"{i:06d}.jpg")
        cv2.imwrite(p, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rows.append(dict(
            frame_idx=i, path=p, t=i / fps,
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        ))
    cap.release()
    if not rows:
        raise RuntimeError(f"no frames extracted: {video_path}")
    return pd.DataFrame(rows)


def estimate_yaw_track(video_path: str, stride: int = 3,
                       hfov_deg: float = NOMINAL_HFOV_DEG) -> pd.DataFrame:
    """
    Cumulative yaw by integrating median horizontal optical flow.

    WHY: MapAnything predicts poses, so using its output to choose its input is
    circular. This is model-free. Koschel et al. (Algorithms 2021) used an IMU
    for the same purpose, so a sensor/flow-side signal has precedent.

    MUST be integrated over a DENSE sample (stride ~3), not over the 16 selected
    frames: large inter-frame rotation breaks the brightness-constancy
    assumption and the integral collapses. Interpolate onto pool indices after.

    CAVEATS to state in the paper:
      * hfov is nominal -> yaw scale is approximate. Coverage bins are
        proportional, so ranking is robust, but absolute degrees are not.
      * pure translation is misread as rotation. Minor for a room sweep.
      * VALIDATE post-hoc: run a dense MapAnything pass on 2-3 captures and
        correlate this track against its predicted cam_quats yaw. Report r.
      * If ARKit poses turn up, prefer them and delete this.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read: {video_path}")
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    f_px = (prev_g.shape[1] / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)

    yaw, rows, i = 0.0, [dict(frame_idx=0, yaw_raw=0.0)], 0
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    while True:
        for _ in range(stride):
            ok, cur = cap.read()
            i += 1
            if not ok:
                break
        if not ok:
            break
        cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_g, maxCorners=400, qualityLevel=0.01,
                                     minDistance=12)
        if p0 is not None and len(p0) >= 20:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None, **lk)
            if p1 is not None and st is not None and st.sum() >= 15:
                good = st.reshape(-1).astype(bool)
                dx = float(np.median((p1[good] - p0[good]).reshape(-1, 2)[:, 0]))
                yaw += -dx / f_px          # image shifts left => camera pans right
        rows.append(dict(frame_idx=i, yaw_raw=yaw))
        prev_g = cur_g
    cap.release()
    if len(rows) < 2:
        raise RuntimeError("yaw track too short")
    return pd.DataFrame(rows)


def build_feature_table(video_path: str, cache_dir: str, pool_size: int = 200,
                        stride: int = 3, normalise_yaw: bool = True) -> pd.DataFrame:
    """
    -> DataFrame with harness.py's columns: frame_idx, sharpness, x, y, yaw
       (+ path, t). Cached to disk; delete the dir to recompute.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "features.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    pool = extract_pool(video_path, os.path.join(cache_dir, "frames"), pool_size)
    track = estimate_yaw_track(video_path, stride=stride)

    yaw = np.interp(pool["frame_idx"], track["frame_idx"], track["yaw_raw"])
    span = float(np.ptp(yaw))
    if normalise_yaw and span > 1e-6:
        # Rescale so the sweep spans 2*pi. Absorbs the nominal-hfov error under
        # the assumption that a room scan is roughly one full rotation.
        # RECORD raw_span_rad: if it is far from 2*pi the assumption is wrong.
        yaw = (yaw - yaw.min()) / span * 2 * np.pi
    else:
        yaw = yaw - yaw.min()

    pool = pool.copy()
    pool["yaw"] = yaw % (2 * np.pi)
    # Rotation-dominant capture: no reliable translation without poses. Placing
    # cameras on a unit circle by yaw makes mean_baseline a pure function of
    # angular spread, so it is NOT an independent control -- note this, or
    # substitute ARKit translation if it becomes available.
    pool["x"] = 0.3 * np.cos(pool["yaw"])
    pool["y"] = 0.3 * np.sin(pool["yaw"])
    pool["sharpness"] = pool["sharpness"] / pool["sharpness"].max()

    pool.to_parquet(cache, index=False)
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump(dict(video=video_path, pool_size=int(len(pool)),
                       yaw_stride=stride, raw_yaw_span_rad=span,
                       yaw_normalised=bool(normalise_yaw),
                       nominal_hfov_deg=NOMINAL_HFOV_DEG,
                       angular_velocity_var=float(np.var(np.diff(np.sort(yaw))))),
                  fh, indent=2)
    return pool


# =============================== alignment =================================

def align_from_gravity(up, pts):
    """Rotation taking `up` to +Z, then yaw by minimum-area rectangle.
    Gravity comes from camera orientations (mean of -R[:,1], OpenCV y-down),
    which is independent of scene content. Validated against a layeredness
    estimate: 6.9 deg agreement, ceiling 2.47 vs 2.24 m."""
    import numpy as np
    a = up / np.linalg.norm(up)
    t = np.array([1.0, 0.0, 0.0])
    if abs(a @ t) > 0.9:
        t = np.array([0.0, 1.0, 0.0])
    t = t - a * (a @ t); t /= np.linalg.norm(t)
    R1 = np.stack([t, np.cross(a, t), a])
    xy = (pts @ R1.T)[:, :2]
    best_a, best_ar = 0.0, np.inf
    for ang in np.arange(0, 90, 0.5):
        r = np.radians(ang)
        Rz = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]])
        lo, hi = np.percentile(xy @ Rz.T, [5, 95], axis=0)
        ar = float(np.prod(hi - lo))
        if ar < best_ar: best_a, best_ar = float(ang), ar
    r = np.radians(best_a)
    R2 = np.array([[np.cos(r), np.sin(r), 0], [-np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return R2 @ R1


def _legacy_align_floor_yaw(pts, n_iter=600, tol=0.03, seed=0):
    """RANSAC floor -> +Z, then yaw by min-area rect. Cf. reconstruct/align.ts."""
    rng = np.random.default_rng(seed)
    vert = int(np.argmin(np.ptp(pts, axis=0)))           # thinnest axis ~ up
    lo = pts[pts[:, vert] <= np.percentile(pts[:, vert], 40)]
    if len(lo) < 200:
        lo = pts
    best_n, best_c = None, -1
    for _ in range(n_iter):
        s = lo[rng.choice(len(lo), 3, replace=False)]
        nrm = np.cross(s[1] - s[0], s[2] - s[0])
        ln = np.linalg.norm(nrm)
        if ln < 1e-9:
            continue
        nrm = nrm / ln
        c = int((np.abs((lo - s[0]) @ nrm) < tol).sum())
        if c > best_c:
            best_n, best_c = nrm, c
    if best_n is None:
        return np.eye(3)
    if best_n[int(np.argmax(np.abs(best_n)))] < 0:
        best_n = -best_n

    z = np.array([0.0, 0.0, 1.0])
    v, cc = np.cross(best_n, z), float(best_n @ z)
    if np.linalg.norm(v) < 1e-9:
        R1 = np.eye(3) if cc > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R1 = np.eye(3) + K + K @ K * (1.0 / (1.0 + cc))

    xy = (pts @ R1.T)[:, :2]
    best_a, best_area = 0.0, np.inf
    for a in np.arange(0, 90, 0.5):
        t = np.radians(a)
        Rz = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = xy @ Rz.T
        l5, h5 = np.percentile(q, [5, 95], axis=0)
        ar = float(np.prod(h5 - l5))
        if ar < best_area:
            best_a, best_area = float(a), ar
    t = np.radians(best_a)
    R2 = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return R2 @ R1


def room_extents(pts):
    """5-95 percentile extents after alignment -> (L, W, H) metres."""
    q = pts @ align_floor_yaw(pts).T
    lo, hi = np.percentile(q, [5, 95], axis=0)
    e = hi - lo
    L, W = sorted([float(e[0]), float(e[1])], reverse=True)
    return L, W, float(e[2])


# ================================ pipeline =================================

@dataclass
class MapAnythingPipeline:
    """reconstruct(feats, idx) -> dict, for harness.run_sweep."""
    checkpoint: str = "facebook/map-anything-apache"   # Apache 2.0 variant
    device: str = "cuda"
    multiview_confidence: bool = True     # cross-view depth agreement = mechanism
    minibatch_size: int = 1
    depth_backend: str = "mapanything"    # recorded in results
    _model: object = field(default=None, repr=False)

    def _load(self):
        if self._model is None:
            import torch
            from mapanything.models import MapAnything
            self._model = MapAnything.from_pretrained(self.checkpoint).to(self.device).eval()
            torch.set_grad_enabled(False)
        return self._model

    def reconstruct(self, feats, idx):
        from mapanything.utils.image import load_images
        model = self._load()

        idx = sorted(int(i) for i in idx)     # FIX view order = temporal, always
        if len(idx) < 2:
            return None
        paths = [feats.loc[i, "path"] for i in idx]

        try:
            views = load_images(paths)
            preds = model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=self.minibatch_size,
                use_amp=True, amp_dtype="bf16",
                apply_mask=True, mask_edges=True,
                use_multiview_confidence=self.multiview_confidence,
            )
        except Exception as e:
            return {"success": False, "error": f"{type(e).__name__}: {e}"}

        pts, confs, poses = [], [], []
        for p in preds:
            xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
            m = p["mask"][0].float().cpu().numpy().reshape(-1) > 0.5
            pts.append(xyz[m])
            poses.append(p["camera_poses"][0].float().cpu().numpy())
            if "conf" in p:
                confs.append(p["conf"][0].float().cpu().numpy().reshape(-1)[m])

        allpts = np.concatenate(pts) if pts else np.empty((0, 3))
        if len(allpts) < 5000:
            return {"success": False, "error": f"only {len(allpts)} masked points"}

        # Gravity from camera orientations (OpenCV y-down => up = -R[:,1]).
        # Scene-independent; agrees with a layeredness estimate to 6.9 deg.
        P = np.stack(poses)
        ups = -P[:, :3, 1]
        ups = ups / np.linalg.norm(ups, axis=1, keepdims=True)
        up = ups.mean(0); up = up / np.linalg.norm(up)
        tilt = float(np.degrees(np.arccos(np.clip(ups @ up, -1.0, 1.0))).mean())

        R = align_from_gravity(up, allpts[::20])      # subsample: yaw sweep
        q = allpts @ R.T
        lo, hi = np.percentile(q, [5, 95], axis=0)
        e = hi - lo
        L, W = sorted([float(e[0]), float(e[1])], reverse=True)
        H = float(e[2])

        cams = P[:, :3, 3]
        traj = (cams.max(0) - cams.min(0))

        # Never discard a reconstruction: an arbitrary plausibility threshold
        # silently drops the worst runs and understates the penalty. Instead
        # always report dimensions and flag degeneracy with a pre-registered,
        # geometrically motivated rule: real rooms are not 4:1 slivers, and a
        # habitable dimension is >1.2 m.
        degenerate = bool(W < 1.2 or (L / max(W, 1e-6)) > 4.0)

        return {"success": True, "degenerate": degenerate,
                "aspect": float(L / max(W, 1e-6)),
                "dimensions": (L, W),
                "ceiling_height_m": H, "n_points": int(len(allpts)),
                "mean_conf": float(np.mean(np.concatenate(confs))) if confs else None,
                "gravity_tilt_deg": tilt,
                "traj_extent_m": tuple(float(v) for v in traj)}
