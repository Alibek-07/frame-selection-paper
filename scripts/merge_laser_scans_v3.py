#!/usr/bin/env python3
"""
Register + merge ARKitScenes Faro laser scans into one downsampled cloud.

v3: all transforms done in numpy. v2 segfaulted because copy.deepcopy on an
Open3D PointCloud is unsafe on some 0.18 builds. Open3D is now used only to
read, downsample, and write - never for per-variant copies.

POSE FORMAT: DATA.md says 4x3 whitespace-delimited (lines 0-2 rotation,
line 3 translation). Actually COMMA-delimited 4x4 with translation in the
LAST ROW (row-vector convention), so the column-vector transform is the
transpose. All four combinations are tested below.

Usage:
    python merge_laser_scans_v3.py ~/arkit_spike/laser_scanner_point_clouds/484534 \
        --voxel 0.02 --out ~/arkit_spike/merged_484534.ply
"""

import argparse
import glob
import os

import numpy as np
import open3d as o3d


def read_pose_matrix(pose_path):
    """Parse tolerantly: any delimiter, 12 or 16 values."""
    raw = open(pose_path).read().replace(",", " ")
    vals = np.array([float(x) for x in raw.split()], dtype=np.float64)
    if vals.size == 16:
        return vals.reshape(4, 4)
    if vals.size == 12:
        grid = vals.reshape(4, 3)
        M = np.eye(4)
        M[:3, :3] = grid[:3, :]
        M[3, :3] = grid[3, :]
        return M
    raise ValueError(f"{pose_path}: expected 12 or 16 values, got {vals.size}")


def to_transform(M, transpose, invert):
    T = M.T.copy() if transpose else M.copy()
    return np.linalg.inv(T) if invert else T


def apply(pts, T):
    """pts: (N,3) -> (N,3), column-vector convention."""
    return pts @ T[:3, :3].T + T[:3, 3]


def looks_like_a_room(ext):
    s = sorted(ext)
    return 2.0 <= s[0] <= 4.5 and 2.0 <= s[1] <= 25.0 and s[2] <= 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir", help="laser_scanner_point_clouds/<visit_id>/")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--out", default="merged.ply")
    ap.add_argument("--force", choices=["T", "M", "Tinv", "Minv"])
    ap.add_argument("--recenter", action="store_true",
                    help="shift merged cloud so its min corner is at origin")
    args = ap.parse_args()

    plys = sorted(glob.glob(os.path.join(args.scan_dir, "*.ply")))
    if not plys:
        raise SystemExit(f"no .ply files in {args.scan_dir}")

    print(f"{len(plys)} scans, voxel={args.voxel}m\n")

    # ---- expensive pass: read once, downsample, drop to numpy -------------
    scans = []   # list of (points Nx3, colors Nx3 or None, pose 4x4)
    for ply in plys:
        pose_path = os.path.splitext(ply)[0] + "_pose.txt"
        print(f"  {os.path.basename(ply)} ({os.path.getsize(ply)/1e9:.2f} GB)")
        pcd = o3d.io.read_point_cloud(ply)
        n_raw = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.voxel)
        pts = np.asarray(pcd.points, dtype=np.float64).copy()
        cols = (np.asarray(pcd.colors, dtype=np.float64).copy()
                if len(pcd.colors) == len(pcd.points) and len(pcd.colors) else None)
        print(f"    {n_raw:,} -> {len(pts):,} pts")

        if os.path.exists(pose_path):
            M = read_pose_matrix(pose_path)
        else:
            print("    WARNING: no pose file, using identity")
            M = np.eye(4)

        scans.append((pts, cols, M))
        del pcd   # release the Open3D object immediately

    print("\nfirst pose matrix as stored on disk:")
    print(np.array2string(scans[0][2], precision=4, suppress_small=True))

    # ---- cheap pass: pure numpy, no Open3D objects ------------------------
    variants = {
        "T":    (True,  False),   # transpose  <- expected correct
        "M":    (False, False),   # as-stored
        "Tinv": (True,  True),
        "Minv": (False, True),
    }

    print("\n" + "-" * 60)
    print(f"{'convention':<12}{'x':>8}{'y':>8}{'z':>8}{'vol m3':>11}  room?")
    print("-" * 60)

    results = {}
    for name, (tp, inv) in variants.items():
        merged = np.vstack([apply(p, to_transform(M, tp, inv))
                            for p, _, M in scans])
        ext = merged.max(axis=0) - merged.min(axis=0)
        vol = float(np.prod(ext))
        ok = looks_like_a_room(ext)
        results[name] = (ext, vol, ok)
        print(f"{name:<12}{ext[0]:>8.2f}{ext[1]:>8.2f}{ext[2]:>8.2f}"
              f"{vol:>11.1f}  {'YES' if ok else '-'}")
        del merged
    print("-" * 60)

    if args.force:
        pick = args.force
    else:
        plausible = [k for k, v in results.items() if v[2]]
        pool = plausible if plausible else list(results)
        pick = min(pool, key=lambda k: results[k][1])

    ext, vol, ok = results[pick]
    print(f"\nselected: {pick}")
    if not ok:
        print("WARNING: no convention gave room-like dimensions. The visit")
        print("may span a whole apartment rather than one room, or the poses")
        print("may be broken. Inspect visually before trusting this.")

    # ---- rebuild only the chosen variant ----------------------------------
    tp, inv = variants[pick]
    all_pts = np.vstack([apply(p, to_transform(M, tp, inv))
                         for p, _, M in scans])
    have_colors = all(c is not None for _, c, _ in scans)
    all_cols = np.vstack([c for _, c, _ in scans]) if have_colors else None

    if args.recenter:
        all_pts = all_pts - all_pts.min(axis=0)
        print("recentred: min corner moved to origin")

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(all_pts)
    if all_cols is not None:
        out.colors = o3d.utility.Vector3dVector(all_cols)
    out = out.voxel_down_sample(args.voxel)
    print(f"merged: {len(out.points):,} points")

    plane, inl = out.segment_plane(0.02, 3, 1000)
    print(f"largest plane normal: [{plane[0]:.3f} {plane[1]:.3f} "
          f"{plane[2]:.3f}]  ({len(inl):,} inliers)")
    print("Near +/-1 on one axis = that axis is up. "
          "Cross-check vs sky_direction in metadata_raw.csv.")

    o3d.io.write_point_cloud(args.out, out)
    print(f"\nwrote {args.out} - work from this, not the raw scans")


if __name__ == "__main__":
    main()
