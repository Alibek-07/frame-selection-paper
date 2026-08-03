#!/usr/bin/env python3
"""
Register + merge ARKitScenes Faro laser scans into a single downsampled cloud.

NOTE ON POSE FORMAT: DATA.md claims _pose.txt is 4x3 (lines 0-2 rotation,
line 3 translation), whitespace-delimited. In reality the files are
COMMA-delimited 4x4 homogeneous matrices with translation in the LAST ROW
(row-vector convention). The correct column-vector transform is therefore
the transpose. This script parses robustly and tests all four
transpose/invert combinations so you can see which one is right.

Usage:
    python merge_laser_scans_v2.py ~/arkit_spike/laser_scanner_point_clouds/484534 \
        --voxel 0.02 --out ~/arkit_spike/merged_484534.ply
"""

import argparse
import copy
import glob
import os

import numpy as np
import open3d as o3d


def read_pose_matrix(pose_path):
    """Parse a pose file tolerantly: any delimiter, 12 or 16 values."""
    raw = open(pose_path).read().replace(",", " ")
    vals = np.array([float(x) for x in raw.split()])

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
    T = M.T if transpose else M.copy()
    return np.linalg.inv(T) if invert else T


def looks_like_a_room(ext):
    """Two horizontal-ish axes of 2-20m, one vertical-ish axis of 2-4.5m."""
    s = sorted(ext)
    return 2.0 <= s[0] <= 4.5 and 2.0 <= s[1] <= 20.0 and s[2] <= 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir", help="laser_scanner_point_clouds/<visit_id>/")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--out", default="merged.ply")
    ap.add_argument("--force", choices=["T", "M", "Tinv", "Minv"],
                    help="skip auto-selection, force one convention")
    args = ap.parse_args()

    plys = sorted(glob.glob(os.path.join(args.scan_dir, "*.ply")))
    if not plys:
        raise SystemExit(f"no .ply files in {args.scan_dir}")

    print(f"{len(plys)} scans, voxel={args.voxel}m\n")

    # --- expensive pass: read + downsample once, keep in memory -------------
    clouds, poses = [], []
    for ply in plys:
        pose_path = os.path.splitext(ply)[0] + "_pose.txt"
        print(f"  {os.path.basename(ply)} ({os.path.getsize(ply)/1e9:.2f} GB)")
        pcd = o3d.io.read_point_cloud(ply)
        n_raw = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"    {n_raw:,} -> {len(pcd.points):,} pts")

        if not os.path.exists(pose_path):
            print("    WARNING: no pose file, using identity")
            poses.append(np.eye(4))
        else:
            poses.append(read_pose_matrix(pose_path))
        clouds.append(pcd)

    print("\nfirst pose matrix as stored on disk:")
    print(np.array2string(poses[0], precision=4, suppress_small=True))

    # --- cheap pass: try every convention on the downsampled clouds --------
    variants = {
        "T":    (True,  False),   # transpose        <- expected correct
        "M":    (False, False),   # as-stored
        "Tinv": (True,  True),
        "Minv": (False, True),
    }

    print("\n" + "-" * 58)
    print(f"{'convention':<12}{'x':>8}{'y':>8}{'z':>8}{'vol m3':>10}  room?")
    print("-" * 58)

    results = {}
    for name, (tp, inv) in variants.items():
        merged = o3d.geometry.PointCloud()
        for pcd, M in zip(clouds, poses):
            merged += copy.deepcopy(pcd).transform(to_transform(M, tp, inv))
        merged = merged.voxel_down_sample(args.voxel)
        ext = merged.get_axis_aligned_bounding_box().get_extent()
        vol = float(ext[0] * ext[1] * ext[2])
        ok = looks_like_a_room(ext)
        results[name] = (merged, ext, vol, ok)
        print(f"{name:<12}{ext[0]:>8.2f}{ext[1]:>8.2f}{ext[2]:>8.2f}"
              f"{vol:>10.1f}  {'YES' if ok else '-'}")
    print("-" * 58)

    if args.force:
        pick = args.force
    else:
        plausible = [k for k, v in results.items() if v[3]]
        pool = plausible if plausible else list(results)
        pick = min(pool, key=lambda k: results[k][2])

    merged, ext, vol, ok = results[pick]
    print(f"\nselected: {pick}   ({len(merged.points):,} points)")
    if not ok:
        print("WARNING: no convention produced room-like dimensions.")
        print("Scans may cover a whole apartment, not one room, or the")
        print("poses may be broken. Inspect visually before trusting this.")

    plane, inl = merged.segment_plane(0.02, 3, 1000)
    print(f"largest plane normal: [{plane[0]:.3f} {plane[1]:.3f} "
          f"{plane[2]:.3f}]  ({len(inl):,} inliers)")
    print("Near +/-1 on one axis = that axis is up. "
          "Cross-check vs sky_direction in metadata_raw.csv.")

    o3d.io.write_point_cloud(args.out, merged)
    print(f"\nwrote {args.out} - work from this, not the raw scans")


if __name__ == "__main__":
    main()
