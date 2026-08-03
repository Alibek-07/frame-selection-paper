#!/usr/bin/env python3
"""
Register + merge ARKitScenes Faro laser scans into a single downsampled cloud.

The raw scans are ~1.8GB each. This loads them ONE AT A TIME, downsamples
immediately, then merges — so peak memory stays manageable.

Usage:
    python merge_laser_scans.py ~/arkit_spike/laser_scanner_point_clouds/484534 \
        --voxel 0.02 --out ~/arkit_spike/merged_484534.ply

Then inspect the printed bounding box. See SANITY CHECK below.
"""

import argparse
import glob
import os

import numpy as np
import open3d as o3d


def load_pose(pose_path, invert=False):
    """
    ARKitScenes _pose.txt: lines 0-2 = rotation matrix, line 3 = translation.
    Direction of the transform is not documented clearly — run with and
    without --invert and keep whichever produces a coherent room.
    """
    vals = np.loadtxt(pose_path)
    if vals.shape != (4, 3):
        raise ValueError(f"unexpected pose shape {vals.shape} in {pose_path}")

    T = np.eye(4)
    T[:3, :3] = vals[:3, :]
    T[:3, 3] = vals[3, :]
    return np.linalg.inv(T) if invert else T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir", help="laser_scanner_point_clouds/<visit_id>/")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="downsample size in metres (0.02 = 2cm)")
    ap.add_argument("--invert", action="store_true",
                    help="invert the pose transforms")
    ap.add_argument("--out", default="merged.ply")
    args = ap.parse_args()

    ply_files = sorted(
        p for p in glob.glob(os.path.join(args.scan_dir, "*.ply"))
        if not p.endswith("_pose.txt")
    )
    if not ply_files:
        raise SystemExit(f"no .ply files found in {args.scan_dir}")

    print(f"found {len(ply_files)} scans, voxel={args.voxel}m, invert={args.invert}\n")

    merged = o3d.geometry.PointCloud()

    for ply in ply_files:
        stem = os.path.splitext(ply)[0]
        pose_path = f"{stem}_pose.txt"

        size_gb = os.path.getsize(ply) / 1e9
        print(f"  loading {os.path.basename(ply)} ({size_gb:.2f} GB)...")
        pcd = o3d.io.read_point_cloud(ply)
        print(f"    {len(pcd.points):,} points")

        # Downsample BEFORE transforming — cheaper, and frees memory sooner.
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"    -> {len(pcd.points):,} after downsample")

        if os.path.exists(pose_path):
            pcd.transform(load_pose(pose_path, invert=args.invert))
        else:
            print(f"    WARNING: no pose file, leaving untransformed")

        merged += pcd
        del pcd

    merged = merged.voxel_down_sample(args.voxel)

    bbox = merged.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()

    print(f"\nmerged: {len(merged.points):,} points")
    print(f"bbox extent (m): "
          f"x={extent[0]:.2f}  y={extent[1]:.2f}  z={extent[2]:.2f}")

    print("\n--- SANITY CHECK ---")
    print("Room-sized (roughly 3-12m across, one axis ~2.2-3.5m for ceiling")
    print("height) means registration worked.")
    print("Tens of metres, or wildly asymmetric, means the transforms are")
    print("going the wrong way — re-run with --invert and compare.")

    # Largest plane should be the floor (or a wall). Its normal reveals up-axis.
    plane, inliers = merged.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane
    print(f"\nlargest plane normal: [{a:.3f} {b:.3f} {c:.3f}]  "
          f"({len(inliers):,} inliers)")
    print("A normal near +/-1 on one axis and ~0 on the others means that")
    print("axis is your up direction. Cross-check against sky_direction")
    print("in metadata_raw.csv.")

    o3d.io.write_point_cloud(args.out, merged)
    print(f"\nwrote {args.out}")
    print("Work from this file from now on — don't reload the raw scans.")


if __name__ == "__main__":
    main()
