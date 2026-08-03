#!/usr/bin/env python3
"""
Decide whether parallel wall lines are real surfaces or registration ghosting.

Renders each Faro scan in its own colour, overlaid, cropped to the occupied
region. Also caches per-scan downsampled PLYs so the 7GB read happens once.

Usage:
    python ghosting_check.py ~/arkit_spike/laser_scanner_point_clouds/484534 \
        --floor -203.11 --out-dir results/probe_484534
"""

import argparse, glob, os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_pose_matrix(p):
    v = np.array([float(x) for x in open(p).read().replace(",", " ").split()])
    if v.size == 16:
        return v.reshape(4, 4)
    g = v.reshape(4, 3); M = np.eye(4); M[:3, :3] = g[:3, :]; M[3, :3] = g[3, :]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--floor", type=float, required=True, help="floor z, e.g. -203.11")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--slab-lo", type=float, default=1.0)
    ap.add_argument("--slab-hi", type=float, default=1.8)
    ap.add_argument("--pct", type=float, default=1.0, help="percentile crop")
    ap.add_argument("--out-dir", default="results/ghosting")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plys = sorted(glob.glob(os.path.join(args.scan_dir, "*.ply")))
    lo, hi = args.floor + args.slab_lo, args.floor + args.slab_hi

    slabs = []
    for ply in plys:
        stem = os.path.splitext(ply)[0]
        cache = os.path.join(args.out_dir, os.path.basename(stem) + "_ds.npy")

        if os.path.exists(cache):
            pts = np.load(cache)
            print(f"  {os.path.basename(ply)}: cached, {len(pts):,} pts")
        else:
            print(f"  {os.path.basename(ply)}: reading "
                  f"({os.path.getsize(ply)/1e9:.2f} GB)...")
            pcd = o3d.io.read_point_cloud(ply)
            pcd = pcd.voxel_down_sample(args.voxel)
            pts = np.asarray(pcd.points, dtype=np.float64).copy()
            del pcd
            T = read_pose_matrix(f"{stem}_pose.txt").T          # convention T
            pts = pts @ T[:3, :3].T + T[:3, 3]
            np.save(cache, pts)
            print(f"    -> {len(pts):,} pts, cached")

        s = pts[(pts[:, 2] > lo) & (pts[:, 2] < hi)]
        slabs.append(s)
        print(f"    {len(s):,} pts in wall slab")

    allslab = np.vstack(slabs)
    xlo, xhi = np.percentile(allslab[:, 0], [args.pct, 100 - args.pct])
    ylo, yhi = np.percentile(allslab[:, 1], [args.pct, 100 - args.pct])
    print(f"\ncropped view: x [{xlo:.2f}, {xhi:.2f}]  y [{ylo:.2f}, {yhi:.2f}]")

    colors = ["#ff3b30", "#34c759", "#0a84ff", "#ffd60a"]
    plt.figure(figsize=(11, 11 * (yhi - ylo) / max(xhi - xlo, 0.1)))
    for i, s in enumerate(slabs):
        m = ((s[:, 0] > xlo) & (s[:, 0] < xhi) &
             (s[:, 1] > ylo) & (s[:, 1] < yhi))
        plt.scatter(s[m, 0], s[m, 1], s=0.12, c=colors[i % 4],
                    label=f"scan {i}", linewidths=0, alpha=0.65)
    plt.gca().set_aspect("equal")
    plt.legend(markerscale=40, loc="upper right")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("per-scan overlay: same-colour pairs = real walls, "
              "different-colour pairs = ghosting")
    plt.grid(alpha=0.2, lw=0.4)
    plt.tight_layout()
    p1 = os.path.join(args.out_dir, "per_scan_overlay.png")
    plt.savefig(p1, dpi=170); plt.close()
    print(f"wrote {p1}")

    # dominant wall orientation, for later Manhattan alignment
    from numpy.linalg import svd
    c = allslab[:, :2] - allslab[:, :2].mean(axis=0)
    ang = np.degrees(np.arctan2(*svd(c, full_matrices=False)[2][0][::-1]))
    print(f"\ndominant wall axis ~{ang % 90:.1f} deg off the x-axis")
    print("(you'll rotate by this before reading wall-to-wall distances)")


if __name__ == "__main__":
    main()
