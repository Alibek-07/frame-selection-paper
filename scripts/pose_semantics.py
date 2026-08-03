#!/usr/bin/env python3
"""
Final ARKitScenes pose diagnostic. Reads *_RAW.npy caches only - no PLY reads.

Two questions:
  1. Are the raw scans scanner-local (all centred near origin) or already
     registered into a shared frame (spread out, non-overlapping centres)?
  2. Which transform - including IDENTITY - maximises real overlap?

    python pose_semantics.py results/probe_484534 \
        --scan-dir ~/arkit_spike/laser_scanner_point_clouds/484534
"""

import argparse, glob, os, itertools
import numpy as np
from scipy.spatial import cKDTree


def read_pose(p):
    v = np.array([float(x) for x in open(p).read().replace(",", " ").split()])
    if v.size == 16:
        return v.reshape(4, 4)
    g = v.reshape(4, 3); M = np.eye(4); M[:3, :3] = g[:3, :]; M[3, :3] = g[3, :]
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    raws = sorted(glob.glob(os.path.join(args.cache_dir, "*_RAW.npy")))
    if not raws:
        raise SystemExit("no *_RAW.npy - run convention_test.py first")

    pts_l, poses, names = [], [], []
    for f in raws:
        stem = os.path.basename(f).replace("_RAW.npy", "")
        pts_l.append(np.load(f))
        poses.append(read_pose(os.path.join(args.scan_dir, f"{stem}_pose.txt")))
        names.append(stem)

    # ---- Q1: what frame are the raw points already in? --------------------
    print("RAW scan geometry (before any transform):")
    print(f"{'scan':<10}{'centre x':>10}{'centre y':>10}{'centre z':>10}"
          f"{'extent':>22}")
    for n, p in zip(names, pts_l):
        c = p.mean(axis=0); e = p.max(axis=0) - p.min(axis=0)
        print(f"{n:<10}{c[0]:>10.2f}{c[1]:>10.2f}{c[2]:>10.2f}"
              f"   {e[0]:>6.1f} x{e[1]:>6.1f} x{e[2]:>6.1f}")

    centres = np.array([p.mean(axis=0) for p in pts_l])
    spread = np.linalg.norm(centres - centres.mean(axis=0), axis=1).max()
    print(f"\nmax centre offset from mean: {spread:.2f} m")
    if spread < 1.5:
        print("=> all scans centred together: SCANNER-LOCAL, need transforms")
    else:
        print("=> centres spread apart: likely ALREADY REGISTERED, "
              "identity may be correct")

    print("\ntranslation vectors from _pose.txt (last row):")
    for n, M in zip(names, poses):
        print(f"  {n}: {np.array2string(M[3,:3], precision=2)}")

    # ---- Q2: which transform maximises tight overlap? ---------------------
    def variants(M):
        A, t = M[:3, :3], M[3, :3]
        return {"identity": (np.eye(3), np.zeros(3)),
                "AT_t":     (A.T, t),
                "A_t":      (A,   t),
                "AT_ATt":   (A.T, -A.T @ t),
                "A_At":     (A,   -A @ t)}

    rng = np.random.default_rng(0)
    print(f"\n{'transform':<12}{'median NN':>12}{'%<5cm':>9}{'%<2cm':>9}"
          "   read")
    print("-" * 58)
    best = (None, -1)
    for name in variants(poses[0]):
        placed = [p @ variants(M)[name][0].T + variants(M)[name][1]
                  for p, M in zip(pts_l, poses)]
        ds = []
        for i, j in itertools.combinations(range(len(placed)), 2):
            src = placed[i]
            if len(src) > args.sample:
                src = src[rng.choice(len(src), args.sample, replace=False)]
            d, _ = cKDTree(placed[j]).query(src, k=1)
            ds.append(d)
        d = np.concatenate(ds)
        med, f5, f2 = np.median(d)*100, (d < .05).mean()*100, (d < .02).mean()*100
        note = ("STRONG overlap" if f5 > 25 else
                "partial overlap" if f5 > 8 else "no real overlap")
        print(f"{name:<12}{med:>11.1f}cm{f5:>8.1f}%{f2:>8.1f}%   {note}")
        if f5 > best[1]:
            best = (name, f5)
    print("-" * 58)

    print(f"\nbest: {best[0]}  ({best[1]:.1f}% of points within 5cm)")
    if best[1] < 8:
        print("\nNo transform produces real overlap. The pose semantics are")
        print("not recoverable from the data in reasonable time.")
        print("=> SHELVE ARKitScenes. Evaluate ScanNet on Friday.")
    else:
        print(f"\n=> Use {best[0]}. Redo merge + ghosting with it.")


if __name__ == "__main__":
    main()
