#!/usr/bin/env python3
"""
Determine the correct ARKitScenes _pose.txt convention by scan overlap.

Stored matrix M = [[A, 0], [t, 1]]. Candidate transforms:
    AT_t     R=A.T, tr=t          (what we assumed)
    A_t      R=A,   tr=t          (never tested)
    AT_ATt   R=A.T, tr=-A.T@t
    A_At     R=A,   tr=-A@t

Correct convention => scans overlap => small median nearest-neighbour
distance between pairs. Wrong => tens of cm.

    python convention_test.py ~/arkit_spike/laser_scanner_point_clouds/484534 \
        --cache results/probe_484534
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


def candidates(M):
    A = M[:3, :3].copy()
    t = M[3, :3].copy()
    return {
        "AT_t":    (A.T, t),
        "A_t":     (A,   t),
        "AT_ATt":  (A.T, -A.T @ t),
        "A_At":    (A,   -A @ t),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--cache", default=None,
                    help="dir with *_ds.npy caches (UNTRANSFORMED needed)")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    import open3d as o3d
    plys = sorted(glob.glob(os.path.join(args.scan_dir, "*.ply")))
    raws, poses = [], []
    for ply in plys:
        stem = os.path.splitext(ply)[0]
        raw_cache = os.path.join(args.cache or ".",
                                 os.path.basename(stem) + "_RAW.npy")
        if os.path.exists(raw_cache):
            pts = np.load(raw_cache)
            print(f"  {os.path.basename(ply)}: cached raw, {len(pts):,}")
        else:
            print(f"  {os.path.basename(ply)}: reading...")
            pcd = o3d.io.read_point_cloud(ply).voxel_down_sample(args.voxel)
            pts = np.asarray(pcd.points, dtype=np.float64).copy()
            del pcd
            os.makedirs(os.path.dirname(raw_cache) or ".", exist_ok=True)
            np.save(raw_cache, pts)
            print(f"    {len(pts):,} pts, cached UNTRANSFORMED")
        raws.append(pts)
        poses.append(read_pose(f"{stem}_pose.txt"))

    rng = np.random.default_rng(0)
    names = list(candidates(poses[0]).keys())

    print(f"\n{'convention':<10}{'median NN (cm)':>16}{'p90 (cm)':>12}   verdict")
    print("-" * 56)
    scores = {}
    for name in names:
        placed = []
        for pts, M in zip(raws, poses):
            R, tr = candidates(M)[name]
            placed.append(pts @ R.T + tr)

        meds = []
        for i, j in itertools.combinations(range(len(placed)), 2):
            tree = cKDTree(placed[j])
            src = placed[i]
            if len(src) > args.sample:
                src = src[rng.choice(len(src), args.sample, replace=False)]
            d, _ = tree.query(src, k=1)
            meds.append(d)
        allpair = np.concatenate(meds)
        med, p90 = np.median(allpair) * 100, np.percentile(allpair, 90) * 100
        scores[name] = med
        flag = ("<-- correct" if med < 8 else
                "plausible" if med < 20 else "wrong")
        print(f"{name:<10}{med:>16.1f}{p90:>12.1f}   {flag}")
    print("-" * 56)

    best = min(scores, key=scores.get)
    print(f"\nbest: {best}  (median NN {scores[best]:.1f} cm)")
    runner = sorted(scores.values())[1]
    if scores[best] > 8:
        print("WARNING: even the best convention shows poor overlap.")
        print("Either the scans genuinely cover disjoint rooms, or the")
        print("pose semantics are something none of these four capture.")
    elif runner / max(scores[best], 1e-9) < 1.5:
        print("WARNING: top two are close; overlap is not discriminating.")
    else:
        print("Clear winner. Re-run the merge and ghosting check with this.")


if __name__ == "__main__":
    main()
