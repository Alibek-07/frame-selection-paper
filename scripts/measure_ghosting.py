#!/usr/bin/env python3
"""
Quantify per-scan wall offsets: constant => real wall faces / translation,
growing along the wall => rotational registration error.

Reads the *_ds.npy caches written by ghosting_check.py.

    python measure_ghosting.py results/probe_484534 --floor -203.11
    python measure_ghosting.py results/probe_484534 --floor -203.11 \
        --box 0.0 2.6 1.4 3.6          # xmin xmax ymin ymax
"""

import argparse, glob, os
import numpy as np


def fit_line(P):
    """Total least squares 2D line. Returns (point_on_line, unit_direction)."""
    c = P.mean(axis=0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    return c, u / np.linalg.norm(u)


def ransac_line(P, tol=0.04, iters=400, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(P), 2, replace=False)
        d = P[j] - P[i]
        L = np.linalg.norm(d)
        if L < 1.0:                       # want a long wall, not a scrap
            continue
        d = d / L
        n = np.array([-d[1], d[0]])
        inl = np.abs((P - P[i]) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = (P[i], d), int(inl.sum())
    return best, best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--floor", type=float, required=True)
    ap.add_argument("--slab-lo", type=float, default=1.0)
    ap.add_argument("--slab-hi", type=float, default=1.8)
    ap.add_argument("--box", nargs=4, type=float, metavar=("XMIN","XMAX","YMIN","YMAX"))
    ap.add_argument("--band", type=float, default=0.30,
                    help="how far from the reference line to gather points")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*_ds.npy")))
    if not files:
        raise SystemExit(f"no *_ds.npy in {args.cache_dir}")

    lo, hi = args.floor + args.slab_lo, args.floor + args.slab_hi
    scans = []
    for f in files:
        p = np.load(f)
        s = p[(p[:, 2] > lo) & (p[:, 2] < hi)][:, :2]
        if args.box:
            xa, xb, ya, yb = args.box
            s = s[(s[:,0]>xa)&(s[:,0]<xb)&(s[:,1]>ya)&(s[:,1]<yb)]
        scans.append(s)
        print(f"{os.path.basename(f):<24} {len(s):>7,} pts in region")

    combined = np.vstack(scans)
    if len(combined) < 200:
        raise SystemExit("too few points - widen --box")

    (p0, d), n_inl = ransac_line(combined)
    if p0 is None:
        raise SystemExit("no dominant line found - try --box on a flat wall")
    nrm = np.array([-d[1], d[0]])
    ang = np.degrees(np.arctan2(d[1], d[0])) % 180
    print(f"\nreference wall: {n_inl:,} inliers, bearing {ang:.1f} deg")

    # arc-length extent of the wall
    t_all = (combined - p0) @ d
    inl = np.abs((combined - p0) @ nrm) < 0.04
    t0, t1 = t_all[inl].min(), t_all[inl].max()
    print(f"wall spans {t1 - t0:.2f} m\n")

    # per-scan line fit within a band around the reference
    fits = {}
    for i, s in enumerate(scans):
        if len(s) == 0:
            continue
        m = np.abs((s - p0) @ nrm) < args.band
        if m.sum() < 60:
            print(f"scan {i}: only {int(m.sum())} pts near wall, skipped")
            continue
        fits[i] = fit_line(s[m])

    if len(fits) < 2:
        raise SystemExit("need >=2 scans on this wall; try another --box")

    # signed offset of each scan's line from the reference, sampled along wall
    samples = np.linspace(t0, t1, 5)
    print(f"{'scan':<6}" + "".join(f"{t:>9.2f}m" for t in samples) + "    spread")
    print("-" * (6 + 10 * len(samples) + 11))
    rows = {}
    for i, (c, u) in sorted(fits.items()):
        offs = []
        for t in samples:
            q = p0 + t * d                      # point on reference line
            offs.append(float((q - c) @ np.array([-u[1], u[0]])))
        rows[i] = np.array(offs)
        print(f"{i:<6}" + "".join(f"{o*100:>8.1f}cm" for o in offs)
              + f"{(max(offs)-min(offs))*100:>10.1f}cm")

    # pairwise: is the gap constant or growing?
    print("\npairwise separation along the wall:")
    keys = sorted(rows)
    verdicts = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            g = np.abs(rows[keys[a]] - rows[keys[b]])
            drift = g.max() - g.min()
            v = ("ROTATIONAL" if drift > 0.05 and drift > 0.5 * g.mean()
                 else "constant")
            verdicts.append(v)
            print(f"  scan {keys[a]} vs {keys[b]}: "
                  f"mean {g.mean()*100:5.1f}cm  drift {drift*100:5.1f}cm  -> {v}")

    print("\n--- VERDICT ---")
    if any(v == "ROTATIONAL" for v in verdicts):
        print("Gap grows along the wall => rotational misregistration.")
        print("Fix: refine with ICP before deriving GT.")
    else:
        gaps = [np.abs(rows[keys[a]] - rows[keys[b]]).mean()
                for a in range(len(keys)) for b in range(a+1, len(keys))]
        m = float(np.mean(gaps)) * 100
        print(f"Gap is constant at ~{m:.1f}cm.")
        if m < 3:
            print("Registration is tight. GT noise floor ~1-2cm. Proceed.")
        elif m < 20:
            print("Consistent with the two faces of a real wall, OR a small")
            print("constant translation. Either way it does NOT accumulate;")
            print(f"report ~{m/2:.0f}cm as the GT noise floor and proceed.")
        else:
            print("Too large for a wall thickness. Investigate before trusting GT.")


if __name__ == "__main__":
    main()
