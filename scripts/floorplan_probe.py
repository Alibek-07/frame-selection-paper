#!/usr/bin/env python3
"""
Probe a merged Faro cloud: sanity-check values, find storeys, and render a
top-down floor-plan projection so you can SEE how many rooms are in there.

Usage:
    python floorplan_probe.py ~/arkit_spike/merged_484534.ply \
        --out-dir results/probe_484534
"""

import argparse
import os

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--out-dir", default="results/probe")
    ap.add_argument("--slab-lo", type=float, default=1.0,
                    help="metres above floor for the wall slab")
    ap.add_argument("--slab-hi", type=float, default=1.8)
    ap.add_argument("--bin", type=float, default=0.05,
                    help="floor-plan pixel size in metres")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pcd.points, dtype=np.float64)
    print(f"loaded {len(pts):,} points")

    # ---- 1. resolve the numpy warnings once and for all -------------------
    finite = np.isfinite(pts).all(axis=1)
    n_bad = int((~finite).sum())
    print(f"non-finite points: {n_bad:,}"
          + ("  <- real problem" if n_bad else "  (warnings were spurious)"))
    pts = pts[finite]

    ext = pts.max(axis=0) - pts.min(axis=0)
    print(f"extent (m): x={ext[0]:.2f} y={ext[1]:.2f} z={ext[2]:.2f}")

    # ---- 2. how many storeys? z-histogram ---------------------------------
    z = pts[:, 2]
    counts, edges = np.histogram(z, bins=int(max(20, ext[2] / 0.05)))
    centres = 0.5 * (edges[:-1] + edges[1:])

    # horizontal surfaces show up as tall spikes
    thresh = counts.max() * 0.25
    peaks = [(centres[i], counts[i]) for i in range(1, len(counts) - 1)
             if counts[i] > thresh
             and counts[i] >= counts[i - 1] and counts[i] >= counts[i + 1]]
    # merge peaks within 30cm
    merged_peaks = []
    for c, n in sorted(peaks):
        if merged_peaks and c - merged_peaks[-1][0] < 0.30:
            if n > merged_peaks[-1][1]:
                merged_peaks[-1] = (c, n)
        else:
            merged_peaks.append((c, n))

    print(f"\nhorizontal surfaces at z = "
          + ", ".join(f"{c:.2f}m ({n:,} pts)" for c, n in merged_peaks))
    print("Two well-separated low peaks ~2.5-3m apart => two storeys.")

    plt.figure(figsize=(6, 4))
    plt.plot(centres, counts, lw=1)
    for c, _ in merged_peaks:
        plt.axvline(c, color="r", ls="--", lw=0.8)
    plt.xlabel("z (m)"); plt.ylabel("point count")
    plt.title("height histogram - spikes are floors/ceilings")
    plt.tight_layout()
    zpath = os.path.join(args.out_dir, "height_histogram.png")
    plt.savefig(zpath, dpi=130); plt.close()
    print(f"wrote {zpath}")

    # ---- 3. floor-plan projection per storey ------------------------------
    floor_z = merged_peaks[0][0] if merged_peaks else z.min()
    storeys = [c for c, _ in merged_peaks if c < z.min() + ext[2] - 1.0]
    if len(storeys) > 2:
        storeys = [storeys[0]]      # ceilings sneak in; just do the base
    print(f"\nrendering floor plan(s) from base z = {floor_z:.2f}m")

    for i, base in enumerate(storeys[:2]):
        lo, hi = base + args.slab_lo, base + args.slab_hi
        slab = pts[(z > lo) & (z < hi)]
        if len(slab) < 1000:
            print(f"  storey {i}: only {len(slab)} pts in slab, skipping")
            continue

        nx = int(np.ceil((pts[:, 0].max() - pts[:, 0].min()) / args.bin))
        ny = int(np.ceil((pts[:, 1].max() - pts[:, 1].min()) / args.bin))
        H, xe, ye = np.histogram2d(
            slab[:, 0], slab[:, 1], bins=[nx, ny],
            range=[[pts[:, 0].min(), pts[:, 0].max()],
                   [pts[:, 1].min(), pts[:, 1].max()]])

        plt.figure(figsize=(10, 10 * ny / max(nx, 1)))
        plt.imshow(np.log1p(H).T, origin="lower", cmap="inferno",
                   extent=[xe[0], xe[-1], ye[0], ye[-1]],
                   interpolation="nearest")
        plt.xlabel("x (m)"); plt.ylabel("y (m)")
        plt.title(f"floor plan, slab {lo:.2f}-{hi:.2f}m  ({len(slab):,} pts)")
        plt.grid(alpha=0.15, lw=0.4)
        plt.tight_layout()
        fpath = os.path.join(args.out_dir, f"floorplan_storey{i}.png")
        plt.savefig(fpath, dpi=150); plt.close()
        print(f"  wrote {fpath}   ({len(slab):,} pts in slab)")

    print("\nOpen the floor plan PNG. Bright lines are walls. Count the")
    print("enclosed rooms - that tells you how hard cropping one room is.")


if __name__ == "__main__":
    main()
