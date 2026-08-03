#!/usr/bin/env python3
"""
Extract room-geometry ground truth from a REGISTERED ARKitScenes Faro cloud.

IMPORTANT: Faro scans are already registered. Apply NO transform from
_pose.txt (those record scanner tripod positions, not transforms).

Method:
  1. floor/ceiling from the two dominant z-histogram peaks
  2. bin (x,y); keep cells whose points span most of floor->ceiling.
     Full-height columns are walls. Furniture and far-field terrain
     are not, so both are removed by the same filter.
  3. Manhattan angle = rotation minimising wall-cell bounding-box area
  4. OBB extents from robust percentiles along the aligned axes
  5. floor area by morphological close + hole fill on the wall mask

Outputs JSON + a verification PNG.

    python extract_gt.py results/probe_484534 --visit 484534 \
        --out-dir results/gt
"""

import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage


def load_points(path):
    if os.path.isdir(path):
        fs = sorted(glob.glob(os.path.join(path, "*_RAW.npy")))
        if not fs:
            raise SystemExit(f"no *_RAW.npy in {path}")
        return np.vstack([np.load(f) for f in fs])
    if path.endswith(".npy"):
        return np.load(path)
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float64)


def floor_ceiling(z, bin_m=0.05):
    counts, edges = np.histogram(z, bins=int(max(20, (z.max()-z.min())/bin_m)))
    centres = 0.5 * (edges[:-1] + edges[1:])
    peaks = [(centres[i], counts[i]) for i in range(1, len(counts)-1)
             if counts[i] >= counts[i-1] and counts[i] >= counts[i+1]
             and counts[i] > counts.max() * 0.15]
    merged = []
    for c, n in sorted(peaks):
        if merged and c - merged[-1][0] < 0.30:
            if n > merged[-1][1]:
                merged[-1] = (c, n)
        else:
            merged.append((c, n))
    if len(merged) < 2:
        raise SystemExit("could not find two horizontal surfaces")
    top2 = sorted(sorted(merged, key=lambda p: -p[1])[:2])
    fl, ce = top2[0][0], top2[1][0]
    if not (1.8 < ce - fl < 5.0):
        print(f"  WARNING: implausible height {ce-fl:.2f}m "
              f"(peaks at {[round(m[0],2) for m in merged]})")
    return fl, ce


def wall_mask(pts, fl, ce, cell, span_frac):
    m = (pts[:, 2] > fl + 0.10) & (pts[:, 2] < ce - 0.10)
    P = pts[m]
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    nx = int(np.ceil((P[:, 0].max() - x0) / cell)) + 1
    ny = int(np.ceil((P[:, 1].max() - y0) / cell)) + 1
    ix = np.clip(((P[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    flat = ix * ny + iy

    zlo = np.full(nx * ny, np.inf); np.minimum.at(zlo, flat, P[:, 2])
    zhi = np.full(nx * ny, -np.inf); np.maximum.at(zhi, flat, P[:, 2])
    span = np.where(np.isfinite(zlo), zhi - zlo, 0.0).reshape(nx, ny)

    mask = span > span_frac * (ce - fl)
    return mask, (x0, y0), cell


def manhattan_angle(xy, step=0.5):
    best, best_area = 0.0, np.inf
    for a in np.arange(0, 90, step):
        t = np.radians(a)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = xy @ R.T
        lo, hi = np.percentile(q, [1, 99], axis=0)
        area = float(np.prod(hi - lo))
        if area < best_area:
            best, best_area = float(a), area
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="cache dir, .npy, or .ply (REGISTERED)")
    ap.add_argument("--visit", required=True)
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--span-frac", type=float, default=0.70,
                    help="fraction of room height a wall column must span")
    ap.add_argument("--close-cells", type=int, default=3,
                    help="morphological close radius, bridges doorways")
    ap.add_argument("--out-dir", default="results/gt")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pts = load_points(args.source)
    print(f"visit {args.visit}: {len(pts):,} points")

    fl, ce = floor_ceiling(pts[:, 2])
    h = ce - fl
    print(f"  floor z={fl:.2f}  ceiling z={ce:.2f}  height={h:.2f} m")

    mask, (x0, y0), cell = wall_mask(pts, fl, ce, args.cell, args.span_frac)
    print(f"  wall cells: {int(mask.sum()):,} of {mask.size:,}")
    if mask.sum() < 100:
        raise SystemExit("too few wall cells - lower --span-frac")

    ii, jj = np.nonzero(mask)
    xy = np.column_stack([x0 + ii * cell, y0 + jj * cell])

    ang = manhattan_angle(xy)
    t = np.radians(ang)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    q = xy @ R.T
    lo, hi = np.percentile(q, [1, 99], axis=0)
    L, W = sorted(hi - lo)[::-1]
    print(f"  manhattan angle {ang:.1f} deg -> OBB {L:.2f} x {W:.2f} m")

    st = np.ones((2 * args.close_cells + 1,) * 2, bool)
    closed = ndimage.binary_closing(mask, structure=st)
    filled = ndimage.binary_fill_holes(closed)
    area = float(filled.sum()) * cell ** 2
    obb_area = float(L * W)
    irreg = area / obb_area if obb_area else float("nan")
    print(f"  floor area {area:.2f} m2   OBB area {obb_area:.2f} m2   "
          f"irregularity {irreg:.3f}")
    if irreg > 1.02:
        print("  WARNING: fill leaked outside the OBB - raise --close-cells")

    rec = dict(visit_id=args.visit, n_points=int(len(pts)),
               floor_z=round(float(fl), 3), ceiling_z=round(float(ce), 3),
               ceiling_height_m=round(float(h), 3),
               manhattan_angle_deg=round(ang, 2),
               obb_length_m=round(float(L), 3), obb_width_m=round(float(W), 3),
               floor_area_m2=round(area, 3), obb_area_m2=round(obb_area, 3),
               irregularity=round(irreg, 4),
               n_wall_cells=int(mask.sum()),
               params=dict(cell=cell, span_frac=args.span_frac,
                           close_cells=args.close_cells))
    jpath = os.path.join(args.out_dir, f"{args.visit}.json")
    json.dump(rec, open(jpath, "w"), indent=2)
    print(f"  wrote {jpath}")

    # verification figure
    ext = [x0, x0 + mask.shape[0] * cell, y0, y0 + mask.shape[1] * cell]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(filled.T, origin="lower", extent=ext, cmap="Blues", alpha=.35)
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c="k", linewidths=0, label="wall cells")
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                        [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])
    ax.plot(*(corners @ R).T, "r-", lw=1.6, label=f"OBB {L:.2f}x{W:.2f} m")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"visit {args.visit}  h={h:.2f}m  "
                 f"area={area:.1f}m2  irreg={irreg:.2f}")
    ax.grid(alpha=.2, lw=.4); fig.tight_layout()
    ppath = os.path.join(args.out_dir, f"{args.visit}_gt.png")
    fig.savefig(ppath, dpi=150); plt.close(fig)
    print(f"  wrote {ppath}")


if __name__ == "__main__":
    main()
