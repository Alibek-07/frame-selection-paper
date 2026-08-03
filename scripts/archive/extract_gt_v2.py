#!/usr/bin/env python3
"""
Extract room-geometry GT from a REGISTERED ARKitScenes Faro cloud.

v2: wall detection keyed on CEILING CONTACT, not total column span.
Furnished rooms occlude the base of most walls, so a full-height span
test finds only bare patches. Walls reach the ceiling; furniture and
far-field terrain do not.

Apply NO transform - Faro scans are already registered.

    python extract_gt_v2.py results/probe_484534 --visit 484534
    python extract_gt_v2.py results/probe_484534 --visit 484534 --diagnose
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
    c = 0.5 * (edges[:-1] + edges[1:])
    pk = [(c[i], counts[i]) for i in range(1, len(counts)-1)
          if counts[i] >= counts[i-1] and counts[i] >= counts[i+1]
          and counts[i] > counts.max() * 0.15]
    mg = []
    for a, n in sorted(pk):
        if mg and a - mg[-1][0] < 0.30:
            if n > mg[-1][1]:
                mg[-1] = (a, n)
        else:
            mg.append((a, n))
    if len(mg) < 2:
        raise SystemExit("could not find two horizontal surfaces")
    top2 = sorted(sorted(mg, key=lambda p: -p[1])[:2])
    return top2[0][0], top2[1][0]


def grid_stats(pts, fl, ce, cell):
    """Per-(x,y)-cell min/max z, excluding floor and ceiling planes."""
    m = (pts[:, 2] > fl + 0.10) & (pts[:, 2] < ce - 0.10)
    P = pts[m]
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    nx = int(np.ceil((P[:, 0].max() - x0) / cell)) + 1
    ny = int(np.ceil((P[:, 1].max() - y0) / cell)) + 1
    ix = np.clip(((P[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    flat = ix * ny + iy
    zlo = np.full(nx * ny, np.inf);  np.minimum.at(zlo, flat, P[:, 2])
    zhi = np.full(nx * ny, -np.inf); np.maximum.at(zhi, flat, P[:, 2])
    occ = np.isfinite(zlo)
    return (zlo.reshape(nx, ny), zhi.reshape(nx, ny),
            occ.reshape(nx, ny), (x0, y0), nx, ny)


def manhattan_angle(xy, step=0.5):
    best, ba = 0.0, np.inf
    for a in np.arange(0, 90, step):
        t = np.radians(a)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        lo, hi = np.percentile(xy @ R.T, [1, 99], axis=0)
        ar = float(np.prod(hi - lo))
        if ar < ba:
            best, ba = float(a), ar
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--visit", required=True)
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--ceiling-gap", type=float, default=0.35,
                    help="column top must be within this of the ceiling")
    ap.add_argument("--min-span", type=float, default=0.60,
                    help="minimum vertical span, rejects ceiling speckle")
    ap.add_argument("--close-cells", type=int, default=3)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--out-dir", default="results/gt")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pts = load_points(args.source)
    fl, ce = floor_ceiling(pts[:, 2])
    h = ce - fl
    print(f"visit {args.visit}: {len(pts):,} pts")
    print(f"  floor {fl:.2f}  ceiling {ce:.2f}  height {h:.2f} m")

    zlo, zhi, occ, (x0, y0), nx, ny = grid_stats(pts, fl, ce, args.cell)
    span = np.where(occ, zhi - zlo, 0.0)
    top_gap = np.where(occ, (ce - 0.10) - zhi, np.inf)

    if args.diagnose:
        s, g = span[occ], top_gap[occ]
        print(f"\n  occupied cells: {int(occ.sum()):,}")
        print("  span percentiles (m):   " +
              "  ".join(f"p{p}={np.percentile(s,p):.2f}"
                        for p in (50, 75, 90, 95, 99)))
        print("  gap-to-ceiling (m):     " +
              "  ".join(f"p{p}={np.percentile(g,p):.2f}"
                        for p in (1, 5, 10, 25, 50)))
        for gp in (0.15, 0.25, 0.35, 0.50):
            n = int(((top_gap < gp) & (span > args.min_span)).sum())
            print(f"    ceiling-gap<{gp:.2f} & span>{args.min_span}: "
                  f"{n:,} cells")
        print()

    mask = (top_gap < args.ceiling_gap) & (span > args.min_span)
    print(f"  wall cells: {int(mask.sum()):,}")
    if mask.sum() < 150:
        raise SystemExit("too few wall cells - run with --diagnose "
                         "and loosen --ceiling-gap / --min-span")

    ii, jj = np.nonzero(mask)
    xy = np.column_stack([x0 + ii * args.cell, y0 + jj * args.cell])
    ang = manhattan_angle(xy)
    t = np.radians(ang)
    R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    lo, hi = np.percentile(xy @ R.T, [1, 99], axis=0)
    L, W = sorted(hi - lo)[::-1]
    print(f"  manhattan {ang:.1f} deg -> OBB {L:.2f} x {W:.2f} m")

    st = np.ones((2 * args.close_cells + 1,) * 2, bool)
    filled = ndimage.binary_fill_holes(ndimage.binary_closing(mask, st))
    area = float(filled.sum()) * args.cell ** 2
    obb_area = float(L * W)
    irreg = area / obb_area if obb_area else float("nan")
    print(f"  floor area {area:.2f} m2  OBB area {obb_area:.2f} m2  "
          f"irregularity {irreg:.3f}")
    if irreg < 0.4:
        print("  WARNING: walls likely not closed - raise --close-cells")
    if irreg > 1.02:
        print("  WARNING: fill leaked outside OBB")

    rec = dict(visit_id=args.visit, n_points=int(len(pts)),
               floor_z=round(float(fl), 3), ceiling_z=round(float(ce), 3),
               ceiling_height_m=round(float(h), 3),
               manhattan_angle_deg=round(ang, 2),
               obb_length_m=round(float(L), 3), obb_width_m=round(float(W), 3),
               floor_area_m2=round(area, 3), obb_area_m2=round(obb_area, 3),
               irregularity=round(irreg, 4), n_wall_cells=int(mask.sum()),
               params=dict(cell=args.cell, ceiling_gap=args.ceiling_gap,
                           min_span=args.min_span,
                           close_cells=args.close_cells, method="ceiling_contact"))
    jp = os.path.join(args.out_dir, f"{args.visit}.json")
    json.dump(rec, open(jp, "w"), indent=2)

    ext = [x0, x0 + nx * args.cell, y0, y0 + ny * args.cell]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(filled.T, origin="lower", extent=ext, cmap="Blues", alpha=.35)
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c="k", linewidths=0, label="wall cells")
    cor = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]],
                    [lo[0], hi[1]], [lo[0], lo[1]]])
    ax.plot(*(cor @ R).T, "r-", lw=1.6, label=f"OBB {L:.2f}x{W:.2f} m")
    ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"visit {args.visit}  h={h:.2f}m  area={area:.1f}m2  "
                 f"irreg={irreg:.2f}")
    ax.grid(alpha=.2, lw=.4); fig.tight_layout()
    pp = os.path.join(args.out_dir, f"{args.visit}_gt.png")
    fig.savefig(pp, dpi=150); plt.close(fig)
    print(f"  wrote {jp} and {pp}")


if __name__ == "__main__":
    main()
