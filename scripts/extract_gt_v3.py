#!/usr/bin/env python3
"""
Room-geometry GT from a REGISTERED ARKitScenes Faro cloud.

v3: measure the FLOOR, not the walls. The floor is one large contiguous
horizontal surface and its boundary IS the wall line. Furniture creates
interior holes and edge notches, both handled by morphology. Reconstructing
a closed wall contour from fragments (v1 span test, v2 ceiling contact)
proved unreliable in furnished rooms.

Apply NO transform - Faro scans are already registered.

    python extract_gt_v3.py results/probe_484534 --visit 484534 --diagnose
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


def min_area_rect(pix_xy, step=0.25):
    """Rotation minimising bbox area. Returns (angle_deg, length, width, lo, hi)."""
    best = None
    for a in np.arange(0, 90, step):
        t = np.radians(a)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        q = pix_xy @ R.T
        lo, hi = q.min(axis=0), q.max(axis=0)
        ar = float(np.prod(hi - lo))
        if best is None or ar < best[0]:
            best = (ar, float(a), lo, hi, R)
    ar, a, lo, hi, R = best
    e = hi - lo
    L, W = (e[0], e[1]) if e[0] >= e[1] else (e[1], e[0])
    return a, float(L), float(W), lo, hi, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--visit", required=True)
    ap.add_argument("--cell", type=float, default=0.05)
    ap.add_argument("--floor-band", type=float, nargs=2, default=[-0.05, 0.12],
                    help="z window around floor, metres")
    ap.add_argument("--close-cells", type=int, default=4,
                    help="bridge furniture shadows, cells (4 = 20cm)")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--out-dir", default="results/gt")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pts = load_points(args.source)
    fl, ce = floor_ceiling(pts[:, 2])
    h = ce - fl
    print(f"visit {args.visit}: {len(pts):,} pts")
    print(f"  floor {fl:.2f}  ceiling {ce:.2f}  height {h:.2f} m")

    lo_z, hi_z = fl + args.floor_band[0], fl + args.floor_band[1]
    F = pts[(pts[:, 2] > lo_z) & (pts[:, 2] < hi_z)][:, :2]
    print(f"  floor-band points: {len(F):,}")
    if len(F) < 2000:
        raise SystemExit("too few floor points - widen --floor-band")

    x0, y0 = F[:, 0].min(), F[:, 1].min()
    nx = int(np.ceil((F[:, 0].max() - x0) / args.cell)) + 1
    ny = int(np.ceil((F[:, 1].max() - y0) / args.cell)) + 1
    ix = np.clip(((F[:, 0] - x0) / args.cell).astype(int), 0, nx - 1)
    iy = np.clip(((F[:, 1] - y0) / args.cell).astype(int), 0, ny - 1)
    occ = np.zeros((nx, ny), bool)
    occ[ix, iy] = True
    print(f"  occupied floor cells: {int(occ.sum()):,} "
          f"({occ.sum()*args.cell**2:.1f} m2 raw)")

    st = np.ones((2 * args.close_cells + 1,) * 2, bool)
    solid = ndimage.binary_fill_holes(ndimage.binary_closing(occ, st))

    lab, n = ndimage.label(solid)
    if n == 0:
        raise SystemExit("no connected floor region")
    sizes = ndimage.sum(solid, lab, range(1, n + 1))
    room = (lab == (int(np.argmax(sizes)) + 1))
    area = float(room.sum()) * args.cell ** 2
    print(f"  components: {n}, largest = {area:.2f} m2")

    if args.diagnose:
        top = sorted(sizes, reverse=True)[:5]
        print("  top component areas (m2): " +
              ", ".join(f"{s*args.cell**2:.1f}" for s in top))

    ii, jj = np.nonzero(room)
    xy = np.column_stack([x0 + ii * args.cell, y0 + jj * args.cell])
    ang, L, W, rlo, rhi, R = min_area_rect(xy)
    obb_area = L * W
    irreg = area / obb_area if obb_area else float("nan")
    print(f"  manhattan {ang:.1f} deg -> OBB {L:.2f} x {W:.2f} m")
    print(f"  floor area {area:.2f} m2  OBB area {obb_area:.2f} m2  "
          f"irregularity {irreg:.3f}")
    if irreg < 0.45:
        print("  WARNING: very irregular - check the PNG, may be two rooms")

    rec = dict(visit_id=args.visit, n_points=int(len(pts)),
               floor_z=round(float(fl), 3), ceiling_z=round(float(ce), 3),
               ceiling_height_m=round(float(h), 3),
               manhattan_angle_deg=round(ang, 2),
               obb_length_m=round(L, 3), obb_width_m=round(W, 3),
               floor_area_m2=round(area, 3), obb_area_m2=round(obb_area, 3),
               irregularity=round(irreg, 4),
               n_floor_cells=int(room.sum()), n_components=int(n),
               params=dict(cell=args.cell, floor_band=args.floor_band,
                           close_cells=args.close_cells, method="floor_footprint"))
    jp = os.path.join(args.out_dir, f"{args.visit}.json")
    json.dump(rec, open(jp, "w"), indent=2)

    ext = [x0, x0 + nx * args.cell, y0, y0 + ny * args.cell]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(room.T, origin="lower", extent=ext, cmap="Blues", alpha=.5)
    ax.scatter(F[::4, 0], F[::4, 1], s=.4, c="k", linewidths=0, alpha=.5,
               label="floor points")
    cor = np.array([[rlo[0], rlo[1]], [rhi[0], rlo[1]], [rhi[0], rhi[1]],
                    [rlo[0], rhi[1]], [rlo[0], rlo[1]]])
    ax.plot(*(cor @ R).T, "r-", lw=1.8, label=f"OBB {L:.2f}x{W:.2f} m")
    pad = 1.0
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
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
