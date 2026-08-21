"""
The money figure: sharpness selection vs coverage selection, showing WHY.

Top row    - polar plot of where the selected views point, over the full
             yaw distribution of the capture. Sharpness clusters; coverage
             spreads.
Bottom row - top-down view of the reconstructed point cloud (gravity-aligned),
             with the fitted room rectangle solid and the dense reference
             dashed. The error is visible, not just tabulated.

    python -u make_figure.py                       # defaults below
    python -u make_figure.py --capture room2 --a clustered_low --b spread_high
"""
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

import strategies_v3
from pipeline_mapanything_v2 import build_feature_table, align_from_gravity
from harness import STRATEGIES, FACTORIAL

ap = argparse.ArgumentParser()
ap.add_argument("--capture", default="hikmatollah_apt")
ap.add_argument("--ext", default="MOV")
ap.add_argument("--a", default="sharpness", help="the failing strategy")
ap.add_argument("--b", default="coverage", help="the working strategy")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/fig_selection.pdf")
args = ap.parse_args()

ALL = {**STRATEGIES, **FACTORIAL}
model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
torch.set_grad_enabled(False)


def reconstruct(feats, idx):
    """Returns gravity/yaw-aligned points and the fitted rectangle."""
    preds = model.infer(load_images([feats.loc[i, "path"] for i in sorted(idx)]),
                        memory_efficient_inference=True, minibatch_size=1,
                        use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)
    pts, poses = [], []
    for p in preds:
        xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        m = p["mask"][0].float().cpu().numpy().reshape(-1) > 0.5
        pts.append(xyz[m])
        poses.append(p["camera_poses"][0].float().cpu().numpy())
    A = np.concatenate(pts)
    P = np.stack(poses)
    ups = -P[:, :3, 1]
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    up = ups.mean(0); up /= np.linalg.norm(up)
    q = A @ align_from_gravity(up, A[::20]).T
    lo, hi = np.percentile(q, [5, 95], axis=0)
    return q, lo, hi


feats = build_feature_table(f"data/{args.capture}.{args.ext}",
                            f"cache/{args.capture}", pool_size=200,
                            normalise_yaw=False)

print("reference (200 frames)...")
_, rlo, rhi = reconstruct(feats, list(feats.index))
ref = sorted([rhi[0] - rlo[0], rhi[1] - rlo[1]], reverse=True)
print(f"  reference {ref[0]:.2f} x {ref[1]:.2f} m")

panels = []
for name in (args.a, args.b):
    idx = ALL[name](feats, args.K, seed=args.seed)
    q, lo, hi = reconstruct(feats, idx)
    d = sorted([hi[0] - lo[0], hi[1] - lo[1]], reverse=True)
    err = float(np.mean([abs(x - y) / y for x, y in zip(d, ref)]))
    yaws = feats.loc[sorted(idx), "yaw"].to_numpy()
    cov = len(set((yaws / (2 * np.pi) * 36).astype(int))) / 36
    sharp = float(feats.loc[idx, "sharpness"].mean())
    panels.append(dict(name=name, q=q, lo=lo, hi=hi, dims=d, err=err,
                       yaws=yaws, cov=cov, sharp=sharp))
    print(f"  {name:14s} {d[0]:.2f} x {d[1]:.2f}  err {err:.1%}  "
          f"cov {cov:.3f}  sharp {sharp:.3f}")

# ------------------------------------------------------------------ figure --
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 9, "pdf.fonttype": 42})
BAD, GOOD, REFC = "#c0392b", "#1a6ea8", "#7f8c8d"
fig = plt.figure(figsize=(7.0, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.32, wspace=0.22)

allyaw = feats.yaw.to_numpy()
for c, pan in enumerate(panels):
    col = BAD if c == 0 else GOOD

    # --- top: where the selected views point -------------------------------
    ax = fig.add_subplot(gs[0, c], projection="polar")
    ax.hist(allyaw, bins=36, range=(0, 2 * np.pi), color="#dfe3e6",
            edgecolor="none", label="all 200 frames")
    for y in pan["yaws"]:
        ax.plot([y, y], [0, ax.get_ylim()[1]], color=col, lw=1.4, alpha=.85)
    ax.set_yticklabels([]); ax.set_theta_zero_location("N")
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["0°", "", "90°", "", "180°", "", "270°", ""])
    ax.set_title(f"{pan['name']}\ncoverage {pan['cov']:.2f} · "
                 f"sharpness {pan['sharp']:.2f}", pad=12, color=col)

    # --- bottom: the reconstruction ----------------------------------------
    ax = fig.add_subplot(gs[1, c])
    q = pan["q"]
    s = q[np.random.default_rng(0).choice(len(q), min(len(q), 40000), replace=False)]
    ax.scatter(s[:, 0], s[:, 1], s=.25, c="#34495e", linewidths=0, alpha=.20)
    lo, hi = pan["lo"], pan["hi"]
    ax.add_patch(Polygon([[lo[0], lo[1]], [hi[0], lo[1]],
                          [hi[0], hi[1]], [lo[0], hi[1]]],
                         fill=False, ec=col, lw=2.0, zorder=5))
    ax.add_patch(Polygon([[rlo[0], rlo[1]], [rhi[0], rlo[1]],
                          [rhi[0], rhi[1]], [rlo[0], rhi[1]]],
                         fill=False, ec=REFC, lw=1.3, ls="--", zorder=4))
    pad = 1.0
    ax.set_xlim(min(lo[0], rlo[0]) - pad, max(hi[0], rhi[0]) + pad)
    ax.set_ylim(min(lo[1], rlo[1]) - pad, max(hi[1], rhi[1]) + pad)
    ax.set_aspect("equal"); ax.grid(alpha=.18, lw=.4)
    ax.set_xlabel("x (m)")
    if c == 0:
        ax.set_ylabel("y (m)")
    ax.set_title(f"{pan['dims'][0]:.2f} × {pan['dims'][1]:.2f} m  "
                 f"({pan['err']:.0%} error)", color=col)

fig.text(0.5, 0.005,
         f"Dashed grey: dense reference over all 200 frames "
         f"({ref[0]:.2f} × {ref[1]:.2f} m). "
         f"Capture: {args.capture}, K={args.K}.",
         ha="center", fontsize=8, color="#555555")
os.makedirs("results", exist_ok=True)
fig.savefig(args.out, bbox_inches="tight")
fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"\nwrote {args.out} and .png")
