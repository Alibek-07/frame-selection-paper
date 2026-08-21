"""
Money figure v2. Fixes over v1:
  - reconstructions REGISTERED so the reference rectangle coincides across
    panels (each recon lands in its own MapAnything frame; we translate both
    so the reference-box centre sits at the origin).
  - SHARED equal axis range on both bottom panels, so the slab is honestly
    small next to the room.
  - standard compass angle labels (0 top, 90 right, clockwise).

    python -u make_figure_v2.py --a sharpness --b uniform --out results/fig_main.pdf
    python -u make_figure_v2.py --a clustered_high --b spread_high --out results/fig_factorial.pdf
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
ap.add_argument("--a", default="sharpness")
ap.add_argument("--b", default="uniform")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/fig_main.pdf")
args = ap.parse_args()

ALL = {**STRATEGIES, **FACTORIAL}
model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
torch.set_grad_enabled(False)


def reconstruct(feats, idx):
    preds = model.infer(load_images([feats.loc[i, "path"] for i in sorted(idx)]),
                        memory_efficient_inference=True, minibatch_size=1,
                        use_amp=True, amp_dtype="bf16",
                        apply_mask=True, mask_edges=True)
    pts, poses = [], []
    for p in preds:
        xyz = p["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        m = p["mask"][0].float().cpu().numpy().reshape(-1) > 0.5
        pts.append(xyz[m]); poses.append(p["camera_poses"][0].float().cpu().numpy())
    A = np.concatenate(pts)
    P = np.stack(poses)
    ups = -P[:, :3, 1]; ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    up = ups.mean(0); up /= np.linalg.norm(up)
    q = A @ align_from_gravity(up, A[::20]).T
    lo, hi = np.percentile(q, [5, 95], axis=0)
    return q, lo, hi


def register(q, lo, hi):
    """Translate so the fitted box CENTRE is at the origin in x,y."""
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.0])
    return q - c, lo - c[:2] if len(lo) == 2 else lo - c, hi - c[:2] if len(hi) == 2 else hi - c


feats = build_feature_table(f"data/{args.capture}.{args.ext}",
                            f"cache/{args.capture}", pool_size=200,
                            normalise_yaw=False)

print("reference (200 frames)...")
_, rlo, rhi = reconstruct(feats, list(feats.index))
ref = sorted([rhi[0] - rlo[0], rhi[1] - rlo[1]], reverse=True)
rcx, rcy = (rlo[0] + rhi[0]) / 2, (rlo[1] + rhi[1]) / 2
rlo2 = np.array([rlo[0] - rcx, rlo[1] - rcy])
rhi2 = np.array([rhi[0] - rcx, rhi[1] - rcy])
print(f"  reference {ref[0]:.2f} x {ref[1]:.2f} m")

panels = []
for name in (args.a, args.b):
    idx = ALL[name](feats, args.K, seed=args.seed)
    q, lo, hi = reconstruct(feats, idx)
    d = sorted([hi[0] - lo[0], hi[1] - lo[1]], reverse=True)
    err = float(np.mean([abs(x - y) / y for x, y in zip(d, ref)]))
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    q = q.copy(); q[:, 0] -= cx; q[:, 1] -= cy
    lo2 = np.array([lo[0] - cx, lo[1] - cy]); hi2 = np.array([hi[0] - cx, hi[1] - cy])
    yaws = feats.loc[sorted(idx), "yaw"].to_numpy()
    cov = len(set((yaws / (2 * np.pi) * 36).astype(int))) / 36
    sharp = float(feats.loc[idx, "sharpness"].mean())
    panels.append(dict(name=name, q=q, lo=lo2, hi=hi2, dims=d, err=err,
                       yaws=yaws, cov=cov, sharp=sharp))
    print(f"  {name:14s} {d[0]:.2f} x {d[1]:.2f}  err {err:.1%}  cov {cov:.3f}")

# shared symmetric range covering both boxes + the reference + a margin
half = max(abs(rlo2).max(), abs(rhi2).max(),
          *[abs(p["lo"]).max() for p in panels],
          *[abs(p["hi"]).max() for p in panels]) + 1.0

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "pdf.fonttype": 42})
BAD, GOOD, REFC = "#c0392b", "#1a6ea8", "#7f8c8d"
fig = plt.figure(figsize=(7.0, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.4], hspace=0.30, wspace=0.20)
allyaw = feats.yaw.to_numpy()

for c, pan in enumerate(panels):
    col = BAD if c == 0 else GOOD

    ax = fig.add_subplot(gs[0, c], projection="polar")
    ax.hist(allyaw, bins=36, range=(0, 2 * np.pi), color="#dfe3e6", edgecolor="none")
    for y in pan["yaws"]:
        ax.plot([y, y], [0, ax.get_ylim()[1]], color=col, lw=1.3, alpha=.85)
    ax.set_yticklabels([])
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)   # compass
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels([])
    ax.set_title(f"{pan['name']}\ncoverage {pan['cov']:.2f} · "
                 f"sharpness {pan['sharp']:.2f}", pad=12, color=col)

    ax = fig.add_subplot(gs[1, c])
    q = pan["q"]
    s = q[np.random.default_rng(0).choice(len(q), min(len(q), 45000), replace=False)]
    ax.scatter(s[:, 0], s[:, 1], s=.3, c="#2c3e50", linewidths=0, alpha=.28)
    lo, hi = pan["lo"], pan["hi"]
    ax.add_patch(Polygon([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
                         fill=False, ec=col, lw=2.2, zorder=5))
    ax.add_patch(Polygon([[rlo2[0], rlo2[1]], [rhi2[0], rlo2[1]],
                          [rhi2[0], rhi2[1]], [rlo2[0], rhi2[1]]],
                         fill=False, ec=REFC, lw=1.3, ls="--", zorder=4))
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.grid(alpha=.18, lw=.4)
    ax.set_xlabel("x (m)")
    if c == 0:
        ax.set_ylabel("y (m)")
    ax.set_title(f"{pan['dims'][0]:.2f} × {pan['dims'][1]:.2f} m  "
                 f"({pan['err']:.0%} error)", color=col)

fig.text(0.5, 0.005,
         f"Dashed grey: dense reference over all 200 frames "
         f"({ref[0]:.2f} × {ref[1]:.2f} m), registered to a common centre. "
         f"{args.capture}, K={args.K}.",
         ha="center", fontsize=8, color="#555555")
os.makedirs("results", exist_ok=True)
fig.savefig(args.out, bbox_inches="tight")
fig.savefig(args.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"wrote {args.out} (+ .png)")
