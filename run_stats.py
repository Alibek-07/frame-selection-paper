"""
Two things the paper needs before drafting.

(A) CONFIDENCE STATISTICS. The draft claims confidence and accuracy
    "anti-correlate". Four cell means do not support that word. This computes
    Pearson/Spearman across all individual reconstructions, plus a
    degenerate-vs-normal test, plus within-capture partial correlations so the
    claim is not just between-capture variance in disguise.

(B) REFERENCE SELF-CONSISTENCY. Our reference is a dense reconstruction from
    the same model, so reported errors measure agreement, not accuracy. We
    cannot fix that without ground truth, but we can BOUND it: reconstruct
    from two disjoint halves of the 200-frame pool (even vs odd indices, 100
    frames each, both densely covering the room) and measure how much they
    disagree. That disagreement is the noise floor of the reference itself.

    python -u run_stats.py 2>&1 | grep -v "Checking"
"""
import glob, os
import numpy as np
import pandas as pd
from scipy import stats

import strategies_v3
from pipeline_mapanything_v2 import build_feature_table, MapAnythingPipeline

print("=" * 68)
print("(A) CONFIDENCE STATISTICS")
print("=" * 68)

g = pd.read_csv("results/factorial_orders.csv")
g = g[g.ok]
print(f"n = {len(g)} reconstructions\n")

for nm, fn in (("pearson", stats.pearsonr), ("spearman", stats.spearmanr)):
    r, p = fn(g.conf, g.err)
    print(f"  {nm:9s}(conf, err) = {r:+.3f}   p = {p:.3e}")

print("\n  confidence by degeneracy:")
print(g.groupby("degenerate")["conf"].agg(["mean", "std", "count"]).round(3).to_string())
if g.degenerate.nunique() == 2:
    a = g[g.degenerate].conf
    b = g[~g.degenerate].conf
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    print(f"  Mann-Whitney (degenerate conf > normal): U={u:.0f}  p={p:.3e}")

# within-capture: strips between-capture variance from the correlation
print("\n  WITHIN-capture spearman(conf, err):")
rs = []
for cap, x in g.groupby("capture"):
    if len(x) > 5 and x.conf.std() > 0:
        r, p = stats.spearmanr(x.conf, x.err)
        rs.append(r)
        print(f"    {cap:22s} r={r:+.3f}  p={p:.3f}  n={len(x)}")
if rs:
    print(f"    mean within-capture r = {np.mean(rs):+.3f} "
          f"({sum(r < 0 for r in rs)}/{len(rs)} negative)")

# same treatment for the VLM coverage claim
if os.path.exists("results/vlm_estimates.csv"):
    v = pd.read_csv("results/vlm_estimates.csv")
    v = v[v.ok]
    print("\n  VLM: WITHIN-capture spearman(angular_coverage, vlm_err):")
    rs = []
    for cap, x in v.groupby("capture"):
        if len(x) > 5 and x.angular_coverage.std() > 0:
            r, p = stats.spearmanr(x.angular_coverage, x.vlm_err)
            rs.append(r)
            print(f"    {cap:22s} r={r:+.3f}  p={p:.3f}  n={len(x)}")
    if rs:
        print(f"    mean within-capture r = {np.mean(rs):+.3f} "
              f"({sum(r < 0 for r in rs)}/{len(rs)} negative)")
        print("    (if the pooled -0.369 survives here, the claim is clean;")
        print("     if it collapses, it was between-capture variance)")

print("\n" + "=" * 68)
print("(B) REFERENCE SELF-CONSISTENCY")
print("=" * 68)

pipe = MapAnythingPipeline()
rows = []
for vid in sorted(glob.glob("data/*.MOV") + glob.glob("data/*.mp4")):
    cap = os.path.splitext(os.path.basename(vid))[0]
    feats = build_feature_table(vid, f"cache/{cap}", pool_size=200,
                                normalise_yaw=False)
    idx = list(feats.index)
    halves = {"even": idx[0::2], "odd": idx[1::2]}
    out = {}
    for nm, sub in halves.items():
        r = pipe.reconstruct(feats, sub)
        if not (r and r.get("success")):
            print(f"{cap}/{nm}: FAILED"); continue
        out[nm] = sorted(r["dimensions"], reverse=True)
    if len(out) == 2:
        e, o = out["even"], out["odd"]
        disagree = float(np.mean([abs(a - b) / ((a + b) / 2)
                                 for a, b in zip(e, o)]))
        rows.append(dict(capture=cap,
                         even_L=e[0], even_W=e[1], odd_L=o[0], odd_W=o[1],
                         disagreement=disagree))
        print(f"{cap:22s} even {e[0]:.2f}x{e[1]:.2f}  odd {o[0]:.2f}x{o[1]:.2f}"
              f"   disagreement {disagree:.1%}")

if rows:
    d = pd.DataFrame(rows)
    d.to_csv("results/reference_consistency.csv", index=False)
    m = d.disagreement.mean()
    print(f"\n  mean reference disagreement: {m:.1%}  (n={len(d)} captures)")
    print(f"  -> the reference's own noise floor is ~{m:.1%}.")
    print(f"     Reported errors of 12-42% are {0.13/max(m,1e-9):.0f}-"
          f"{0.42/max(m,1e-9):.0f}x this, so they are signal, not reference noise.")
