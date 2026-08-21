"""
Statistics with CAPTURE as the experimental unit.

WHY THIS REPLACES THE POOLED NUMBERS: the 324 reconstructions are 9 captures
x 4 cells x 3 seeds x 3 orderings. Seeds and orderings are REPEATED MEASURES
within capture, not independent samples. Pooling them and running a
correlation over n=324 pseudoreplicates: it inflates significance by orders of
magnitude. Effective n is 9.

So: collapse to per-capture cell means first, then do paired inference across
the 9 captures. Fewer stars, defensible ones.

    python run_stats_v2.py
"""
import numpy as np
import pandas as pd
from scipy import stats

d = pd.read_csv("results/factorial_orders.csv")
d = d[d.ok]
print(f"{len(d)} reconstructions from {d.capture.nunique()} captures")
print("=> experimental unit is the CAPTURE (n=9); seeds and orderings are "
      "repeated measures\n")

# ---------------------------------------------------------------- factorial --
cell = (d.groupby(["capture", "strategy"])
          .agg(err=("err", "mean"), conf=("conf", "mean"),
               degen=("degenerate", "mean"))
          .reset_index())
w = cell.pivot(index="capture", columns="strategy", values="err")
w = w[["spread_high", "spread_low", "clustered_high", "clustered_low"]]
print("=== per-capture cell means (the actual data, n=9) ===")
print(w.round(3).to_string())

spread_eff = ((w.clustered_high + w.clustered_low) / 2
              - (w.spread_high + w.spread_low) / 2)
qual_eff = ((w.spread_low + w.clustered_low) / 2
            - (w.spread_high + w.clustered_high) / 2)
inter = ((w.clustered_low - w.clustered_high)
         - (w.spread_low - w.spread_high))

print("\n=== main effects and interaction, paired across captures (n=9) ===")
for name, x in (("spread", spread_eff), ("quality", qual_eff),
                ("interaction", inter)):
    t, pt = stats.ttest_1samp(x, 0)
    try:
        _, pw = stats.wilcoxon(x)
    except ValueError:
        pw = np.nan
    ci = stats.t.interval(0.95, len(x) - 1, loc=x.mean(),
                          scale=stats.sem(x))
    print(f"  {name:12s} mean={x.mean():+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
          f"  t={t:+.2f} p={pt:.4f}  wilcoxon p={pw:.4f}"
          f"  ({(x > 0).sum()}/{len(x)} positive)")

print("\n  interaction reads: the cost of low quality is this much LARGER")
print("  under clustering than under spread.")

# the decisive comparison, paired
diff = w.clustered_high - w.spread_low
t, pt = stats.ttest_1samp(diff, 0)
_, pw = stats.wilcoxon(diff)
print(f"\n  clustered_high - spread_low: mean={diff.mean():+.3f} "
      f"t={t:+.2f} p={pt:.4f} wilcoxon p={pw:.4f} "
      f"({(diff > 0).sum()}/{len(diff)} captures favour spread_low)")

# --------------------------------------------------------------- confidence --
print("\n=== confidence vs error: within-capture, then across captures ===")
rows = []
for cap, x in d.groupby("capture"):
    r, p = stats.spearmanr(x.conf, x.err)
    rows.append((cap, r, p, len(x)))
    print(f"  {cap:22s} rho={r:+.3f}  p={p:.4f}  n={len(x)}")
rr = np.array([r for _, r, _, _ in rows])
t, pt = stats.ttest_1samp(rr, 0)
_, pw = stats.wilcoxon(rr)
sign_p = stats.binomtest((rr > 0).sum(), len(rr), 0.5).pvalue
print(f"\n  mean rho = {rr.mean():+.3f}  ({(rr > 0).sum()}/{len(rr)} positive)")
print(f"  across captures: t={t:+.2f} p={pt:.4f}  wilcoxon p={pw:.4f}  "
      f"sign test p={sign_p:.4f}")
print("  ^ THESE are the numbers to report, not the pooled n=324 correlation.")

# degeneracy vs confidence, at capture level
cd = (d.groupby(["capture", "degenerate"]).conf.mean().unstack())
if cd.shape[1] == 2:
    cd = cd.dropna()
    t, pt = stats.ttest_rel(cd[True], cd[False])
    _, pw = stats.wilcoxon(cd[True], cd[False])
    print(f"\n  confidence, degenerate vs valid, paired within capture "
          f"(n={len(cd)}):")
    print(f"    {cd[True].mean():.3f} vs {cd[False].mean():.3f}  "
          f"t={t:+.2f} p={pt:.4f}  wilcoxon p={pw:.4f}  "
          f"({(cd[True] > cd[False]).sum()}/{len(cd)} captures)")

# --------------------------------------------------------------------- VLM --
try:
    v = pd.read_csv("results/vlm_estimates.csv"); v = v[v.ok]
    print(f"\n=== VLM (n={v.capture.nunique()} captures) ===")
    vc = v.groupby(["capture", "strategy"]).vlm_err.mean().unstack()
    print(vc.round(3).to_string())
    rows = [stats.spearmanr(x.vlm_conf, x.vlm_err)[0]
            for _, x in v.groupby("capture") if x.vlm_conf.std() > 0]
    if rows:
        print(f"\n  within-capture spearman(vlm_conf, vlm_err): "
              f"mean {np.mean(rows):+.3f}  ({sum(r < 0 for r in rows)}/{len(rows)} negative)")
    bet = v.groupby("capture").vlm_err.mean()
    wit = v.groupby("capture").vlm_err.std()
    print(f"  between-capture SD of mean error: {bet.std():.3f}")
    print(f"  mean within-capture SD:            {wit.mean():.3f}")
    print(f"  ratio: {bet.std()/wit.mean():.1f}x  <- capture dominates strategy")
except FileNotFoundError:
    pass
