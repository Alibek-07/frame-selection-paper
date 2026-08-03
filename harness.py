"""
harness.py - frame-selection experiment harness (v2)

v2 changes:
  * select_coverage is now GREEDY MARGINAL BIN-COVERAGE over 36 yaw bins
    (10 deg each), matching the comparison heuristic in Wang et al., ICME 2014.
    Inherits the (1-1/e) guarantee for coverage functions (Nemhauser-Wolsey-
    Fisher 1978). FPS retained as a separate arm.
  * GEOMETRIC MEASURES recorded per run, so error can be regressed on
    geometry rather than asserted:
        angular_coverage        fraction of 10-deg yaw bins observed
        direction_entropy       normalised Shannon entropy of yaw histogram
        max_angular_gap         largest unobserved yaw wedge  <- missing wall
        mean_baseline           mean pairwise camera distance  <- NEGATIVE
                                CONTROL: irrelevant under monocular depth
        mean_adjacent_rotation  mean |yaw| change between selected frames
        mean_sharpness          the quality axis
  * correlate() reports which geometry predicts error.

NOTE ON THE MECHANISM: do NOT frame the causal chain as "reduced baseline
diversity degrades depth." There is no triangulation here - depth is predicted
per frame. The chain is about WHICH SURFACES ARE OBSERVED:
  quality ranking concentrates on slow-pan segments
    -> fewer distinct wall surfaces seen, or seen only at grazing incidence
    -> walls missing / poorly constrained in floor-plan assembly
    -> dimension error
plus a monocular-specific term: per-frame depth predictions disagree on scale,
and fewer viewpoints means less redundancy to reconcile them.
mean_baseline exists precisely to show it does NOT predict error.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

FEATURE_COLS = ["frame_idx", "sharpness", "x", "y", "yaw"]
N_BINS = 36                      # 10-degree bins, per Wang et al. ICME 2014
TWO_PI = 2 * np.pi


# --------------------------------------------------------------------------
# feature table
# --------------------------------------------------------------------------

def synthetic_features(n=600, seed=0):
    """Stand-in feature table: handheld sweep with uneven dwell."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    yaw = TWO_PI * (t + 0.15 * np.sin(TWO_PI * t)) % TWO_PI
    speed = np.abs(np.gradient(yaw))
    sharp = 1.0 / (1.0 + 40 * speed) + rng.normal(0, 0.04, n)
    return pd.DataFrame({
        "frame_idx": np.arange(n),
        "sharpness": sharp,
        "x": 0.3 * np.cos(yaw) + rng.normal(0, .01, n),
        "y": 0.3 * np.sin(yaw) + rng.normal(0, .01, n),
        "yaw": yaw,
    })


def _bin_of(yaw):
    return np.minimum((np.asarray(yaw) / TWO_PI * N_BINS).astype(int), N_BINS - 1)


def _circ_dist(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % TWO_PI
    return np.minimum(d, TWO_PI - d)


# --------------------------------------------------------------------------
# geometric measures
# --------------------------------------------------------------------------

def angular_coverage(feats, idx):
    """Fraction of 10-degree yaw bins observed. 0..1"""
    return float(len(set(_bin_of(feats.loc[idx, "yaw"]))) / N_BINS)


def direction_entropy(feats, idx):
    """Normalised Shannon entropy of the yaw histogram. Evenness, not presence."""
    c = np.bincount(_bin_of(feats.loc[idx, "yaw"]), minlength=N_BINS).astype(float)
    p = c[c > 0] / c.sum()
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(N_BINS))


def max_angular_gap(feats, idx):
    """Largest unobserved yaw wedge, radians. Big gap => a wall never seen."""
    y = np.sort(np.asarray(feats.loc[idx, "yaw"]) % TWO_PI)
    if len(y) < 2:
        return TWO_PI
    gaps = np.diff(y)
    wrap = (y[0] + TWO_PI) - y[-1]
    return float(max(gaps.max(), wrap))


def mean_baseline(feats, idx):
    """
    Mean pairwise camera-centre distance. NEGATIVE CONTROL.
    Under triangulation this drives depth accuracy; under learned monocular
    depth it should be uncorrelated with error. Reporting that is direct
    evidence for the regime claim.
    """
    P = feats.loc[idx, ["x", "y"]].to_numpy()
    if len(P) < 2:
        return 0.0
    d = [np.linalg.norm(P[i] - P[j]) for i, j in itertools.combinations(range(len(P)), 2)]
    return float(np.mean(d))


def mean_adjacent_rotation(feats, idx):
    """Mean |yaw| change between temporally consecutive selected frames."""
    y = feats.loc[sorted(idx), "yaw"].to_numpy()
    if len(y) < 2:
        return 0.0
    return float(np.mean(_circ_dist(y[1:], y[:-1])))


def mean_sharpness(feats, idx):
    return float(feats.loc[idx, "sharpness"].mean())


GEOMETRY = {
    "angular_coverage":       angular_coverage,
    "direction_entropy":      direction_entropy,
    "max_angular_gap":        max_angular_gap,
    "mean_baseline":          mean_baseline,      # negative control
    "mean_adjacent_rotation": mean_adjacent_rotation,
    "mean_sharpness":         mean_sharpness,
}

# backwards-compat alias
coverage_score = angular_coverage


# --------------------------------------------------------------------------
# selection strategies:  fn(feats, K, seed) -> list[int]
# --------------------------------------------------------------------------

def select_uniform(feats, K, seed=0):
    return list(np.linspace(0, len(feats) - 1, K).round().astype(int))


def select_random(feats, K, seed=0):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(feats), size=min(K, len(feats)), replace=False).tolist())


def select_sharpness_topk(feats, K, seed=0):
    return sorted(feats.nlargest(K, "sharpness")["frame_idx"].tolist())


def select_coverage(feats, K, seed=0, pool=None):
    """
    Greedy marginal bin-coverage. At each step add the frame covering the most
    uncovered 10-deg yaw bins; ties broken by max angular distance from the
    already-selected set. Once every bin is covered, keep spreading by
    farthest-point. This is the comparison heuristic of Wang et al. ICME 2014
    and, as a coverage function, gets the tight (1-1/e) greedy bound.
    """
    pool = np.asarray(pool if pool is not None else feats.index)
    if len(pool) <= K:
        return sorted(pool.tolist())

    yaw = feats.loc[pool, "yaw"].to_numpy()
    bins = _bin_of(yaw)
    rng = np.random.default_rng(seed)

    covered = np.zeros(N_BINS, bool)
    chosen: list[int] = []

    while len(chosen) < K:
        gain = (~covered[bins]).astype(int)                  # 1 if new bin
        if chosen:
            far = np.min(_circ_dist(yaw[:, None], yaw[chosen][None, :]), axis=1)
        else:
            far = rng.random(len(pool))                      # seeded first pick
        far = far.copy()
        far[chosen] = -1.0
        gain[chosen] = -1

        best = int(np.lexsort((far, gain))[-1])              # max gain, then max far
        chosen.append(best)
        covered[bins[best]] = True

    return sorted(pool[chosen].tolist())


def select_coverage_fps(feats, K, seed=0, pool=None):
    """Farthest-point sampling in yaw. Kept as a separate arm for comparison."""
    pool = np.asarray(pool if pool is not None else feats.index)
    if len(pool) <= K:
        return sorted(pool.tolist())
    yaw = feats.loc[pool, "yaw"].to_numpy()
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(pool)))]
    d = _circ_dist(yaw, yaw[chosen[0]])
    for _ in range(K - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, _circ_dist(yaw, yaw[nxt]))
    return sorted(pool[chosen].tolist())


STRATEGIES = {
    "uniform":      select_uniform,
    "random":       select_random,
    "sharpness":    select_sharpness_topk,
    "coverage":     select_coverage,
    "coverage_fps": select_coverage_fps,
}


# --------------------------------------------------------------------------
# 2x2 factorial: spread x quality  (the mechanism experiment)
# --------------------------------------------------------------------------

def factorial(spread: bool, high_quality: bool):
    def fn(feats, K, seed=0):
        med = feats["sharpness"].median()
        mask = (feats["sharpness"] >= med) if high_quality else (feats["sharpness"] < med)
        pool = feats.index[mask].to_numpy()
        if len(pool) < K:
            pool = feats.index.to_numpy()
        if spread:
            return select_coverage(feats, K, seed=seed, pool=pool)
        rng = np.random.default_rng(seed)
        start = int(rng.integers(0, max(1, len(pool) - K)))
        return sorted(pool[start:start + K].tolist())
    return fn


FACTORIAL = {
    "spread_high":    factorial(True,  True),
    "spread_low":     factorial(True,  False),
    "clustered_high": factorial(False, True),
    "clustered_low":  factorial(False, False),
}


# --------------------------------------------------------------------------
# pipeline protocol
# --------------------------------------------------------------------------

class MockPipeline:
    """
    Plumbing test ONLY. The coverage->accuracy relationship below is ASSUMED.
    Proves nothing scientific.
    """

    def __init__(self, gt_dims=(5.84, 3.43), noise=0.01, seed=0):
        self.gt = tuple(sorted(gt_dims, reverse=True))
        self.noise = noise
        self.seed = seed

    def reconstruct(self, feats, idx):
        if len(idx) < 5:
            return None
        cov = angular_coverage(feats, idx)
        if cov < 0.30:
            return None
        rng = np.random.default_rng(self.seed + int(np.sum(idx)) % 10_000)
        blur = 1.0 - mean_sharpness(feats, idx)
        bias = 0.02 + 0.35 * (1 - cov) ** 2 + 0.05 * blur
        dims = [g * (1 + rng.normal(bias * rng.choice([-1, 1]), self.noise))
                for g in self.gt]
        return {"success": True, "dimensions": tuple(sorted(dims, reverse=True))}


class ArbuzPipeline:
    """Swap this in. depth_backend lets us run recommendation #3."""

    def __init__(self, depth_backend="metric3d"):
        self.depth_backend = depth_backend

    def reconstruct(self, feats, idx):
        raise NotImplementedError(
            "Run the arbuz scan pipeline on the frames in `idx` and return "
            "{'success': bool, 'dimensions': (L, W)} in metres.")


# --------------------------------------------------------------------------
# metric  (notes/metric.md - two-part, never collapsed)
# --------------------------------------------------------------------------

def dimension_error(pred, gt):
    p = sorted(pred, reverse=True)
    g = sorted(gt, reverse=True)
    return float(np.mean([abs(a - b) / b for a, b in zip(p, g)]))


def aggregate(df, tail=0.05):
    out = []
    for keys, g in df.groupby(["strategy", "K"]):
        s, K = keys
        ok = g[g["success"]]
        out.append(dict(
            strategy=s, K=K, n=len(g),
            success_rate=len(ok) / len(g),
            mean_rel_err=ok["rel_err"].mean() if len(ok) else np.nan,
            median_rel_err=ok["rel_err"].median() if len(ok) else np.nan,
            frac_over_tail=(ok["rel_err"] > tail).mean() if len(ok) else np.nan,
        ))
    return pd.DataFrame(out).sort_values(["strategy", "K"]).reset_index(drop=True)


def correlate(df):
    """
    Which geometry predicts error? Spearman on successes, plus a
    success/failure split. This is the table behind the sentence
    "error tracks angular coverage but not sharpness".
    """
    ok = df[df["success"]]
    rows = []
    for name in GEOMETRY:
        if name not in df.columns:
            continue
        if len(ok) > 2 and ok[name].std() > 0 and ok["rel_err"].std() > 0:
            rho = ok[name].corr(ok["rel_err"], method="spearman")
        else:
            rho = np.nan
        rows.append(dict(
            measure=name,
            spearman_vs_err=rho,
            mean_when_success=df.loc[df["success"], name].mean(),
            mean_when_fail=df.loc[~df["success"], name].mean(),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def run_sweep(captures, strategies, budgets, pipeline, seeds=(0, 1, 2)):
    """captures: {name: (feature_table, gt_dims)} -> long-format DataFrame."""
    rows = []
    for cname, (feats, gt) in captures.items():
        for sname, fn in strategies.items():
            for K in budgets:
                for seed in seeds:
                    idx = fn(feats, K, seed=seed)
                    res = pipeline.reconstruct(feats, idx)
                    ok = bool(res and res.get("success"))
                    row = dict(
                        capture=cname, strategy=sname, K=K, seed=seed,
                        n_selected=len(idx), success=ok,
                        rel_err=dimension_error(res["dimensions"], gt) if ok else np.nan,
                        depth_backend=getattr(pipeline, "depth_backend", "mock"),
                    )
                    for gname, gfn in GEOMETRY.items():
                        row[gname] = gfn(feats, idx)
                    rows.append(row)
    return pd.DataFrame(rows)
