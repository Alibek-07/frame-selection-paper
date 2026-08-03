"""
harness.py - frame-selection experiment harness.

Strategies operate on a FEATURE TABLE, not raw video:
    columns: frame_idx, sharpness, x, y, yaw
This decouples selection from I/O and makes the 2x2 factorial nearly free.

The pipeline is a protocol. MockPipeline validates the plumbing; swap in
the arbuz call on Friday by implementing ArbuzPipeline.reconstruct().

    from harness import run_sweep, STRATEGIES, MockPipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# feature table
# --------------------------------------------------------------------------

FEATURE_COLS = ["frame_idx", "sharpness", "x", "y", "yaw"]


def synthetic_features(n=600, seed=0):
    """Stand-in feature table: a handheld sweep with uneven dwell."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    # operator lingers in two spots -> yaw coverage is non-uniform
    yaw = 2 * np.pi * (t + 0.15 * np.sin(2 * np.pi * t)) % (2 * np.pi)
    # sharpness is WORSE while panning fast (the correlation that matters)
    speed = np.abs(np.gradient(yaw))
    sharp = 1.0 / (1.0 + 40 * speed) + rng.normal(0, 0.04, n)
    return pd.DataFrame({
        "frame_idx": np.arange(n),
        "sharpness": sharp,
        "x": 0.3 * np.cos(yaw) + rng.normal(0, .01, n),
        "y": 0.3 * np.sin(yaw) + rng.normal(0, .01, n),
        "yaw": yaw,
    })


def _circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def coverage_score(feats, idx):
    """Fraction of yaw bins hit by the selection. 0..1, higher = better."""
    bins = np.linspace(0, 2 * np.pi, 37)
    hit = np.histogram(feats.loc[idx, "yaw"], bins=bins)[0] > 0
    return float(hit.mean())


# --------------------------------------------------------------------------
# selection strategies:  fn(feats, K, seed) -> list[int]
# --------------------------------------------------------------------------

def select_uniform(feats, K, seed=0):
    n = len(feats)
    return list(np.linspace(0, n - 1, K).round().astype(int))


def select_random(feats, K, seed=0):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(len(feats), size=min(K, len(feats)), replace=False).tolist())


def select_sharpness_topk(feats, K, seed=0):
    return sorted(feats.nlargest(K, "sharpness")["frame_idx"].tolist())


def _farthest_point(feats, K, pool=None, seed=0):
    """Greedy farthest-point sampling in yaw space (circular)."""
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


def select_coverage(feats, K, seed=0):
    return _farthest_point(feats, K, seed=seed)


STRATEGIES = {
    "uniform":   select_uniform,
    "random":    select_random,
    "sharpness": select_sharpness_topk,
    "coverage":  select_coverage,
}

# --------------------------------------------------------------------------
# 2x2 factorial: spread x quality, for the mechanism claim
# --------------------------------------------------------------------------

def factorial(spread: bool, high_quality: bool):
    def fn(feats, K, seed=0):
        med = feats["sharpness"].median()
        pool = feats.index[(feats["sharpness"] >= med) if high_quality
                           else (feats["sharpness"] < med)].to_numpy()
        if len(pool) < K:
            pool = feats.index.to_numpy()
        if spread:
            return _farthest_point(feats, K, pool=pool, seed=seed)
        # clustered: contiguous run of frames within the pool
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
    Fake reconstruction for plumbing tests ONLY.

    Error falls with yaw coverage and rises with blur, and it fails outright
    below a coverage floor. Those relationships are ASSUMED, not measured -
    this proves nothing scientific. It exists so the harness, metric and
    aggregation can be validated before arbuz is wired in.
    """

    def __init__(self, gt_dims=(5.84, 3.43), noise=0.01, seed=0):
        self.gt = tuple(sorted(gt_dims, reverse=True))
        self.noise = noise
        self.seed = seed

    def reconstruct(self, feats, idx):
        if len(idx) < 5:
            return None                       # hard failure
        cov = coverage_score(feats, idx)
        if cov < 0.30:
            return None                       # too little of the room seen
        rng = np.random.default_rng(self.seed + int(np.sum(idx)) % 10_000)
        blur = 1.0 - float(feats.loc[idx, "sharpness"].mean())
        bias = 0.02 + 0.35 * (1 - cov) ** 2 + 0.05 * blur
        dims = [g * (1 + rng.normal(bias * rng.choice([-1, 1]), self.noise))
                for g in self.gt]
        return {"success": True, "dimensions": tuple(sorted(dims, reverse=True)),
                "coverage": cov}


class ArbuzPipeline:
    """Swap this in on Friday."""

    def reconstruct(self, feats, idx):
        raise NotImplementedError(
            "Wire to the arbuz scan pipeline: run it on the frames given by "
            "`idx` and return {'success': bool, 'dimensions': (L, W)}."
        )


# --------------------------------------------------------------------------
# metric  (see notes/metric.md - two-part, never collapsed)
# --------------------------------------------------------------------------

def dimension_error(pred, gt):
    p = sorted(pred, reverse=True)
    g = sorted(gt, reverse=True)
    return float(np.mean([abs(a - b) / b for a, b in zip(p, g)]))


def aggregate(df, tail=0.05):
    """success rate, and error conditional on success. Never merged."""
    out = []
    for (s, K), g in df.groupby(["strategy", "K"]):
        ok = g[g["success"]]
        out.append(dict(
            strategy=s, K=K, n=len(g),
            success_rate=len(ok) / len(g),
            mean_rel_err=ok["rel_err"].mean() if len(ok) else np.nan,
            median_rel_err=ok["rel_err"].median() if len(ok) else np.nan,
            frac_over_tail=(ok["rel_err"] > tail).mean() if len(ok) else np.nan,
        ))
    return pd.DataFrame(out).sort_values(["strategy", "K"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------

def run_sweep(captures, strategies, budgets, pipeline, seeds=(0, 1, 2)):
    """
    captures: {name: (feature_table, gt_dims)}
    returns long-format DataFrame, one row per (capture, strategy, K, seed)
    """
    rows = []
    for cname, (feats, gt) in captures.items():
        for sname, fn in strategies.items():
            for K in budgets:
                for seed in seeds:
                    idx = fn(feats, K, seed=seed)
                    res = pipeline.reconstruct(feats, idx)
                    ok = bool(res and res.get("success"))
                    rows.append(dict(
                        capture=cname, strategy=sname, K=K, seed=seed,
                        n_selected=len(idx), success=ok,
                        coverage=coverage_score(feats, idx),
                        rel_err=dimension_error(res["dimensions"], gt) if ok else np.nan,
                    ))
    return pd.DataFrame(rows)
