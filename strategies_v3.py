"""Sharpness-floored coverage + overlap measure. Import to register."""
import numpy as np
from harness import STRATEGIES, GEOMETRY, select_coverage, _circ_dist


def coverage_with_floor(pct):
    """Greedy bin-coverage among frames above the pct-th sharpness percentile.
    First real run: pure coverage picks frames at HALF uniform's sharpness
    (0.085 vs 0.181) because separated yaws fall in fast-pan segments."""
    def fn(feats, K, seed=0):
        thr = np.percentile(feats["sharpness"], pct) if pct > 0 else -np.inf
        pool = feats.index[feats["sharpness"] >= thr].to_numpy()
        if len(pool) < K:
            pool = feats.nlargest(K, "sharpness").index.to_numpy()
        return select_coverage(feats, K, seed=seed, pool=pool)
    return fn


for _p in (25, 50, 75):
    STRATEGIES[f"coverage_q{_p}"] = coverage_with_floor(_p)


def mean_consecutive_overlap(feats, idx):
    """Analytic inter-view overlap from yaw gaps, nominal 65 deg hFOV."""
    y = feats.loc[sorted(idx), "yaw"].to_numpy()
    if len(y) < 2:
        return 0.0
    return float(np.mean(np.clip(1.0 - _circ_dist(y[1:], y[:-1]) / np.radians(65.0), 0, 1)))


GEOMETRY["mean_consecutive_overlap"] = mean_consecutive_overlap
