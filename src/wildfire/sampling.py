"""Spatial sampling helpers: thinning positives, proposing negative centres."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def thin_points(xy: np.ndarray, min_dist: float, priority: np.ndarray | None = None) -> np.ndarray:
    """Greedy thinning. Returns indices of kept points, none closer than min_dist."""
    n = len(xy)
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(-priority) if priority is not None else np.arange(n)
    tree = cKDTree(xy)
    removed = np.zeros(n, dtype=bool)
    keep = []
    for i in order:
        if removed[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(xy[i], min_dist):
            removed[j] = True
    return np.array(keep, dtype=int)


def propose_negative_centres(anchor_xy, rng, n, min_d, max_d) -> np.ndarray:
    """Random points on an annulus [min_d, max_d] around the anchor (projected metres)."""
    ang = rng.uniform(0, 2 * np.pi, n)
    d = rng.uniform(min_d, max_d, n)
    return np.asarray(anchor_xy)[None, :] + np.c_[d * np.cos(ang), d * np.sin(ang)]


def far_from(points_xy: np.ndarray, tree: cKDTree | None, min_dist: float) -> np.ndarray:
    """Boolean mask: True where a candidate is farther than min_dist from every tree point."""
    if tree is None:
        return np.ones(len(points_xy), dtype=bool)
    d, _ = tree.query(points_xy)
    return d > min_dist
