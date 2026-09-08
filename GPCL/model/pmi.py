"""Partial mutual-information estimators for the 9-variable graph."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_regression

from config import Config


def _rank_uniform(x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    return np.column_stack([(rankdata(x[:, j]) - 0.5) / n for j in range(d)])


def conditional_ksg_pmi(x: np.ndarray, k: int, seed: int, max_samples: int) -> np.ndarray:
    """Estimate I(q_d;q_e | q_rest) with the Frenzel-Pompe/KSG estimator.

    Chebyshev radii are found once in the full 9-D joint space.  Counts in the
    two 8-D marginals and 7-D conditioning space then give conditional MI.
    """
    rng = np.random.default_rng(seed)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    u = _rank_uniform(x)
    n, d = u.shape
    if n <= k:
        raise ValueError("PMI sample count must exceed pmi_neighbors.")

    joint_tree = cKDTree(u)
    radii = joint_tree.query(u, k=k + 1, p=np.inf, workers=-1)[0][:, k]
    radii = np.nextafter(radii, 0.0)
    result = np.zeros((d, d), dtype=np.float64)
    for a in range(d):
        for b in range(a + 1, d):
            z_cols = [j for j in range(d) if j not in (a, b)]
            xz_cols = [j for j in range(d) if j != b]
            yz_cols = [j for j in range(d) if j != a]
            nz = cKDTree(u[:, z_cols]).query_ball_point(
                u[:, z_cols], radii, p=np.inf, return_length=True
            ) - 1
            nxz = cKDTree(u[:, xz_cols]).query_ball_point(
                u[:, xz_cols], radii, p=np.inf, return_length=True
            ) - 1
            nyz = cKDTree(u[:, yz_cols]).query_ball_point(
                u[:, yz_cols], radii, p=np.inf, return_length=True
            ) - 1
            estimate = digamma(k) + np.mean(
                digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1)
            )
            result[a, b] = result[b, a] = max(0.0, float(estimate))
    return result


def pairwise_ksg_approximation(
    x: np.ndarray, k: int, seed: int, max_samples: int
) -> np.ndarray:
    """Pairwise kNN MI approximation offered for practical sensitivity runs."""
    rng = np.random.default_rng(seed)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    d = x.shape[1]
    result = np.zeros((d, d), dtype=np.float64)
    for a in range(d):
        for b in range(a + 1, d):
            value = mutual_info_regression(
                x[:, [a]], x[:, b], n_neighbors=k, random_state=seed
            )[0]
            result[a, b] = result[b, a] = max(0.0, float(value))
    return result


def build_variable_graph(x: np.ndarray, cfg: Config, seed: int) -> tuple[np.ndarray, str]:
    method = cfg.pmi_estimator.lower()
    if method == "ksg_conditional":
        pmi = conditional_ksg_pmi(x, cfg.pmi_neighbors, seed, cfg.pmi_max_samples)
    elif method == "pairwise_ksg":
        pmi = pairwise_ksg_approximation(x, cfg.pmi_neighbors, seed, cfg.pmi_max_samples)
    else:
        raise ValueError("pmi_estimator must be 'ksg_conditional' or 'pairwise_ksg'.")
    if cfg.pmi_log_base_2:
        pmi /= np.log(2.0)

    retained = np.triu(pmi > cfg.pmi_threshold, 1).sum()
    used_method = method
    if retained == 0 and cfg.pmi_fallback_to_pairwise_if_empty and method == "ksg_conditional":
        pmi = pairwise_ksg_approximation(x, cfg.pmi_neighbors, seed, cfg.pmi_max_samples)
        if cfg.pmi_log_base_2:
            pmi /= np.log(2.0)
        used_method = "pairwise_ksg_fallback"

    adjacency = np.where(pmi > cfg.pmi_threshold, pmi, 0.0)
    np.fill_diagonal(adjacency, 0.0)
    if not np.any(adjacency):
        raise RuntimeError(
            "PMI threshold removed every variable edge. Lower pmi_threshold or "
            "set pmi_estimator='pairwise_ksg' in config.py."
        )
    return adjacency.astype(np.float32), used_method

