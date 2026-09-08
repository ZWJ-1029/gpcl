"""GPG: paper Eqs. (3)-(9), sparse graph propagation and confidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from config import Config
from data import DataBundle
from pmi import build_variable_graph


@dataclass
class GPGResult:
    pseudo_labels: np.ndarray
    confidence: np.ndarray
    classes: np.ndarray
    retained_mask: np.ndarray
    variable_adjacency: np.ndarray
    sigma: float
    pmi_method_used: str
    graph_nnz: int


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable for graph construction; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def estimate_sigma(windows: np.ndarray, cfg: Config, seed: int) -> float:
    """Monte-Carlo median of pairwise Frobenius distances (paper's sigma)."""
    rng = np.random.default_rng(seed)
    count = min(cfg.sigma_sample_pairs, max(len(windows) * 4, 10_000))
    i = rng.integers(0, len(windows), size=count)
    j = rng.integers(0, len(windows), size=count)
    keep = i != j
    distance = np.linalg.norm(
        windows[i[keep]].reshape(keep.sum(), -1)
        - windows[j[keep]].reshape(keep.sum(), -1),
        axis=1,
    )
    sigma = float(np.median(distance))
    if not np.isfinite(sigma) or sigma <= 0:
        raise RuntimeError("Estimated Gaussian bandwidth is not positive.")
    return sigma


def _torch_similarity_block(
    xi: torch.Tensor,
    xj: torch.Tensor,
    adjacency: torch.Tensor,
    sigma: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian similarity and PMI consistency for one block pair."""
    t = xi.shape[1]
    fi = xi.flatten(1)
    fj = xj.flatten(1)
    dist2 = (fi.square().sum(1)[:, None] + fj.square().sum(1)[None, :] - 2 * fi @ fj.T)
    dist2 = dist2.clamp_min_(0.0)
    sim = torch.exp(-dist2 / (2.0 * sigma * sigma))

    g_parts = []
    for d in range(xi.shape[2]):
        ai = xi[:, :, d]
        aj = xj[:, :, d]
        d2 = ai.square().sum(1)[:, None] + aj.square().sum(1)[None, :] - 2 * ai @ aj.T
        g_parts.append((1.0 - torch.sqrt(d2.clamp_min(0.0) + 1e-12) / np.sqrt(t)).clamp(0.0, 1.0))
    g = torch.stack(g_parts, dim=-1)
    denominator = torch.triu(adjacency, diagonal=1).sum().clamp_min(1e-12)
    consistency = 0.5 * torch.einsum("ijd,de,ije->ij", g, adjacency, g) / denominator
    return sim, consistency


def build_exact_blockwise_graph(
    windows: np.ndarray, adjacency: np.ndarray, sigma: float, cfg: Config
) -> sp.csr_matrix:
    """All-pairs paper graph. This intentionally has the paper's high cost."""
    dev = _device(cfg.graph_device)
    a_t = torch.as_tensor(adjacency, dtype=torch.float32, device=dev)
    n = len(windows)
    block = cfg.graph_block_size
    n_blocks = (n + block - 1) // block
    rows, cols, values = [], [], []
    total = n_blocks * (n_blocks + 1) // 2
    with torch.inference_mode():
        with tqdm(total=total, desc="Exact sample graph blocks") as bar:
            for i0 in range(0, n, block):
                xi = torch.as_tensor(windows[i0 : i0 + block], device=dev)
                for j0 in range(i0, n, block):
                    xj = torch.as_tensor(windows[j0 : j0 + block], device=dev)
                    sim, consistency = _torch_similarity_block(
                        xi, xj, a_t, sigma, cfg.graph_consistency_threshold
                    )
                    mask = consistency > cfg.graph_consistency_threshold
                    if i0 == j0:
                        mask = torch.triu(mask, diagonal=1)
                    ij = mask.nonzero(as_tuple=False)
                    if len(ij):
                        r = (ij[:, 0] + i0).cpu().numpy().astype(np.int32)
                        c = (ij[:, 1] + j0).cpu().numpy().astype(np.int32)
                        v = sim[ij[:, 0], ij[:, 1]].cpu().numpy().astype(np.float32)
                        rows.extend((r, c))
                        cols.extend((c, r))
                        values.extend((v, v))
                    bar.update(1)
    if not values:
        raise RuntimeError("The sample-graph threshold removed every edge.")
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(values)
    graph = sp.coo_matrix((val, (row, col)), shape=(n, n), dtype=np.float32).tocsr()
    graph.sum_duplicates()
    graph.eliminate_zeros()
    return graph


def _pair_equations_numpy(
    windows: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    adjacency: np.ndarray,
    sigma: float,
    rho: float,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept_r, kept_c, kept_v = [], [], []
    denom = max(float(np.triu(adjacency, 1).sum()), 1e-12)
    for start in tqdm(range(0, len(row), chunk), desc="PMI-filtering kNN pairs"):
        stop = min(start + chunk, len(row))
        r, c = row[start:stop], col[start:stop]
        diff = windows[r] - windows[c]
        g = 1.0 - np.linalg.norm(diff, axis=1) / np.sqrt(windows.shape[1])
        g = np.clip(g, 0.0, 1.0)
        w = 0.5 * np.einsum("nd,de,ne->n", g, adjacency, g, optimize=True) / denom
        mask = w > rho
        if np.any(mask):
            d2 = np.einsum("ntd,ntd->n", diff[mask], diff[mask], optimize=True)
            kept_r.append(r[mask].astype(np.int32))
            kept_c.append(c[mask].astype(np.int32))
            kept_v.append(np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32))
    if not kept_v:
        raise RuntimeError("The approximate sample graph contains no retained edges.")
    return np.concatenate(kept_r), np.concatenate(kept_c), np.concatenate(kept_v)


def build_knn_approx_graph(
    windows: np.ndarray, adjacency: np.ndarray, sigma: float, cfg: Config, seed: int
) -> sp.csr_matrix:
    """Memory-bounded candidate graph; retained edges still use Eqs. (3)-(5)."""
    flat = windows.reshape(len(windows), -1)
    dim = min(cfg.knn_projection_dim, flat.shape[1], len(flat) - 1)
    projected = PCA(n_components=dim, svd_solver="randomized", random_state=seed).fit_transform(flat)
    k = min(cfg.knn_neighbors + 1, len(projected))
    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm=cfg.knn_algorithm,
        leaf_size=cfg.knn_leaf_size,
        n_jobs=-1,
    ).fit(projected)
    all_r, all_c = [], []
    query_block = max(64, cfg.graph_block_size)
    for start in tqdm(range(0, len(projected), query_block), desc="kNN candidate search"):
        stop = min(start + query_block, len(projected))
        indices = nn.kneighbors(projected[start:stop], return_distance=False)
        r = np.repeat(np.arange(start, stop, dtype=np.int64), k)
        c = indices.reshape(-1).astype(np.int64)
        keep = r != c
        all_r.append(r[keep])
        all_c.append(c[keep])
    row, col = np.concatenate(all_r), np.concatenate(all_c)
    row, col, val = _pair_equations_numpy(
        windows, row, col, adjacency, sigma,
        cfg.graph_consistency_threshold, cfg.pair_eval_chunk,
    )
    graph = sp.coo_matrix((val, (row, col)), shape=(len(windows), len(windows))).tocsr()
    graph = graph.maximum(graph.T).tocsr()
    graph.eliminate_zeros()
    return graph


def row_normalize(graph: sp.csr_matrix) -> sp.csr_matrix:
    sums = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.zeros_like(sums, dtype=np.float32)
    np.divide(1.0, sums, out=inv, where=sums > 0)
    return (sp.diags(inv) @ graph).tocsr()


def propagate_and_score(
    graph: sp.csr_matrix,
    anchors: np.ndarray,
    real_labels: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    p = row_normalize(graph)
    n = graph.shape[0]
    y0 = np.zeros(n, dtype=np.float64)
    m0 = np.zeros(n, dtype=np.float64)
    y0[anchors] = real_labels
    m0[anchors] = 1.0
    numerator, mass = y0.copy(), m0.copy()
    alpha = cfg.label_retention_alpha
    for _ in range(cfg.propagation_iterations):
        numerator = (1.0 - alpha) * (p @ numerator) + alpha * y0
        if cfg.propagation_mass_normalization:
            mass = (1.0 - alpha) * (p @ mass) + alpha * m0
    if cfg.propagation_mass_normalization:
        pseudo = np.divide(
            numerator, mass, out=np.full(n, float(np.mean(real_labels))), where=mass > cfg.confidence_eps
        )
    else:
        pseudo = numerator
    pseudo[anchors] = real_labels

    anchor_graph = p[:, anchors]
    direct_mass = np.asarray(anchor_graph.sum(axis=1)).ravel()
    direct_value = np.asarray(anchor_graph @ real_labels).ravel()
    y_ref = np.divide(direct_value, direct_mass, out=pseudo.copy(), where=direct_mass > 0)
    residual = pseudo[anchors] - y_ref[anchors]
    sigma_err = max(float(np.std(residual, ddof=1)), cfg.confidence_eps)
    confidence = np.exp(-np.abs(pseudo - y_ref) / sigma_err)
    confidence[direct_mass <= 0] = 0.0
    confidence[anchors] = 1.0
    return pseudo.astype(np.float32), confidence.astype(np.float32)


def discretize(labels: np.ndarray, cfg: Config) -> np.ndarray:
    low, high = cfg.class_boundaries
    return np.where(labels < low, 0, np.where(labels <= high, 1, 2)).astype(np.int64)


def _cache_key(cfg: Config, data: DataBundle) -> str:
    payload = {
        "shape": data.train_candidate_windows.shape,
        "data_sample": np.asarray(data.train_candidate_windows[[0, -1]]).round(7).tolist(),
        "pmi": [cfg.pmi_estimator, cfg.pmi_threshold, cfg.pmi_neighbors, cfg.pmi_max_samples],
        "graph": [cfg.graph_mode, cfg.graph_consistency_threshold, cfg.knn_neighbors],
        "prop": [cfg.propagation_iterations, cfg.label_retention_alpha,
                 cfg.propagation_mass_normalization, cfg.confidence_threshold],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_gpg(data: DataBundle, cfg: Config, seed: int = 42) -> GPGResult:
    cache_dir = cfg.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(cfg, data)
    graph_path = cache_dir / f"sample_graph_{key}.npz"
    arrays_path = cache_dir / f"gpg_{key}.npz"
    if cfg.cache_gpg and graph_path.exists() and arrays_path.exists():
        graph = sp.load_npz(graph_path).tocsr()
        z = np.load(arrays_path, allow_pickle=False)
        return GPGResult(
            pseudo_labels=z["pseudo_labels"], confidence=z["confidence"],
            classes=z["classes"], retained_mask=z["retained_mask"].astype(bool),
            variable_adjacency=z["variable_adjacency"], sigma=float(z["sigma"]),
            pmi_method_used=str(z["pmi_method_used"]), graph_nnz=graph.nnz,
        )

    adjacency, method = build_variable_graph(data.train_stream, cfg, seed)
    sigma = estimate_sigma(data.train_candidate_windows, cfg, seed)
    if cfg.graph_mode == "exact_blockwise":
        graph = build_exact_blockwise_graph(data.train_candidate_windows, adjacency, sigma, cfg)
    elif cfg.graph_mode == "knn_approx":
        graph = build_knn_approx_graph(data.train_candidate_windows, adjacency, sigma, cfg, seed)
    else:
        raise ValueError("graph_mode must be 'exact_blockwise' or 'knn_approx'.")
    pseudo, confidence = propagate_and_score(
        graph, data.train_anchor_indices, data.train_targets, cfg
    )
    classes = discretize(pseudo, cfg)
    retained = confidence > cfg.confidence_threshold
    retained[data.train_anchor_indices] = True
    if retained.sum() <= len(data.train_anchor_indices):
        raise RuntimeError("Confidence filtering retained no pseudo-labeled samples.")

    if cfg.cache_gpg:
        sp.save_npz(graph_path, graph, compressed=True)
        np.savez_compressed(
            arrays_path,
            pseudo_labels=pseudo,
            confidence=confidence,
            classes=classes,
            retained_mask=retained,
            variable_adjacency=adjacency,
            sigma=np.float64(sigma),
            pmi_method_used=np.asarray(method),
        )
    return GPGResult(pseudo, confidence, classes, retained, adjacency, sigma, method, graph.nnz)
