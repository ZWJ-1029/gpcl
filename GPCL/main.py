"""End-to-end GPCL reproduction entry point."""

import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from config import CFG
from data import load_data
from evaluation import save_results
from graph import run_gpg
from train import predict_fcao, pretrain_encoder, train_regressor


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce GPCL on btrain.csv and fcao.csv")
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke/engineering run: approximate graph, 2 pretrain and 2 regression epochs, one seed.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Rebuild graph and pseudo-labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CFG
    if args.no_cache:
        cfg.cache_gpg = False
    if args.quick:
        cfg.graph_mode = "knn_approx"
        cfg.output_dir = cfg.output_dir / "quick"
        cfg.knn_neighbors = min(cfg.knn_neighbors, cfg.quick_knn_neighbors)
        cfg.knn_projection_dim = min(cfg.knn_projection_dim, 16)
        cfg.pmi_max_samples = min(cfg.pmi_max_samples, 1000)
        cfg.pretrain_epochs = cfg.quick_pretrain_epochs
        cfg.finetune_epochs = cfg.quick_finetune_epochs
        cfg.seeds = (42,)

    print(f"Python/PyTorch: {torch.__version__}; torch CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data = load_data(cfg)
    print(
        f"Loaded {len(data.train_targets)} train, {len(data.val_targets)} validation, "
        f"{len(data.test_targets)} test labels; "
        f"{len(data.train_candidate_windows)} training-only candidate windows."
    )
    gpg_data = data
    if args.quick:
        limit = min(cfg.quick_candidate_windows, len(data.train_candidate_windows))
        quick_anchors = data.train_anchor_indices[data.train_anchor_indices < limit]
        gpg_data = replace(
            data,
            train_stream=data.train_stream[: limit + cfg.window_size - 1],
            train_candidate_windows=data.train_candidate_windows[:limit],
            train_anchor_indices=quick_anchors,
            train_targets=data.train_targets[: len(quick_anchors)],
        )
        print(f"QUICK MODE: GPG uses only {limit} candidate windows; metrics are smoke-test results.")
    gpg = run_gpg(gpg_data, cfg, seed=cfg.seeds[0])
    print(
        f"GPG: PMI={gpg.pmi_method_used}, sigma={gpg.sigma:.6f}, "
        f"graph nnz={gpg.graph_nnz:,}, retained={gpg.retained_mask.sum():,}."
    )

    predictions = []
    model_dir = cfg.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed in cfg.seeds:
        encoder = pretrain_encoder(gpg_data, gpg, cfg, seed)
        regressor = train_regressor(encoder, data, cfg, seed)
        predictions.append(predict_fcao(regressor, data.test_windows, data, cfg))
        torch.save(
            {"seed": seed, "encoder": encoder.state_dict(), "regressor": regressor.state_dict()},
            model_dir / f"gpcl_seed_{seed}.pt",
        )

    metadata = {
        "graph_mode": cfg.graph_mode,
        "pmi_method": gpg.pmi_method_used,
        "graph_nnz": int(gpg.graph_nnz),
        "retained_samples": int(gpg.retained_mask.sum()),
        "candidate_samples": int(len(gpg.retained_mask)),
        "sigma": gpg.sigma,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "quick_smoke_test": bool(args.quick),
    }
    report = save_results(cfg.output_dir, data.test_targets, predictions, cfg.seeds, metadata)
    print(json.dumps(report["paper_style_mean_std"], indent=2))
    print(f"Results saved to: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()
