"""All user-adjustable settings for the GPCL reproduction.

Edit only this file for normal experiments.  Paths are resolved relative to
the folder containing this file, so the project can be launched from any cwd.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    # Data and output
    process_csv: Path = ROOT / "btrain.csv"
    target_csv: Path = ROOT / "fcao.csv"
    output_dir: Path = ROOT / "outputs"
    window_size: int = 60
    n_variables: int = 9
    n_train: int = 1208
    first_gap_index: int = 1208       # zero-based sample 1209
    val_start: int = 1209             # zero-based sample 1210
    n_val: int = 200
    second_gap_index: int = 1409      # zero-based sample 1410
    test_start: int = 1410            # zero-based sample 1411
    n_test: int = 200

    # PMI variable graph (paper: theta=0.1)
    # "ksg_conditional" implements Eq. (1); "pairwise_ksg" is a faster,
    # numerically stable approximation because the paper omits its estimator.
    pmi_estimator: str = "ksg_conditional"
    pmi_threshold: float = 0.1
    pmi_neighbors: int = 5
    pmi_max_samples: int = 5000
    pmi_log_base_2: bool = False
    pmi_fallback_to_pairwise_if_empty: bool = True

    # Sample graph (paper: rho=0.05, sigma=median pairwise Frobenius distance)
    # "exact_blockwise" evaluates every sample pair as in Eqs. (3)-(6).
    # "knn_approx" is provided for machines that cannot hold ~142M sparse
    # entries; it applies the same equations to nearest-neighbour candidates.
    graph_mode: str = "exact_blockwise"
    graph_consistency_threshold: float = 0.05
    graph_block_size: int = 512
    sigma_sample_pairs: int = 250_000
    knn_neighbors: int = 2048
    knn_projection_dim: int = 32
    knn_algorithm: str = "kd_tree"
    knn_leaf_size: int = 40
    pair_eval_chunk: int = 25_000
    graph_device: str = "cuda"        # falls back to CPU if CUDA is absent
    cache_gpg: bool = True

    # Label propagation and confidence (paper: 10, alpha=0.3, beta=0.6)
    propagation_iterations: int = 10
    label_retention_alpha: float = 0.3
    propagation_mass_normalization: bool = True
    confidence_threshold: float = 0.6
    confidence_eps: float = 1e-8
    class_boundaries: Tuple[float, float] = (1.0, 1.5)

    # Encoder/GCCL (paper: filters 32/64, kernel 5, embedding 128, tau=.07)
    conv_channels: Tuple[int, int] = (32, 64)
    conv_kernel_size: int = 5
    max_pool_size: int = 2
    projection_hidden_dim: int = 256
    embedding_dim: int = 128
    temperature: float = 0.07
    batch_size: int = 64
    negatives_per_anchor: int = 32    # paper: Nb/2
    pretrain_epochs: int = 200
    pretrain_lr: float = 1e-3
    weight_decay: float = 1e-4

    # Frozen-encoder downstream regression (paper: lr=5e-4, 100 epochs)
    regression_hidden_dim: int = 64
    finetune_epochs: int = 100
    finetune_lr: float = 5e-4
    early_stopping_patience: Optional[int] = None

    # Repeated evaluation (paper: five independent runs)
    seeds: Tuple[int, ...] = field(default_factory=lambda: (42, 43, 44, 45, 46))
    num_workers: int = 0              # safest setting on Windows
    use_amp: bool = True
    deterministic: bool = True

    # --quick smoke preset (not a paper-result run)
    quick_candidate_windows: int = 6000
    quick_knn_neighbors: int = 64
    quick_pretrain_epochs: int = 2
    quick_finetune_epochs: int = 2


CFG = Config()
