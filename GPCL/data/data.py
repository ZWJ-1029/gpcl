"""CSV validation, chronological partitioning and 60x9 window creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from config import Config


@dataclass
class DataBundle:
    train_stream: np.ndarray
    train_candidate_windows: np.ndarray
    train_anchor_indices: np.ndarray
    train_targets: np.ndarray
    val_windows: np.ndarray
    val_targets: np.ndarray
    test_windows: np.ndarray
    test_targets: np.ndarray
    process_min: np.ndarray
    process_max: np.ndarray
    target_min: float
    target_max: float
    variable_names: Tuple[str, ...]


def _read_numeric_csv(path, expected_cols: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if expected_cols is not None and frame.shape[1] != expected_cols:
        raise ValueError(f"{path} should have {expected_cols} columns, got {frame.shape[1]}.")
    converted = frame.apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        bad = int(converted.isna().sum().sum())
        raise ValueError(f"{path} contains {bad} missing or non-numeric values.")
    return converted


def load_data(cfg: Config) -> DataBundle:
    x_frame = _read_numeric_csv(cfg.process_csv, cfg.n_variables)
    y_frame = _read_numeric_csv(cfg.target_csv)
    if y_frame.shape[1] != 1:
        raise ValueError(f"{cfg.target_csv} should contain exactly one target column.")

    x_raw = x_frame.to_numpy(dtype=np.float64)
    y = y_frame.iloc[:, 0].to_numpy(dtype=np.float64)
    expected_rows = len(y) * cfg.window_size
    if len(x_raw) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} process rows ({len(y)} labels x "
            f"{cfg.window_size}), got {len(x_raw)}."
        )
    required_samples = cfg.test_start + cfg.n_test
    if len(y) < required_samples:
        raise ValueError(f"Need at least {required_samples} hourly samples, got {len(y)}.")

    # Only the earliest 1208 labeled intervals estimate normalization values.
    train_end_row = cfg.n_train * cfg.window_size
    x_train_raw = x_raw[:train_end_row]
    x_min = x_train_raw.min(axis=0)
    x_max = x_train_raw.max(axis=0)
    x_range = np.where(x_max > x_min, x_max - x_min, 1.0)
    x_norm = ((x_raw - x_min) / x_range).astype(np.float32)

    # The supplied file is already reconstructed.  No time-step rebuilding is
    # performed.  Hourly labeled samples are non-overlapping 60-row blocks.
    labeled_windows = x_norm.reshape(len(y), cfg.window_size, cfg.n_variables)
    train_stream = x_norm[:train_end_row]
    candidate = sliding_window_view(train_stream, cfg.window_size, axis=0)
    candidate = np.transpose(candidate, (0, 2, 1))  # (72421, 60, 9)
    candidate = np.ascontiguousarray(candidate, dtype=np.float32)
    anchor_indices = np.arange(0, len(candidate), cfg.window_size, dtype=np.int64)
    if len(anchor_indices) != cfg.n_train:
        raise AssertionError("Training anchor alignment is inconsistent with the paper split.")

    train_y = y[: cfg.n_train].astype(np.float32)
    val_slice = slice(cfg.val_start, cfg.val_start + cfg.n_val)
    test_slice = slice(cfg.test_start, cfg.test_start + cfg.n_test)
    target_min = float(train_y.min())
    target_max = float(train_y.max())

    return DataBundle(
        train_stream=train_stream,
        train_candidate_windows=candidate,
        train_anchor_indices=anchor_indices,
        train_targets=train_y,
        val_windows=np.ascontiguousarray(labeled_windows[val_slice], dtype=np.float32),
        val_targets=y[val_slice].astype(np.float32),
        test_windows=np.ascontiguousarray(labeled_windows[test_slice], dtype=np.float32),
        test_targets=y[test_slice].astype(np.float32),
        process_min=x_min,
        process_max=x_max,
        target_min=target_min,
        target_max=target_max,
        variable_names=tuple(str(c) for c in x_frame.columns),
    )
