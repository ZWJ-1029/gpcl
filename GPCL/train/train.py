"""GCCL pretraining, frozen-encoder regression and inference."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from config import Config
from data import DataBundle
from graph import GPGResult
from losses import confidence_weighted_infonce
from models import FrozenGPCLRegressor, GPCL1DEncoder


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _grad_scaler(enabled: bool):
    # torch.amp.GradScaler is the current API; the fallback keeps CPU smoke
    # tests and older local PyTorch installations usable.
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(*arrays: np.ndarray, batch_size: int, shuffle: bool, cfg: Config) -> DataLoader:
    tensors = [torch.from_numpy(np.ascontiguousarray(a)) for a in arrays]
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def pretrain_encoder(data: DataBundle, gpg: GPGResult, cfg: Config, seed: int) -> GPCL1DEncoder:
    set_seed(seed, cfg.deterministic)
    device = select_device()
    idx = np.flatnonzero(gpg.retained_mask)
    loader = _loader(
        data.train_candidate_windows[idx], gpg.classes[idx], gpg.confidence[idx],
        batch_size=cfg.batch_size, shuffle=True, cfg=cfg,
    )
    model = GPCL1DEncoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay)
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = _grad_scaler(amp_enabled)
    model.train()
    epochs = trange(cfg.pretrain_epochs, desc=f"GCCL seed {seed}")
    for _ in epochs:
        total, count = 0.0, 0
        for x, classes, confidence in loader:
            x = x.to(device, non_blocking=True)
            classes = classes.to(device, non_blocking=True)
            confidence = confidence.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                z = model(x)
                loss = confidence_weighted_infonce(
                    z, classes, confidence, cfg.temperature, cfg.negatives_per_anchor
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(x)
            count += len(x)
        epochs.set_postfix(loss=f"{total / max(count, 1):.5f}")
    return model


@torch.inference_mode()
def predict_normalized(model: nn.Module, windows: np.ndarray, cfg: Config) -> np.ndarray:
    device = next(model.parameters()).device
    loader = _loader(windows, batch_size=cfg.batch_size, shuffle=False, cfg=cfg)
    model.eval()
    output = []
    for (x,) in loader:
        output.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(output)


def train_regressor(
    encoder: GPCL1DEncoder, data: DataBundle, cfg: Config, seed: int
) -> FrozenGPCLRegressor:
    set_seed(seed, cfg.deterministic)
    device = select_device()
    y_range = max(data.target_max - data.target_min, 1e-8)
    y_train = ((data.train_targets - data.target_min) / y_range).astype(np.float32)
    y_val = ((data.val_targets - data.target_min) / y_range).astype(np.float32)
    x_train = data.train_candidate_windows[data.train_anchor_indices]
    train_loader = _loader(x_train, y_train, batch_size=cfg.batch_size, shuffle=True, cfg=cfg)
    val_loader = _loader(data.val_windows, y_val, batch_size=cfg.batch_size, shuffle=False, cfg=cfg)

    model = FrozenGPCLRegressor(encoder, cfg).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = _grad_scaler(amp_enabled)
    best_loss, best_state, stale = float("inf"), None, 0
    epochs = trange(cfg.finetune_epochs, desc=f"Regression seed {seed}")
    for _ in epochs:
        model.train()
        for x, target in train_loader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(x), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.inference_mode():
            for x, target in val_loader:
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
                value = criterion(model(x), target)
                val_sum += float(value) * len(x)
                val_n += len(x)
        val_loss = val_sum / val_n
        epochs.set_postfix(val_mse=f"{val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if cfg.early_stopping_patience and stale >= cfg.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_fcao(model: nn.Module, windows: np.ndarray, data: DataBundle, cfg: Config) -> np.ndarray:
    normalized = predict_normalized(model, windows, cfg)
    return normalized * (data.target_max - data.target_min) + data.target_min
