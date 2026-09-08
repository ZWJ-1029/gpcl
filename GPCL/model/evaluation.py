"""Metrics and publication-style true/predicted test curve."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 0 else float("nan")
    return {"rmse": rmse, "r2": r2}


def save_results(
    output_dir: Path,
    y_true: np.ndarray,
    predictions_by_seed: list[np.ndarray],
    seeds: tuple[int, ...],
    metadata: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.vstack(predictions_by_seed)
    mean_prediction = matrix.mean(axis=0)
    seed_metrics = [regression_metrics(y_true, row) for row in matrix]
    ensemble_metrics = regression_metrics(y_true, mean_prediction)
    rmse_values = np.array([m["rmse"] for m in seed_metrics])
    r2_values = np.array([m["r2"] for m in seed_metrics])
    report = {
        "per_seed": {str(seed): metric for seed, metric in zip(seeds, seed_metrics)},
        "paper_style_mean_std": {
            "rmse_mean": float(rmse_values.mean()),
            "rmse_std": float(rmse_values.std(ddof=1)) if len(rmse_values) > 1 else 0.0,
            "r2_mean": float(r2_values.mean()),
            "r2_std": float(r2_values.std(ddof=1)) if len(r2_values) > 1 else 0.0,
        },
        "five_seed_mean_prediction": ensemble_metrics,
        "metadata": metadata,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    frame = pd.DataFrame({"test_sample": np.arange(1, len(y_true) + 1), "true_fcao": y_true})
    for seed, row in zip(seeds, matrix):
        frame[f"prediction_seed_{seed}"] = row
    frame["prediction_mean"] = mean_prediction
    frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    x = np.arange(1, len(y_true) + 1)
    ax.plot(x, y_true, color="#202124", linewidth=1.5, label="True f-CaO")
    ax.plot(x, mean_prediction, color="#d62728", linewidth=1.35, label="GPCL prediction (seed mean)")
    ax.set_xlabel("Test sample (chronological)")
    ax.set_ylabel("f-CaO content (%)")
    ax.set_title(
        f"GPCL test prediction | RMSE={ensemble_metrics['rmse']:.4f}, "
        f"R2={ensemble_metrics['r2']:.4f}"
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_curve.png", bbox_inches="tight")
    plt.close(fig)
    return report

