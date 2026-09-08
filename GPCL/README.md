# Graph-based Pseudo-label Contrastive Learning for Soft Measurement of Free Calcium Oxide

`btrain.csv` is treated as a temporally continuous reconstructed dataset containing 96,600 rows. Every consecutive 60 rows correspond to one hourly f-CaO label in `fcao.csv`.

## File Structure

* `config.py`: Contains all configurable parameters.
* `data.py`: Handles CSV validation, training-set normalization, construction of 60×9 windows, and strict temporal splitting.
* `pmi.py`: Constructs the variable-level PMI graph.
* `graph.py`: Handles the sample similarity graph, label propagation, confidence-based screening, and three-level discretization.
* `models.py`: Implements the two-layer 1D-CNN, projection head, and regression model with a frozen encoder.
* `losses.py`: Implements the confidence-weighted InfoNCE loss following the formulation in the paper.
* `train.py`: Handles GCCL pre-training, downstream regression, and prediction.
* `evaluation.py`: Computes RMSE and R², exports CSV prediction results, and generates prediction curves.
* `main.py`: Provides the end-to-end execution entry point.

## Windows 11 / Python 3.10 / CUDA 12.8

Run the following commands in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install_windows_cuda128.ps1
.\.venv\Scripts\python.exe .\main.py
```

The installation script uses the official PyTorch Windows wheel for CUDA 12.8. The program automatically detects CUDA. If CUDA is unavailable, it can fall back to CPU execution; however, graph construction at the scale reported in the paper is not suitable for CPU-only execution.

## Data Split

* Samples 1–1208: training set;
* Sample 1209: a complete 60-minute isolation window, fully discarded;
* Samples 1210–1409: validation set;
* Sample 1410: a complete 60-minute isolation window, fully discarded;
* Samples 1411–1610: independent test set.

The training process stream contains a total of `1208*60=72480` rows. Using a stride of 1 produces `72421` candidate windows. Among them, the 1208 windows whose starting positions occur every 60 time steps serve as anchors with real f-CaO labels, while the remaining 71213 windows are used only for pseudo-labeling and representation learning. Process data from the validation and test sets are never incorporated into the GPG graph.

## Exact Mode and Engineering Mode

By default, `config.py` sets:

```python
graph_mode = "exact_blockwise"
```

This mode computes all sample pairs block by block and implements Eqs. (3)–(6) from the paper exactly. It produces a sparse graph at the scale reported in the paper and therefore requires substantial system memory, GPU memory, and computation time. The constructed graph and GPG results are cached in `outputs/cache/`.

If the available hardware resources are insufficient, change the configuration to:

```python
graph_mode = "knn_approx"
knn_neighbors = 2048
```

This mode first uses low-dimensional nearest-neighbor search to generate candidate edges and then applies the original PMI-consistency threshold and Gaussian similarity formulation to those candidate edges without modification. This is an engineering approximation and should not be presented as equivalent to the all-pairs graph construction used in the paper.



## Outputs

After successful execution, `outputs/` contains:

* `metrics.json`: RMSE and R² for each random seed, the mean and standard deviation across five runs, and metrics calculated from the mean predictions;
* `predictions.csv`: ground-truth values, predictions from each run, and the averaged predictions across five runs;
* `prediction_curve.png`: curves comparing the ground-truth and predicted values on the test set;
* `models/gpcl_seed_*.pt`: trained model weights;
* `cache/`: cached sample graphs and pseudo-labeling results that are computationally expensive to regenerate.
