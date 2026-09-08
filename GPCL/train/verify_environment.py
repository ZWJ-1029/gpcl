"""Small preflight check for Windows 11 / Python 3.10 / CUDA 12.8."""

import platform
import sys

import numpy
import pandas
import scipy
import sklearn
import torch


print("OS:", platform.platform())
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(256, 256, device="cuda")
    print("CUDA matrix test:", float((x @ x.T).mean()))
print("NumPy/Pandas/SciPy/sklearn:", numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__)

if sys.version_info[:2] != (3, 10):
    print("WARNING: This reproduction was requested for Python 3.10.")
if torch.version.cuda != "12.8":
    print("WARNING: Install the cu128 PyTorch wheel with install_windows_cuda128.ps1.")

