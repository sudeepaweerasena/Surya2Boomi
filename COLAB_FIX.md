# Quick Fix for Missing Dependencies in Google Colab

## Problem
If you encounter the error:
```
ModuleNotFoundError: No module named 'hdf5plugin'
```

## Solution
Add this cell **at the very beginning** of the notebook (before Step 1):

```python
# Install missing dependencies
!pip install -q hdf5plugin numba

print("✅ Additional dependencies installed!")
```

Then run all cells normally.

## Alternative: Full Corrected Installation Cell

Replace the Step 1 installation cell with:

```python
# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Install required packages
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers huggingface-hub einops timm
!pip install -q flask flask-cors pyngrok
!pip install -q netCDF4 xarray h5py hdf5plugin scipy numpy pandas matplotlib
!pip install -q PyYAML tqdm peft numba

print("✅ All dependencies installed successfully!")
```

The key additions are:
- **hdf5plugin** - Required for HDF5 compression used in Surya datasets
- **numba** - Required for performance optimization in data processing
