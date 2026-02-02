# Missing Dependency Fix for Google Colab
# Run this cell BEFORE the main installation cell (Step 1)

!pip install -q hdf5plugin numba

print("✅ Additional dependencies installed!")
print("   - hdf5plugin (required for HDF5 compression)")
print("   - numba (required for performance optimization)")
print("\nNow continue with the rest of the notebook.")
