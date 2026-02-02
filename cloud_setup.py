#!/usr/bin/env python3
"""
Cloud Setup Verification Script
Runs on Streamlit Cloud to verify the environment is ready.
"""

import sys
import os
import streamlit as st

def verify_environment():
    print("☁️ Verifying Cloud Environment...")
    
    # Check Python version
    print(f"   Python: {sys.version.split()[0]}")
    
    # Check key dependencies
    try:
        import surya
        print(f"   ✅ Surya package installed: {surya.__file__}")
    except ImportError as e:
        print(f"   ❌ Surya package MISSING: {e}")
    
    try:
        import torch
        print(f"   ✅ PyTorch installed: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   🚀 CUDA GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ️  Running on CPU (Standard for Streamlit Free Tier)")
    except ImportError:
        print("   ❌ PyTorch MISSING")

if __name__ == "__main__":
    verify_environment()
