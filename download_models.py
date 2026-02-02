#!/usr/bin/env python3
"""
Solar Flare Model & Data Downloader

Downloads all necessary models, data, and configuration files for local inference.
"""

import os
import sys
from pathlib import Path
import urllib.request
import shutil
from tqdm import tqdm
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess

# Directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SURYA_DIR = BASE_DIR / "Surya"

class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def clone_surya_repo():
    """Clone Surya repository if not exists"""
    print("\n📦 Cloning Surya Repository...")
    
    if SURYA_DIR.exists() and (SURYA_DIR / ".git").exists():
        print("   ✅ Repository already exists")
        return
    
    try:
        subprocess.run([
            "git", "clone",
            "https://github.com/NASA-IMPACT/Surya.git",
            str(SURYA_DIR)
        ], check=True, capture_output=True)
        print("   ✅ Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error cloning repository: {e}")
        sys.exit(1)

def download_from_huggingface():
    """Download models and data from Hugging Face"""
    print("\n📥 Downloading from Hugging Face...")
    print("   This may take 10-20 minutes depending on your internet speed")
    print("")
    
    try:
        # Download scalers
        print("   Downloading scalers...")
        scalers_path = hf_hub_download(
            repo_id="nasa-ibm-ai4science/Surya-1.0",
            filename="scalers.yaml",
            local_dir=DATA_DIR,
            token=None
        )
        print(f"   ✅ Scalers: {scalers_path}")
        
        # Download foundation model weights
        print("\n   Downloading foundation model weights (~1.8GB)...")
        print("   This will take a while...")
        foundation_path = hf_hub_download(
            repo_id="nasa-ibm-ai4science/Surya-1.0",
            filename="surya.366m.v1.pt",  # Correct filename
            local_dir=MODELS_DIR,
            token=None
        )
        # Symlink to expected name for compatibility
        checkpoint_path = MODELS_DIR / "checkpoint.pth"
        if not checkpoint_path.exists():
            checkpoint_path.symlink_to(Path(foundation_path).name)
        print(f"   ✅ Foundation model: {foundation_path}")
        
        # Download solar flare specific weights
        print("\n   Downloading solar flare task weights (~1.8GB)...")
        print("   This will also take a while...")
        try:
            task_path = hf_hub_download(
                repo_id="nasa-ibm-ai4science/solar_flares_surya",  # Correct repo
                filename="solar_flare_weights.pth",  # Correct filename
                local_dir=MODELS_DIR,
                token=None
            )
            print(f"   ✅ Task weights: {task_path}")
        except Exception as e:
            print(f"   ⚠️  Solar flare weights download failed: {e}")
            print(f"   ℹ️  Will use foundation model only")
        
        # Download config
        print("\n   Downloading config...")
        try:
            config_path = hf_hub_download(
                repo_id="nasa-ibm-ai4science/Surya-1.0",
                filename="config.yaml",
                local_dir=DATA_DIR,
                token=None
            )
            print(f"   ✅ Config: {config_path}")
        except:
            print("   ℹ️  Config not found, will use local copy")
        
        print("\n   ✅ Hugging Face downloads complete")
        
    except Exception as e:
        print(f"\n   ❌ Error downloading from Hugging Face: {e}")
        print("   ℹ️  You may need to:")
        print("      1. Check your internet connection")
        print("      2. Login to Hugging Face: huggingface-cli login")
        sys.exit(1)

def download_config_files():
    """Download or copy configuration files"""
    print("\n📄 Setting up configuration files...")
    
    # Check if config exists in Surya repo
    surya_config = SURYA_DIR / "downstream_examples" / "solar_flare_forcasting" / "config_infer.yaml"
    local_config = DATA_DIR / "config_infer.yaml"
    
    if surya_config.exists() and not local_config.exists():
        shutil.copy(surya_config, local_config)
        print(f"   ✅ Copied config from Surya repo")
    else:
        print("   ℹ️  Config file already exists or not found in repo")
    
    # Create a default config if none exists
    if not local_config.exists():
        print("   Creating default configuration...")
        default_config = """# Solar Flare Forecasting Configuration
model:
  model_type: "spectformer"
  img_size: 512
  patch_size: 16
  in_channels: 9
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  drop_rate: 0.0
  window_size: 8
  dp_rank: 0
  learned_flow: true
  use_latitude_in_learned_flow: false
  init_weights: false
  checkpoint_layers: null
  rpe: false
  ensemble: null
  finetune: true
  nglo: 0
  dropout: 0.1
  num_penultimate_transformer_layers: 1
  num_penultimate_heads: 8
  use_lora: true
  spectral_blocks: 3
  time_embedding:
    time_dim: 12

dtype: "bfloat16"

data:
  sdo_data_root_path: "./data/validation"
  valid_data_path: "./data/valid_index.csv"
  flare_data_path: "./data/flare_index.csv"
  scalers_path: "./data/scalers.yaml"
  channels: ["94A", "131A", "171A", "193A", "211A", "304A", "335A", "1600A", "1700A"]
  time_delta_input_minutes: 12
  time_delta_target_minutes: 24
  pooling: "mean"

classifier_metrics:
  num_classes: 1

rollout_steps: 1
drop_hmi_probability: 1.0
num_mask_aia_channels: 0
use_latitude_in_learned_flow: false
"""
        with open(local_config, 'w') as f:
            f.write(default_config)
        print("   ✅ Created default config")

def create_sample_data_index():
    """Create sample data index files for testing"""
    print("\n📊 Creating sample data indices...")
    
    # Create a minimal valid_index.csv
    valid_index = DATA_DIR / "valid_index.csv"
    if not valid_index.exists():
        with open(valid_index, 'w') as f:
            f.write("timestamp\n")
            f.write("2024-01-01T00:00:00\n")
            f.write("2024-01-01T01:00:00\n")
            f.write("2024-01-01T02:00:00\n")
        print("   ✅ Created sample validation index")
    
    # Create a minimal flare_index.csv
    flare_index = DATA_DIR / "flare_index.csv"
    if not flare_index.exists():
        with open(flare_index, 'w') as f:
            f.write("timestamp,flare_class,peak_flux\n")
            f.write("2024-01-01T00:00:00,B,0.0\n")
        print("   ✅ Created sample flare index")

def verify_downloads():
    """Verify all necessary files are downloaded"""
    print("\n✅ Verifying downloads...")
    
    required_files = {
        "Foundation model": MODELS_DIR / "checkpoint.pth",
        "Scalers": DATA_DIR / "scalers.yaml",
        "Config": DATA_DIR / "config_infer.yaml",
    }
    
    all_good = True
    for name, path in required_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {name}: NOT FOUND")
            all_good = False
    
    return all_good

def main():
    """Main download process"""
    print("🌞 Solar Flare Model & Data Downloader")
    print("=" * 50)
    
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "validation").mkdir(exist_ok=True, parents=True)
    
    # Step 1: Clone Surya repository
    clone_surya_repo()
    
    # Step 2: Download from Hugging Face
    download_from_huggingface()
    
    # Step 3: Setup config files
    download_config_files()
    
    # Step 4: Create sample data
    create_sample_data_index()
    
    # Step 5: Verify
    if verify_downloads():
        print("\n" + "=" * 50)
        print("✨ All downloads complete!")
        print("\nNext steps:")
        print("  1. Run: ./run_local.sh")
        print("  2. Open web app in browser")
        print("  3. Generate forecasts!")
        print("")
    else:
        print("\n⚠️  Some files are missing. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
