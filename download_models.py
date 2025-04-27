#!/usr/bin/env python3
"""
SkyReels-V2 Model Downloader (HuggingFace only)
Downloads SkyReels-V2 models from Hugging Face repositories.
"""

import os
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# List of models from Hugging Face
MODELS = [
    {
        "repo_id": "Skywork/SkyReels-V2-DF-1.3B-540P",
        "local_dir": "./models/SkyReels-V2-DF-1.3B-540P"
    },
    {
        "repo_id": "Skywork/SkyReels-V2-DF-14B-540P",
        "local_dir": "./models/SkyReels-V2-DF-14B-540P"
    },
    {
        "repo_id": "Skywork/SkyReels-V2-DF-14B-720P",
        "local_dir": "./models/SkyReels-V2-DF-14B-720P"
    },
    {
        "repo_id": "Skywork/SkyReels-V2-T2V-14B-540P",
        "local_dir": "./models/SkyReels-V2-T2V-14B-540P"
    },
    {
        "repo_id": "Skywork/SkyReels-V2-T2V-14B-720P",
        "local_dir": "./models/SkyReels-V2-T2V-14B-720P"
    },
    {
        "repo_id": "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "local_dir": "./models/SkyReels-V2-I2V-1.3B-540P"
    },
    {
        "repo_id": "Skywork/SkyReels-V2-I2V-14B-540P",
        "local_dir": "./models/SkyReels-V2-I2V-14B-540P"
    },
    {
        "repo_id": "Skywork/SkyReels-V2-I2V-14B-720P",
        "local_dir": "./models/SkyReels-V2-I2V-14B-720P"
    }
]

def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import huggingface_hub
        print("✓ huggingface_hub is already installed.")
    except ImportError:
        print("➔ Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)

def download_from_huggingface(model_info, resume=True):
    """Download a model from Hugging Face."""
    from huggingface_hub import snapshot_download

    repo_id = model_info["repo_id"]
    local_dir = model_info["local_dir"]

    print(f"⏳ Downloading {repo_id}...")
    start_time = time.time()

    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=resume)
        elapsed = time.time() - start_time
        print(f"✅ Successfully downloaded {repo_id} in {elapsed:.1f} seconds.")
        return True
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download SkyReels-V2 models from Hugging Face.")
    parser.add_argument("--model", type=str, help="Download a specific model (e.g., 'I2V-14B-720P')")
    parser.add_argument("--parallel", action="store_true", help="Download models in parallel.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume download if interrupted (default: True)")

    args = parser.parse_args()

    check_dependencies()

    selected_models = MODELS

    if args.model:
        selected_models = [m for m in MODELS if args.model in m["repo_id"]]
        if not selected_models:
            print(f"❌ No models matching '{args.model}' found.")
            return

    print("\n=== DOWNLOAD SUMMARY ===")
    print(f"Models to be downloaded: {len(selected_models)}")
    for model in selected_models:
        print(f"  - {model['repo_id']} ➔ {model['local_dir']}")

    print("\nStarting downloads...\n")

    if args.parallel:
        with ThreadPoolExecutor(max_workers=min(4, len(selected_models))) as executor:
            futures = {executor.submit(download_from_huggingface, model, args.resume): model for model in selected_models}
            for future in futures:
                future.result()
    else:
        for model in selected_models:
            download_from_huggingface(model, args.resume)

    print("\n✅ All downloads completed successfully!")

if __name__ == "__main__":
    main()
