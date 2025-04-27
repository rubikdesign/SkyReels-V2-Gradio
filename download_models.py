#!/usr/bin/env python3
"""
SkyReels-V2 Model Downloader
This script downloads SkyReels-V2 models from the specified repositories.
"""

import os
import argparse
import subprocess
import time
import sys
from concurrent.futures import ThreadPoolExecutor

# Define models to download based on provided repositories
MODELS = {
    "huggingface": [
        {
            "repo_id": "Skywork/SkyReels-V2-T2V-14B-540P",
            "local_dir": "./models/SkyReels-V2-T2V-14B-540P"
        },
        {
            "repo_id": "Skywork/SkyReels-V2-I2V-14B-540P",
            "local_dir": "./models/SkyReels-V2-I2V-14B-540P"
        },
        {
            "repo_id": "Skywork/SkyReels-V2-I2V-14B-720P",
            "local_dir": "./models/SkyReels-V2-I2V-14B-720P"
        }
    ],
    "modelscope": [
        "Skywork/SkyReels-V2-DF-1.3B-540P",
        "Skywork/SkyReels-V2-DF-14B-540P"
    ]
}

def check_dependencies():
    """Check if required packages are installed and install them if not."""
    try:
        import huggingface_hub
        print("✓ huggingface_hub is already installed")
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        
    try:
        import modelscope
        print("✓ modelscope is already installed")
    except ImportError:
        print("Installing modelscope...")
        subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=True)

def download_from_huggingface(model_info, resume=True):
    """Download a model from Hugging Face."""
    from huggingface_hub import snapshot_download
    
    repo_id = model_info["repo_id"]
    local_dir = model_info["local_dir"]
    
    print(f"⏳ Downloading {repo_id} from Hugging Face...")
    start_time = time.time()
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=resume)
        elapsed = time.time() - start_time
        print(f"✅ Successfully downloaded {repo_id} in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {str(e)}")
        return False

def download_from_modelscope(model_id):
    """Download a model from ModelScope."""
    print(f"⏳ Downloading {model_id} from ModelScope...")
    start_time = time.time()
    
    try:
        subprocess.run(["modelscope", "download", "--model", model_id], check=True)
        elapsed = time.time() - start_time
        print(f"✅ Successfully downloaded {model_id} in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download SkyReels-V2 models from specified repositories")
    parser.add_argument("--source", choices=["both", "huggingface", "modelscope"], default="both",
                        help="Source to download models from (default: both)")
    parser.add_argument("--model", type=str, help="Download a specific model (e.g., 'I2V-14B-720P')")
    parser.add_argument("--parallel", action="store_true", help="Download models in parallel")
    parser.add_argument("--resume", action="store_true", default=True, 
                        help="Resume download if files already exist (default: True)")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Filter models if specific model requested
    if args.model:
        hf_models = [m for m in MODELS["huggingface"] if args.model in m["repo_id"]]
        ms_models = [m for m in MODELS["modelscope"] if args.model in m]
        
        if not hf_models and not ms_models:
            print(f"No models matching '{args.model}' found.")
            return
            
        MODELS["huggingface"] = hf_models
        MODELS["modelscope"] = ms_models
    
    # Show summary of what will be downloaded
    print("\n=== DOWNLOAD SUMMARY ===")
    if args.source in ["both", "huggingface"]:
        print(f"Will download {len(MODELS['huggingface'])} models from Hugging Face:")
        for model in MODELS["huggingface"]:
            print(f"  - {model['repo_id']}")
            
    if args.source in ["both", "modelscope"]:
        print(f"Will download {len(MODELS['modelscope'])} models from ModelScope:")
        for model in MODELS["modelscope"]:
            print(f"  - {model}")
    
    print("\nStarting downloads...\n")
    
    # Download from Hugging Face
    if args.source in ["both", "huggingface"]:
        if args.parallel:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(download_from_huggingface, model, args.resume): model for model in MODELS["huggingface"]}
                for future in futures:
                    future.result()
        else:
            for model in MODELS["huggingface"]:
                download_from_huggingface(model, args.resume)
    
    # Download from ModelScope
    if args.source in ["both", "modelscope"]:
        if args.parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(download_from_modelscope, model): model for model in MODELS["modelscope"]}
                for future in futures:
                    future.result()
        else:
            for model in MODELS["modelscope"]:
                download_from_modelscope(model)
    
    print("\n✅ Download operations completed!")
    print("\nModels downloaded:")
    print("  - Hugging Face models are saved in their respective './models/' directories")
    print("  - ModelScope models need to be moved to the appropriate './models/' directories")
    
if __name__ == "__main__":
    main()
