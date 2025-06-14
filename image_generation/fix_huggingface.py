#!/usr/bin/env python3
"""
Script to fix Hugging Face model loading issues
"""
import os
import shutil
import subprocess
import sys

def install_packages():
    """Install/update required packages"""
    packages = [
        "torch",
        "torchvision", 
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface_hub"
    ]
    
    for package in packages:
        try:
            print(f"📦 Installing/updating {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
            print(f"✅ {package} installed/updated successfully")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")

def clear_huggingface_cache():
    """Clear Hugging Face cache"""
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        "C:\\Users\\zhuol\\.cache\\huggingface"  # Windows specific path from error
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                print(f"🧹 Clearing cache: {cache_dir}")
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not clear {cache_dir}: {e}")

def test_model_loading():
    """Test if the model can be loaded"""
    try:
        print("🧪 Testing model loading...")
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        print("✅ Model loading test successful!")
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing Hugging Face model issues...")
    print("=" * 50)
    
    # Step 1: Update packages
    install_packages()
    
    # Step 2: Clear cache
    clear_huggingface_cache()
    
    # Step 3: Test loading
    test_model_loading()
    
    print("=" * 50)
    print("🏁 Fix attempt completed. Try running the image editing again.")
