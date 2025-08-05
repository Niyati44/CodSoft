#!/usr/bin/env python3
"""
Setup script for Instagram Caption Generator

This script helps set up the environment by installing dependencies
and downloading the required model file.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_requirements():
    """Install Python requirements."""
    if not os.path.exists("requirements.txt"):
        print("✗ requirements.txt not found!")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def download_model():
    """Download the Mistral model file."""
    model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    if os.path.exists(model_file):
        print(f"✓ Model file {model_file} already exists")
        return True
    
    print(f"\nDownloading model file ({model_file})...")
    print("This may take several minutes (file size: ~4GB)")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\rProgress: {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")
        
        urllib.request.urlretrieve(model_url, model_file, progress_hook)
        print("\n✓ Model downloaded successfully")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download model: {e}")
        print("You can manually download it from:")
        print(model_url)
        return False

def check_system_dependencies():
    """Check for system dependencies."""
    print("\nChecking system dependencies...")
    
    # Check for tesseract
    tesseract_available = run_command("tesseract --version", "Checking Tesseract OCR")
    
    if not tesseract_available:
        print("\n⚠️  Tesseract OCR not found. Install it with:")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-eng")
        print("  macOS: brew install tesseract")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("\nNote: EasyOCR will be used as fallback if available")
    
    return tesseract_available

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    Path("captions").mkdir(exist_ok=True)
    print("✓ Created captions directory")

def main():
    """Main setup function."""
    print("Instagram Caption Generator Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Create directories
    create_directories()
    
    # Install Python dependencies
    if not install_requirements():
        print("\n✗ Failed to install Python dependencies")
        sys.exit(1)
    
    # Check system dependencies
    check_system_dependencies()
    
    # Download model
    model_downloaded = download_model()
    
    print("\n" + "=" * 40)
    print("Setup Summary:")
    print("✓ Python dependencies installed")
    print("✓ Directories created")
    
    if model_downloaded:
        print("✓ Model downloaded")
        print("\n🎉 Setup complete! You can now run:")
        print("   python instagram_caption_generator.py your_image.jpg")
    else:
        print("⚠️  Model download failed - you'll need to download it manually")
        print("   See README.md for instructions")
    
    print("\nFor usage instructions, see README.md")

if __name__ == "__main__":
    main()