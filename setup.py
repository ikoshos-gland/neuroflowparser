#!/usr/bin/env python3
"""
Setup script for Jina v4 RAG Pipeline

This script helps with:
- Environment setup
- Dependency installation
- Model downloading
- System configuration
- Pipeline initialization
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version < (3, 8):
        logger.error("Python 3.8+ is required. Current version: {}.{}".format(version.major, version.minor))
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro} ✓")
    return True

def check_system_dependencies():
    """Check system-level dependencies"""
    logger.info("Checking system dependencies...")
    
    dependencies = {
        'tesseract': 'tesseract',
        'git': 'git'
    }
    
    missing = []
    
    for name, command in dependencies.items():
        try:
            subprocess.run([command, '--version'], 
                         capture_output=True, check=True)
            logger.info(f"{name}: Found ✓")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning(f"{name}: Not found ✗")
            missing.append(name)
    
    if missing:
        logger.warning("Missing dependencies: {}".format(', '.join(missing)))
        show_install_instructions(missing)
        return False
    
    return True

def show_install_instructions(missing_deps):
    """Show installation instructions for missing dependencies"""
    system = platform.system().lower()
    
    instructions = {
        'tesseract': {
            'linux': 'sudo apt-get install tesseract-ocr',
            'darwin': 'brew install tesseract',
            'windows': 'Download from: https://github.com/UB-Mannheim/tesseract/wiki'
        },
        'git': {
            'linux': 'sudo apt-get install git',
            'darwin': 'brew install git',
            'windows': 'Download from: https://git-scm.com/download/win'
        }
    }
    
    logger.info("\nInstallation instructions:")
    for dep in missing_deps:
        if dep in instructions:
            instruction = instructions[dep].get(system, instructions[dep]['linux'])
            logger.info(f"  {dep}: {instruction}")

def install_python_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ], check=True)
        logger.info("Python dependencies installed ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def download_spacy_model():
    """Download required spaCy model"""
    logger.info("Downloading spaCy English model...")
    
    try:
        # Try to download spaCy model
        subprocess.run([
            sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
        ], check=True, capture_output=True)
        logger.info("spaCy model downloaded ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to download spaCy model: {e}")
        logger.info("You can download it manually with: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        logger.warning(f"spaCy not available: {e}")
        return False

def check_gpu_availability():
    """Check for GPU availability"""
    logger.info("Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU available: {gpu_name} ({gpu_count} devices) ✓")
            return True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) GPU available ✓")
            return True
        else:
            logger.info("No GPU available - will use CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed yet")
        return False

def download_jina_model():
    """Pre-download Jina v4 model"""
    logger.info("Pre-downloading Jina v4 model (this may take a while)...")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_name = "jinaai/jina-embeddings-v4"
        
        # Download model and tokenizer
        logger.info("Downloading model...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        logger.info("Jina v4 model downloaded successfully ✓")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download Jina v4 model: {e}")
        logger.info("You can download it later when running the pipeline")
        return False

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        'saved_pipeline',
        'processed',
        'logs',
        'cache'
    ]
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_name}")
    
    return True

def create_config_files():
    """Create default configuration files"""
    logger.info("Creating configuration files...")
    
    # Default embedding config
    embedding_config = {
        "model_name": "jinaai/jina-embeddings-v4",
        "embedding_dim": 2048,
        "max_length": 32768,
        "task_type": "retrieval",
        "batch_size": 4,
        "device": "auto"
    }
    
    # Default processing config
    processing_config = {
        "extract_images": True,
        "extract_tables": True,
        "use_ocr": True,
        "max_chunk_size": 1200,
        "overlap_size": 150,
        "min_image_size": [100, 100]
    }
    
    # Pipeline config
    pipeline_config = {
        "downloads_dir": "../downloads",
        "vector_store_type": "faiss",
        "save_dir": "./saved_pipeline"
    }
    
    configs = {
        'embedding_config.json': embedding_config,
        'processing_config.json': processing_config,
        'pipeline_config.json': pipeline_config
    }
    
    for filename, config in configs.items():
        config_path = Path(filename)
        if not config_path.exists():
            import json
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created: {filename}")
    
    return True

def run_basic_test():
    """Run a basic functionality test"""
    logger.info("Running basic functionality test...")
    
    try:
        # Test imports
        from jina_v4_pipeline import EmbeddingConfig, JinaV4Pipeline
        from pdf_processor import ProcessingConfig
        
        logger.info("Module imports: ✓")
        
        # Test configuration
        embedding_config = EmbeddingConfig(batch_size=1)
        processing_config = ProcessingConfig()
        
        logger.info("Configuration creation: ✓")
        
        # If we have torch available, test device detection
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Device detection: {device} ✓")
        except ImportError:
            pass
        
        logger.info("Basic functionality test passed ✓")
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("Jina v4 RAG Pipeline Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install Python dependencies
    if not install_python_dependencies():
        logger.error("Failed to install Python dependencies")
        return 1
    
    # Check system dependencies
    system_deps_ok = check_system_dependencies()
    if not system_deps_ok:
        logger.warning("Some system dependencies are missing. The pipeline may not work fully.")
    
    # Download spaCy model
    download_spacy_model()
    
    # Check GPU
    check_gpu_availability()
    
    # Setup directories
    if not setup_directories():
        return 1
    
    # Create config files
    if not create_config_files():
        return 1
    
    # Run basic test
    if not run_basic_test():
        logger.error("Basic functionality test failed")
        return 1
    
    # Optional: Download model
    print("\nOptional: Pre-download Jina v4 model?")
    print("This is a large model (~7GB) and will take time to download.")
    download_model = input("Download now? (y/N): ").strip().lower()
    
    if download_model in ['y', 'yes']:
        download_jina_model()
    else:
        logger.info("Model will be downloaded automatically when first used")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Ensure your PDF files are in the downloads folder")
    print("2. Run: python rag_interface.py interactive")
    print("3. Or run: python rag_interface.py api")
    print("\nFor help: python rag_interface.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())