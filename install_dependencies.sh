#!/bin/bash
# Unsloth Dependencies Installation Script
# Follows the exact order from working Dockerfile to avoid Flash Attention issues

set -e  # Exit on any error

echo "============================================================"
echo "🦥 UNSLOTH DEPENDENCIES INSTALLATION"
echo "============================================================"
echo ""
echo "This script installs dependencies in the correct order to avoid"
echo "Flash Attention 2 and other compatibility issues."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '3\.\d+')
echo "📋 Python version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(10|11|12)$ ]]; then
    echo "⚠️  Warning: Python 3.10+ required (3.12 recommended)"
    echo "   Current: $PYTHON_VERSION"
    echo "   Please upgrade your Python version"
fi

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "✅ CUDA version: $CUDA_VERSION"
else
    echo "⚠️  nvcc not found - CUDA version unknown"
fi

echo ""
echo "============================================================"
echo "Step 1/7: Cleaning up any broken installations"
echo "============================================================"
echo "Checking for corrupted packages..."

# Clean pip cache to avoid download issues
echo "Cleaning pip cache..."
python3 -m pip cache purge

echo ""
echo "============================================================"
echo "Step 2/7: Upgrading pip"
echo "============================================================"
python3 -m pip install --upgrade pip

echo ""
echo "============================================================"
echo "Step 3/7: Installing core ML frameworks (latest versions)"
echo "============================================================"
echo "Installing: trl, peft, bitsandbytes, transformers"
echo "⚠️  This may take a while for large packages like bitsandbytes..."
python3 -m pip install --timeout 600 --retries 5 "trl>=0.12.0" "peft>=0.13.0" "bitsandbytes>=0.45.0" "transformers[sentencepiece]>=4.46.0"

echo ""
echo "============================================================"
echo "Step 4/7: Installing PyTorch 2.8 (for Flash Attention compatibility)"
echo "============================================================"

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | cut -d. -f1,2 | tr -d '.')
    echo "Detected CUDA version: $(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
else
    echo "⚠️  nvcc not found, defaulting to CUDA 12.8"
    CUDA_VERSION="128"
fi

# Determine PyTorch index URL
if [[ "$CUDA_VERSION" == "118" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
elif [[ "$CUDA_VERSION" == "121" ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
elif [[ "$CUDA_VERSION" =~ ^12[4-9]$ ]] || [[ "$CUDA_VERSION" =~ ^1[3-9][0-9]$ ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
else
    echo "⚠️  CUDA version $CUDA_VERSION not directly supported, using cu128"
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
fi

echo "Installing: torch 2.8, torchvision from $TORCH_INDEX"
echo "⚠️  This will download ~2GB, may take several minutes..."
echo "💡 Using PyTorch 2.8 for maximum Flash Attention compatibility"
python3 -m pip install --timeout 600 torch==2.8.0 torchvision --index-url $TORCH_INDEX

echo ""
echo "============================================================"
echo "Step 5/7: Installing Unsloth (latest version)"
echo "============================================================"
echo "Installing from latest main branch"
python3 -m pip install --timeout 600 "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo ""
echo "============================================================"
echo "Step 6/7: Installing xformers (for PyTorch 2.8)"
echo "============================================================"
echo "⚠️  IMPORTANT: This is the fallback if Flash Attention doesn't work"
echo "Installing xformers 0.0.32.x pre-built wheel for PyTorch 2.8..."

python3 -m pip install --timeout 600 "xformers>=0.0.32,<0.0.33" --index-url $TORCH_INDEX

if [ $? -eq 0 ]; then
    echo "✅ xformers installed successfully"
else
    echo "❌ ERROR: xformers installation failed"
    echo "   xformers is required for training"
    exit 1
fi

echo ""
echo "============================================================"
echo "Step 7/7: Installing Flash Attention 2 (Optional)"
echo "============================================================"
echo "⚡ Flash Attention provides:"
echo "   - 10-20% faster training"
echo "   - 10-20% lower VRAM usage"
echo "   - Essential for large models (64GB+) on cloud GPUs"
echo ""

# Detect Python version
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "Detected Python version: 3.${PYTHON_VER#3}"

# Detect PyTorch version
TORCH_VER=$(python3 -c "import torch; v = torch.__version__.split('+')[0]; print(v)")
echo "Detected PyTorch version: $TORCH_VER"
echo ""

# Try to install Flash Attention
FA_INSTALLED=false

# Method 1: Install Flash Attention (latest stable)
echo "Installing Flash Attention (latest stable version)..."
echo "⚠️  This may take 5-10 minutes to compile..."

python3 -m pip install --timeout 1200 --no-deps --no-build-isolation flash-attn

if [ $? -eq 0 ]; then
    # Verify version
    FA_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    echo "✅ Flash Attention $FA_VERSION installed successfully"

    # Note about Unsloth compatibility
    if [[ "$FA_VERSION" == "2.8.3" ]]; then
        echo "ℹ️  Note: Unsloth may show a version warning, but Flash Attention 2.8.3 works fine"
    fi

    FA_INSTALLED=true
fi

# Report status if installation failed
if [ "$FA_INSTALLED" = false ]; then
    echo ""
    echo "⚠️  Flash Attention installation failed"
    echo ""
    echo "📊 Impact:"
    echo "   - Small models (<8B): Minimal (~10% slower with xformers)"
    echo "   - Large models (64GB+): May hit VRAM limits sooner"
    echo ""
    echo "✅ Fallback: xformers is installed and will be used"
    echo "   Training will still work perfectly!"
    echo ""
    echo "💡 To retry Flash Attention later:"
    echo "   pip install --no-build-isolation \"flash-attn>=2.7.1,<=2.8.2\""
fi

echo ""
echo "============================================================"
echo "Installing additional dependencies"
echo "============================================================"
python3 -m pip install datasets huggingface_hub accelerate sentencepiece protobuf python-dotenv

echo ""
echo "============================================================"
echo "✅ INSTALLATION COMPLETE"
echo "============================================================"
echo ""
echo "Verifying installation..."

# Verify imports
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "❌ PyTorch import failed"
python3 -c "import unsloth; print('✅ Unsloth: OK')" || echo "❌ Unsloth import failed"
python3 -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" || echo "❌ Transformers import failed"
python3 -c "import trl; print(f'✅ TRL: {trl.__version__}')" || echo "❌ TRL import failed"
python3 -c "import peft; print(f'✅ PEFT: {peft.__version__}')" || echo "❌ PEFT import failed"

echo ""
echo "============================================================"
echo "Attention Implementation Status"
echo "============================================================"
echo ""

# Check Flash Attention 2 (optimal)
FA_AVAILABLE=false
python3 -c "
try:
    import flash_attn
    print('✅ Flash Attention 2: Available (optimal performance)')
    exit(0)
except:
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    FA_AVAILABLE=true
fi

# Check xformers (fallback)
XFORMERS_AVAILABLE=false
python3 -c "
try:
    import xformers
    import xformers.ops
    print('✅ xformers: Available (fallback attention)')
    print(f'   Version: {xformers.__version__}')
    exit(0)
except Exception as e:
    print(f'❌ xformers: Import failed - {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    XFORMERS_AVAILABLE=true
fi

# Determine which will be used
echo ""
if [ "$FA_AVAILABLE" = true ]; then
    echo "🎯 Active: Flash Attention 2 (best performance)"
    echo "   - 10-20% faster training"
    echo "   - 10-20% lower VRAM usage"
    echo "   - Optimal for large models"
elif [ "$XFORMERS_AVAILABLE" = true ]; then
    echo "🎯 Active: xformers (fallback)"
    echo "   - Reliable and stable"
    echo "   - Works on all GPUs"
    echo "   - ~85-90% of Flash Attention speed"
else
    echo "❌ ERROR: No attention implementation available!"
    echo "   Both Flash Attention and xformers failed to install"
    echo "   Training will not work without one of these"
    exit 1
fi

echo ""
echo "Checking CUDA availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available - GPU training will not work')
"

echo ""
echo "============================================================"
echo "🎉 SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Copy example config: cp .env.example .env"
echo "  2. Edit .env with your settings"
echo "  3. Run quick test: python train.py"
echo ""
echo "For troubleshooting, see SETUP.md"
echo "============================================================"
