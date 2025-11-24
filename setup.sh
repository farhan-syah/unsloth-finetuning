#!/bin/bash
# Unsloth Fine-tuning Setup Script
# Handles: Python dependencies, llama.cpp installation, symlinks, verification

set -e

echo "============================================================"
echo "🚀 UNSLOTH FINE-TUNING SETUP"
echo "============================================================"
echo ""

# Check if we're in the correct directory
if [ ! -f "train.py" ] || [ ! -f "build.py" ]; then
    echo "❌ Error: Must run from unsloth project directory"
    echo "   Expected files: train.py, build.py"
    exit 1
fi

PROJECT_DIR=$(pwd)

# ============================================================
# Step 1: Python Dependencies
# ============================================================
echo "============================================================"
echo "📦 Step 1: Installing Python Dependencies"
echo "============================================================"
echo ""

if [ ! -f "install_dependencies.sh" ]; then
    echo "❌ Error: install_dependencies.sh not found"
    exit 1
fi

echo "Running install_dependencies.sh..."
bash install_dependencies.sh

echo ""
echo "✅ Python dependencies installed"
echo ""

# ============================================================
# Step 2: llama.cpp Installation
# ============================================================
echo "============================================================"
echo "🔧 Step 2: llama.cpp Setup for GGUF Conversion"
echo "============================================================"
echo ""

# Check if user wants to skip llama.cpp installation
read -p "Install llama.cpp for GGUF conversion? (y/n, default: y): " install_llama
install_llama=${install_llama:-y}

if [[ "$install_llama" =~ ^[Yy]$ ]]; then

    # Check if llama.cpp already exists
    if [ -d "llama.cpp" ] && [ -f "llama.cpp/build/bin/llama-quantize" ]; then
        echo "✅ llama.cpp already installed at: $PROJECT_DIR/llama.cpp"
        echo "   Binaries found in: llama.cpp/build/bin/"
        read -p "Reinstall llama.cpp? (y/n, default: n): " reinstall
        reinstall=${reinstall:-n}
        if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
            echo "   Skipping llama.cpp installation..."
            SKIP_LLAMA_BUILD=true
        fi
    fi

    if [ "$SKIP_LLAMA_BUILD" != "true" ]; then
        echo ""
        echo "Detecting GPU backend..."

        # Detect available GPU backends
        HAS_NVIDIA=false
        HAS_AMD=false
        HAS_INTEL=false

        # Check for NVIDIA
        if command -v nvidia-smi &> /dev/null; then
            HAS_NVIDIA=true
            echo "✅ NVIDIA GPU detected (nvidia-smi found)"
        fi

        # Check for AMD ROCm
        if command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
            HAS_AMD=true
            echo "✅ AMD GPU detected (ROCm found)"
        fi

        # Check for Intel
        if lspci 2>/dev/null | grep -i "vga.*intel" &> /dev/null; then
            HAS_INTEL=true
            echo "✅ Intel GPU detected"
        fi

        # Determine backend priority: Vulkan > CUDA > ROCm > None
        echo ""
        echo "Available backends:"
        echo "  1) Vulkan (recommended - works on NVIDIA, AMD, Intel)"
        if [ "$HAS_NVIDIA" = true ]; then
            echo "  2) CUDA (NVIDIA only)"
        fi
        if [ "$HAS_AMD" = true ]; then
            echo "  3) ROCm (AMD only)"
        fi
        echo "  4) CPU only (no GPU acceleration)"
        echo ""

        read -p "Select backend (1-4, default: 1 for Vulkan): " backend_choice
        backend_choice=${backend_choice:-1}

        CMAKE_FLAGS=""
        case $backend_choice in
            1)
                echo "Selected: Vulkan backend"
                CMAKE_FLAGS="-DGGML_VULKAN=ON"
                ;;
            2)
                if [ "$HAS_NVIDIA" = true ]; then
                    echo "Selected: CUDA backend"
                    # Try to detect CUDA architecture
                    if command -v nvidia-smi &> /dev/null; then
                        # Get GPU compute capability (e.g., 8.6 for RTX 3060)
                        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
                        echo "Detected CUDA architecture: $GPU_ARCH"
                        CMAKE_FLAGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH"
                    else
                        CMAKE_FLAGS="-DGGML_CUDA=ON"
                    fi
                else
                    echo "⚠️  CUDA not available, falling back to Vulkan"
                    CMAKE_FLAGS="-DGGML_VULKAN=ON"
                fi
                ;;
            3)
                if [ "$HAS_AMD" = true ]; then
                    echo "Selected: ROCm backend"
                    CMAKE_FLAGS="-DGGML_HIPBLAS=ON"
                else
                    echo "⚠️  ROCm not available, falling back to Vulkan"
                    CMAKE_FLAGS="-DGGML_VULKAN=ON"
                fi
                ;;
            4)
                echo "Selected: CPU only (no GPU acceleration)"
                CMAKE_FLAGS=""
                ;;
            *)
                echo "Invalid choice, using Vulkan"
                CMAKE_FLAGS="-DGGML_VULKAN=ON"
                ;;
        esac

        # Check for system package manager installation
        echo ""
        echo "Installation options:"
        echo "  1) Build from source (recommended - latest version)"
        echo "  2) Use system package manager (if available)"
        echo ""

        read -p "Select installation method (1-2, default: 1): " install_method
        install_method=${install_method:-1}

        if [ "$install_method" = "2" ]; then
            # Try to install from system package manager
            echo ""
            echo "Detecting package manager..."

            if command -v pacman &> /dev/null; then
                echo "Detected: Arch Linux (pacman)"
                echo "Available packages:"
                echo "  - llama.cpp (CPU)"
                echo "  - llama.cpp-cuda (NVIDIA)"
                echo "  - llama.cpp-rocm (AMD)"
                echo ""
                read -p "Install from AUR? Package name (or 'skip'): " aur_package
                if [ "$aur_package" != "skip" ] && [ -n "$aur_package" ]; then
                    if command -v yay &> /dev/null; then
                        yay -S "$aur_package"
                    elif command -v paru &> /dev/null; then
                        paru -S "$aur_package"
                    else
                        echo "⚠️  AUR helper (yay/paru) not found"
                        echo "   Falling back to source build..."
                        install_method=1
                    fi
                else
                    install_method=1
                fi
            elif command -v apt-get &> /dev/null; then
                echo "Detected: Debian/Ubuntu (apt)"
                echo "Note: llama.cpp not in official repos, building from source..."
                install_method=1
            elif command -v dnf &> /dev/null; then
                echo "Detected: Fedora/RHEL (dnf)"
                echo "Note: llama.cpp not in official repos, building from source..."
                install_method=1
            else
                echo "⚠️  Unknown package manager, building from source..."
                install_method=1
            fi
        fi

        if [ "$install_method" = "1" ]; then
            echo ""
            echo "Building llama.cpp from source..."
            echo ""

            # Install build dependencies
            echo "Installing build dependencies..."
            if command -v pacman &> /dev/null; then
                sudo pacman -S --needed --noconfirm base-devel cmake git
            elif command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y build-essential cmake git
            elif command -v dnf &> /dev/null; then
                sudo dnf groupinstall -y "Development Tools"
                sudo dnf install -y cmake git
            else
                echo "⚠️  Please install: build-essential, cmake, git manually"
                read -p "Press Enter to continue after installing dependencies..."
            fi

            # Clone llama.cpp if not exists
            if [ ! -d "llama.cpp" ]; then
                echo ""
                echo "Cloning llama.cpp repository..."
                git clone https://github.com/ggerganov/llama.cpp.git
            fi

            cd llama.cpp

            # Build llama.cpp
            echo ""
            echo "Building llama.cpp with: $CMAKE_FLAGS"
            echo "⏱️  This will take 2-3 minutes..."
            echo ""

            cmake -B build $CMAKE_FLAGS
            cmake --build build --config Release -j$(nproc)

            cd "$PROJECT_DIR"

            echo ""
            echo "✅ llama.cpp built successfully"
        fi
    fi

    # Create symlinks for Unsloth
    echo ""
    echo "Creating symlinks for Unsloth compatibility..."

    if [ -f "llama.cpp/build/bin/llama-quantize" ]; then
        # Create symlink in llama.cpp root (where Unsloth expects it)
        if [ ! -e "llama.cpp/llama-quantize" ]; then
            cd llama.cpp
            ln -sf build/bin/llama-quantize llama-quantize
            echo "✅ Created: llama.cpp/llama-quantize → build/bin/llama-quantize"
            cd "$PROJECT_DIR"
        else
            echo "✅ Symlink already exists: llama.cpp/llama-quantize"
        fi
    elif [ -f "/usr/bin/llama-quantize" ]; then
        echo "✅ System installation found: /usr/bin/llama-quantize"
    else
        echo "⚠️  llama-quantize not found - GGUF conversion will not work"
    fi

    echo ""
    echo "✅ llama.cpp setup complete"

else
    echo "⏭️  Skipping llama.cpp installation"
    echo "   Note: GGUF conversion will not be available"
fi

# ============================================================
# Step 3: Configuration
# ============================================================
echo ""
echo "============================================================"
echo "⚙️  Step 3: Configuration"
echo "============================================================"
echo ""

# Legacy .env configuration (deprecated but still supported)
if [ ! -f ".env" ]; then
    echo "No .env file found. Creating from example..."

    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
        echo ""
        echo "📝 .env is pre-configured for quick testing (100 samples, 50 steps)"
        echo ""
        echo "⚠️  Note: .env is deprecated. Please use training_params.yaml instead"
        echo ""
        read -p "Press Enter to continue..."
    else
        echo "⚠️  .env.example not found"
        echo "   You'll need to create .env manually"
    fi
else
    echo "✅ .env already exists"
fi

echo ""

# YAML configuration (recommended)
if [ ! -f "training_params.yaml" ]; then
    echo "No training_params.yaml found. Creating from template..."

    if [ -f "training_params_example.yaml" ]; then
        cp training_params_example.yaml training_params.yaml
        echo "✅ Created training_params.yaml from training_params_example.yaml"
        echo ""
        echo "📝 training_params.yaml contains all training configuration:"
        echo "   • Model selection (base model, output formats)"
        echo "   • Dataset configuration (name, config, max_samples)"
        echo "   • Training hyperparameters (LoRA, batch size, learning rate)"
        echo "   • Logging and checkpoint settings"
        echo "   • Benchmark configuration"
        echo ""
        echo "Default configuration:"
        echo "   • Model: Llama-3.2-1B-Instruct (4-bit quantized)"
        echo "   • Dataset: openai/gsm8k (main config)"
        echo "   • LoRA rank: 64, alpha: 128"
        echo "   • Batch size: 4, gradient accumulation: 2"
        echo "   • Learning rate: 3e-4, 3 epochs"
        echo ""
        echo "Edit training_params.yaml to customize your training setup"
        echo ""
        read -p "Press Enter to continue..."
    else
        echo "⚠️  training_params_example.yaml not found"
        echo "   You'll need to create training_params.yaml manually"
    fi
else
    echo "✅ training_params.yaml already exists"
fi

# ============================================================
# Step 4: Verification
# ============================================================
echo ""
echo "============================================================"
echo "🔍 Step 4: Verifying Installation"
echo "============================================================"
echo ""

echo "Checking Python packages..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "❌ PyTorch not found"
python3 -c "import unsloth; print('✅ Unsloth installed')" || echo "❌ Unsloth not found"
python3 -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" || echo "❌ Transformers not found"
python3 -c "import lm_eval; print('✅ lm-eval installed (for benchmarking)')" || echo "⚠️  lm-eval not available (optional - needed for benchmarking)"

echo ""
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  CUDA not available - CPU only mode')
" || echo "⚠️  Could not check GPU"

echo ""
echo "Checking llama.cpp..."
if [ -f "llama.cpp/llama-quantize" ] || [ -f "llama.cpp/build/bin/llama-quantize" ] || command -v llama-quantize &> /dev/null; then
    echo "✅ llama.cpp found - GGUF conversion available"

    # Show llama.cpp build info
    if [ -f "llama.cpp/build/bin/llama-quantize" ]; then
        echo ""
        echo "llama.cpp build info:"
        llama.cpp/build/bin/llama-quantize 2>&1 | grep -i "usage" | head -1 || true

        # Detect backend
        if ldd llama.cpp/build/bin/llama-quantize 2>/dev/null | grep -q "vulkan"; then
            echo "   Backend: Vulkan"
        elif ldd llama.cpp/build/bin/llama-quantize 2>/dev/null | grep -q "cuda"; then
            echo "   Backend: CUDA"
        elif ldd llama.cpp/build/bin/llama-quantize 2>/dev/null | grep -q "hip"; then
            echo "   Backend: ROCm"
        else
            echo "   Backend: CPU"
        fi
    fi
else
    echo "⚠️  llama.cpp not found - GGUF conversion not available"
    echo "   Set OUTPUT_FORMATS=lora_only in .env"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE"
echo "============================================================"
echo ""
echo "📁 Project directory: $PROJECT_DIR"
echo ""
echo "🚀 Next steps:"
echo ""
echo "1. Configure your training:"
echo "   vim training_params.yaml"
echo "   (or edit .env for legacy configuration)"
echo ""
echo "2. (Optional) Test your setup:"
echo "   python test_setup.py"
echo ""
echo "3. Start training:"
echo "   python scripts/train.py"
echo ""
echo "4. Convert to other formats:"
echo "   python scripts/build.py"
echo ""
echo "5. Benchmark your model:"
echo "   python scripts/benchmark.py"
echo ""
echo "📖 Documentation:"
echo "   • docs/INSTALLATION.md - Detailed setup guide"
echo "   • docs/TRAINING.md - Training guide and workflow"
echo "   • docs/CONFIGURATION.md - Configuration options"
echo "   • docs/BENCHMARK.md - Benchmarking guide"
echo "   • docs/FAQ.md - Frequently asked questions"
echo "   • README.md - Project overview"
echo ""
echo "============================================================"
