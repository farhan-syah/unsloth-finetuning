#!/bin/bash
# Install llama.cpp for GGUF conversion on Arch Linux

set -e

echo "============================================================"
echo "📦 Installing llama.cpp for GGUF conversion"
echo "============================================================"
echo ""

# Check if we're in the unsloth directory
if [ ! -f "build.py" ]; then
    echo "❌ Error: Must run from unsloth project directory"
    exit 1
fi

# Install build dependencies for Arch Linux
echo "Step 1: Installing build dependencies..."
echo "This requires sudo access for pacman"
sudo pacman -S --needed --noconfirm base-devel cmake git

echo ""
echo "Step 2: Cloning llama.cpp..."
if [ -d "llama.cpp" ]; then
    echo "✅ llama.cpp directory already exists"
else
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

echo ""
echo "Step 3: Building llama.cpp with CUDA support..."
echo "⏱️  This will take 2-3 minutes..."

# Build with CUDA support
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build --config Release -j$(nproc)

cd ..

echo ""
echo "============================================================"
echo "✅ llama.cpp installed successfully"
echo "============================================================"
echo ""
echo "Location: $(pwd)/llama.cpp"
echo "Quantizer: $(pwd)/llama.cpp/build/bin/llama-quantize"
echo "Converter: $(pwd)/llama.cpp/build/bin/llama-convert-hf-to-gguf.py"
echo ""
echo "You can now run: python build.py"
