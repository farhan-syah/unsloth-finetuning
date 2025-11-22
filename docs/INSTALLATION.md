# Installation Guide

## Quick Start (Recommended)

```bash
# Run the automated installer
bash setup.sh
```

That's it! The script handles everything - Python dependencies, llama.cpp, and configuration.

---

## What the Script Does

### 1. Cleanup (Step 1/7)
- Removes corrupted torch (`~orch`) if it exists
- Cleans pip cache to avoid download issues

### 2. Pip Upgrade (Step 2/7)
- Updates pip to latest version

### 3. Core ML Frameworks (Step 3/7)
- Installs: `trl`, `peft`, `bitsandbytes`, `transformers`
- Uses `--timeout 600 --retries 5` for large packages
- bitsandbytes is ~137MB - may take a while

### 4. PyTorch with CUDA (Step 4/7)
- Installs: `torch==2.8.0`, `torchvision`
- Downloads ~2GB from CUDA index (auto-detects your CUDA version)
- Uses `--timeout 600` for slow connections

### 5. Unsloth (Step 5/7)
- Installs from specific git commit
- Ensures compatibility with other packages

### 6. xformers (Step 6/7) ⭐ **CRITICAL**
- **Fallback attention mechanism** when Flash Attention doesn't work
- Uses `--no-deps --force-reinstall` to prevent PyTorch downgrade
- **Must succeed** - training won't work without it
- Has retry logic if first attempt fails

### 7. Flash Attention 2 (Step 7/7) ⚙️ **OPTIONAL**
- Optimal performance but not required
- Compiles C++/CUDA code (5-10 minutes)
- If this fails, training still works with xformers
- Uses `--timeout 1200` (20 minutes) for compilation

---

## Two-Tier Fallback System

```
┌─────────────────────────┐
│  Flash Attention 2      │  ← Best performance (optional)
│  (if GPU supports it)   │
└─────────┬───────────────┘
          │ Fallback
          ▼
┌─────────────────────────┐
│  xformers               │  ← Required fallback (must work)
│  (works on all GPUs)    │
└─────────────────────────┘
```

### Flash Attention 2 (FA2)
- ✅ Fastest attention implementation
- ✅ Best memory efficiency
- ❌ Requires newer GPU/CUDA
- ❌ C++ compilation can fail
- **If broken**: Unsloth uses xformers automatically

### xformers
- ✅ Works on all CUDA GPUs
- ✅ No compilation required
- ✅ Stable and reliable
- ⚠️  Slightly slower than FA2 (~10-15%)
- **This MUST work** - training depends on it

---

## Common Issues

### Issue 1: Network Timeout (bitsandbytes)

**Error**:
```
TimeoutError: The read operation timed out
```

**Solution**:
```bash
# Install with longer timeout
pip install --timeout 600 --retries 5 bitsandbytes==0.43.3

# Then run the script
bash install_dependencies.sh
```

### Issue 2: Broken Torch (`~orch`)

**Error**:
```
WARNING: Ignoring invalid distribution ~orch
```

**Solution**:
```bash
# Find your Python site-packages directory
python -c "import site; print(site.getsitepackages()[0])"

# Remove corrupted package (replace /path/to with your actual path)
rm -rf /path/to/site-packages/~orch

# Reinstall
bash install_dependencies.sh
```

### Issue 3: xformers Failed

**Error**:
```
❌ xformers: FAILED
```

**Solution**:
```bash
# Try manual installation
pip uninstall -y xformers
pip install --no-deps xformers==0.0.25.post1

# Verify
python -c "import xformers; print('OK')"
```

### Issue 4: Flash Attention Compilation Failed

**Error**:
```
error: command 'gcc' failed
```

**Solution**:
```bash
# This is OK! xformers will be used instead
# Training will still work

# If you want to fix it, install build tools for your distro:
# Ubuntu/Debian: sudo apt-get install build-essential
# Arch: sudo pacman -S base-devel
# Fedora: sudo dnf install @development-tools
# macOS: xcode-select --install

pip install flash-attn==2.6.3
```

---

## Verification

After installation completes, you should see:

```
✅ PyTorch: 2.8.0+cu128
✅ Unsloth: OK
✅ Transformers: 4.46.0+
✅ TRL: 0.12.0+
✅ PEFT: 0.13.0+

✅ xformers: OK (fallback attention)
✅ Flash Attention 2: OK (optimal performance)
   OR
⚠️  Flash Attention 2: Not available
   Will use xformers instead (slightly slower but works)

✅ CUDA available: GeForce RTX 3060
   CUDA version: 12.8
   Memory: 12.0 GB
```

**Minimum required** (for training to work):
- ✅ PyTorch with CUDA
- ✅ Unsloth
- ✅ **xformers** ← Must be OK
- ✅ CUDA available

**Nice to have** (for best performance):
- ✅ Flash Attention 2

---

## What If Installation Fails?

### Nuclear Option: Fresh Environment

```bash
# Option 1: Python venv (built-in, no extras needed)
python3.11 -m venv ~/.venv/unsloth
source ~/.venv/unsloth/bin/activate
cd /path/to/unsloth
bash install_dependencies.sh

# Option 2: Conda environment
conda create -n unsloth python=3.11 -y
conda activate unsloth
bash install_dependencies.sh

# Option 3: pyenv (if you use it)
pyenv virtualenv 3.11.10 unsloth
pyenv activate unsloth
bash install_dependencies.sh

# Option 3: New venv
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
bash install_dependencies.sh
```

### Manual Installation (if script fails)

Follow the exact order in `docs/installation_order.md`:

```bash
# 1. Core frameworks
pip install --timeout 600 --retries 5 "trl>=0.12.0" "peft>=0.13.0" "bitsandbytes>=0.45.0" "transformers[sentencepiece]>=4.46.0"

# 2. PyTorch
pip install --timeout 600 torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Unsloth
pip install --timeout 600 "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 4. xformers (CRITICAL - must use --no-deps)
pip install --no-deps --timeout 600 "xformers>=0.0.32,<0.0.33" --index-url https://download.pytorch.org/whl/cu128

# 5. Flash Attention (optional)
pip install --timeout 1200 flash-attn

# 6. Additional
pip install datasets huggingface_hub accelerate sentencepiece protobuf python-dotenv
```

---

## Testing Installation

```bash
# Quick test
python -c "
import torch
import unsloth
import xformers
print('✅ All critical packages OK')
print(f'CUDA: {torch.cuda.is_available()}')
"

# Full test (runs training for quick test)
python scripts/train.py  # Uses .env config (copy from .env.example)
```

---

## Why Not `pip install -r requirements.txt`?

❌ **Don't do this**:
```bash
pip install -r requirements.txt  # Wrong!
```

**Problems**:
1. Pip installs in arbitrary order
2. xformers downgrades PyTorch
3. Flash Attention compiles against wrong PyTorch
4. No timeout handling for large packages
5. No cleanup of broken packages

✅ **Do this instead**:
```bash
bash install_dependencies.sh  # Correct!
```

**Benefits**:
1. Installs in correct order
2. xformers uses `--no-deps` flag
3. Flash Attention compiles last
4. Handles timeouts and retries
5. Cleans up broken packages
6. Verifies each step
7. Graceful fallback if FA2 fails

---

## Summary

| Component | Required? | Installation Time | Fallback |
|-----------|-----------|-------------------|----------|
| PyTorch | ✅ Yes | 2-5 min | None |
| Unsloth | ✅ Yes | 1-2 min | None |
| **xformers** | **✅ Yes** | **1-2 min** | **None** |
| Flash Attention 2 | ⚙️ Optional | 5-10 min | xformers |
| bitsandbytes | ✅ Yes | 1-3 min | None |
| transformers, trl, peft | ✅ Yes | 1-2 min | None |

**Total time**: 10-25 minutes (depending on network and CPU)

**Critical**: xformers must install successfully. Everything else can fail and still work.
