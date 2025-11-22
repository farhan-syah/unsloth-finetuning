# Unsloth Fine-tuning Pipeline

A streamlined pipeline for LLM fine-tuning using Unsloth and LoRA. Includes training, model merging, and GGUF conversion.

## 📖 What is Fine-tuning?

**Fine-tuning** is the process of taking a pre-trained AI model (like Llama, Qwen, or Gemma) and teaching it new skills or behaviors using your own data. Think of it like this:

- **Base Model:** A general-purpose AI that knows a lot about everything
- **Fine-tuned Model:** The same AI, now specialized for YOUR specific task

### Why Fine-tune?

1. **Customize behavior:** Make the model respond in a specific style, tone, or format
2. **Domain expertise:** Teach it specialized knowledge (medical, legal, technical, etc.)
3. **Task-specific:** Train for specific tasks (customer support, code generation, translation)
4. **Data privacy:** Keep your data private by training locally instead of using APIs
5. **Cost savings:** Run your own specialized model instead of paying for API calls

### Common Use Cases

- 💬 **Custom chatbots:** Train on your company's knowledge base
- 📝 **Content generation:** Generate content in your brand's voice
- 🔧 **Code assistants:** Fine-tune on your codebase or specific frameworks
- 🌍 **Language adaptation:** Improve performance on specific languages or dialects
- 📊 **Data extraction:** Train for structured data extraction from documents

## 🦥 Why Unsloth?

[Unsloth](https://github.com/unslothai/unsloth) is an optimization library that makes fine-tuning **2x faster** and uses **up to 80% less VRAM**:

- **Faster training:** What takes hours with standard tools takes minutes with Unsloth
- **Lower memory:** Train 8B models on consumer GPUs (RTX 3060, 4060)
- **Same quality:** Produces identical results to standard fine-tuning methods
- **Easy to use:** Drop-in replacement for Hugging Face Transformers

### Real-world Comparison

| Method | Training Time | VRAM Usage | Cost |
|--------|---------------|------------|------|
| Standard PyTorch | 4 hours | 24GB VRAM | Cloud GPU: $4-8 |
| **Unsloth** | **2 hours** | **12GB VRAM** | **Local GPU: Free** |

## 🎯 Why This Pipeline?

Most fine-tuning tutorials are complex and require deep ML knowledge. This pipeline makes it **beginner-friendly**:

### 1. **One-Command Setup**
```bash
bash setup.sh  # Installs everything automatically
```
No need to manually install PyTorch, CUDA versions, or dependencies - it's all automated.

### 2. **Simple Configuration**
Everything is controlled by a single `.env` file:
```bash
cp .env.example .env
vim .env  # Modify required settings
```
No Python coding required - just edit text configuration.

### 3. **Automatic Format Conversion**
- **LoRA adapters** (small, efficient)
- **Merged models** (full model, ready to use)
- **GGUF quantization** (for Ollama, llama.cpp)

One command creates all formats you need.

### 4. **Smart Caching**
The pipeline remembers what it's already done:
- Already trained? Skip training
- Already merged? Skip merging
- Already quantized? Skip quantization

### 5. **Dataset Flexibility**
Works with any HuggingFace dataset in SFT (Supervised Fine-Tuning) format:
- High-quality curated datasets (1K-5K samples)
- Large instruction datasets (10K-100K samples)
- Your own custom dataset on HuggingFace

### What is SFT Format?

SFT (Supervised Fine-Tuning) format is a structured conversation format:

```json
{
  "conversations": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

Or simpler format:
```json
{
  "instruction": "What is machine learning?",
  "output": "Machine learning is..."
}
```

The pipeline automatically detects and converts between these formats.

## ✨ Features

- 🚀 **2x faster training** with Unsloth
- 💾 **Low VRAM usage** - Train 2B models with ~4-6GB VRAM (depends on dataset size)
- 🎯 **One-command setup** - Automated installation
- 🔄 **Smart caching** - Skip completed steps automatically
- 📦 **Multiple formats** - LoRA, merged safetensors, GGUF quantizations
- 🔧 **Single config file** - Everything controlled via `.env`
- 🤖 **Auto-detection** - Chat templates, dataset formats, and model types detected automatically

## 🚀 Quick Start

Choose your environment:

### Option A: Google Colab

**Use case:** Cloud-based training with access to T4, A100, or V100 GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farhan-syah/unsloth-finetuning/blob/main/colab_train.ipynb)

Click the badge above or:
1. Open [`colab_train.ipynb`](colab_train.ipynb) in Google Colab
2. Runtime → Change runtime type → **GPU (T4 free, or A100/V100 with Colab Pro)**
3. Edit configuration in Step 3
4. Run all cells (Shift+Enter)
5. Download to Google Drive or push to HuggingFace

**Features:**
- GPU access: T4 (free), A100/V100 (Colab Pro)
- No local installation required
- Automated dependency management
- Model export to Google Drive or HuggingFace

**Setup time:** Approximately 10-15 minutes (5 min setup + training)

See [Google Colab Setup](#-google-colab-setup) for detailed instructions.

---

### Option B: Local Setup

**Use case:** Training on local hardware, offline environments, or when you need full system control

**Setup time:** Approximately 7 minutes for quick test (2 min training + 5 min conversion on GTX 1660-class GPU)

### 1. Setup (One-time, ~10 minutes)

```bash
# Clone repository
git clone https://github.com/farhan-syah/unsloth-finetuning.git
cd unsloth-finetuning

# Run automated setup (installs Python packages, llama.cpp, etc.)
bash setup.sh
```

The script will:
- ✅ Install PyTorch, Unsloth, and dependencies
- ✅ Build llama.cpp for GGUF conversion
- ✅ Create a `.env` configuration file

### 2. Configure (30 seconds)

The `.env.example` is already configured for a quick test. For your first run, just copy it:

```bash
cp .env.example .env
```

**Default configuration:**
- Use a 2B model (fits on 6-8GB GPUs)
- Train on 100 samples from Alpaca dataset
- Run for 50 steps (~2 minutes)

**To customize:**
```bash
vim .env  # or nano, code, etc.
```

Key settings to change:
```bash
LORA_BASE_MODEL=unsloth/Llama-3.2-1B-Instruct-bnb-4bit  # Pick your model
DATASET_NAME=your-dataset/name                          # Pick your dataset
OUTPUT_FORMATS=gguf_q4_k_m                              # Choose output format
```

See [Configuration Guide](docs/CONFIGURATION.md) for all options.

### 3. Train (2 minutes)

```bash
python scripts/train.py
```

**What happens:**
1. Downloads the base model (~2.4GB for 2B model)
2. Loads and preprocesses your dataset
3. Trains LoRA adapters on your data
4. Saves adapters to `outputs/{model-name}/lora/`

**Output:**
- `outputs/{model-name}-{dataset-name}/lora/` - LoRA adapters (~50-200MB)

These adapters are lightweight and can be applied to the base model.

### 4. Build (5 minutes)

```bash
python scripts/build.py
```

**What happens:**
1. Loads base model + LoRA adapters
2. Merges them into a complete model
3. (Optional) Converts to GGUF for Ollama/llama.cpp

**Output:**
- `merged_16bit/` - Complete merged model (size varies by model)
- `gguf/` - GGUF quantized versions (if `OUTPUT_FORMATS` set)
  - `model.Q4_K_M.gguf` - 4-bit quantization (typically 50-60% of original size)

### 5. Use Your Model

**With Ollama:**
```bash
cd outputs/{model-name}-{dataset-name}/merged_16bit
ollama create my-model -f Modelfile
ollama run my-model "Your prompt here"
```

**With Python:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "outputs/{model-name}-{dataset-name}/merged_16bit"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

**Quality Note:** By default, merging uses the 4-bit training base. For true 16-bit quality, set `INFERENCE_BASE_MODEL` to the original unquantized model in `.env` (requires 20-30GB VRAM during build).

## 📋 Requirements

- **GPU:** NVIDIA with 6GB+ VRAM (AMD via ROCm, or Apple Silicon also supported)
  - 2B models: ~4-6GB VRAM (depending on dataset and batch size)
  - 4B models: ~8-12GB VRAM
  - 8B models: ~16-24GB VRAM
- **Python:** 3.10+ (3.12 recommended)
- **Disk:** ~15GB free space for model + outputs
- **OS:** Linux, macOS, Windows (WSL2)

**Real VRAM usage examples** (2B model, LoRA r=16):
- Quick test (100 samples): ~4GB VRAM
- Full training (10K+ samples): ~6GB VRAM

## 📊 Choosing Your Dataset

The pipeline works with any HuggingFace dataset in **SFT (Supervised Fine-Tuning) format**. Here's what you need to know:

### Popular Pre-made Datasets

| Dataset Type | Size Range | Best For | Examples |
|--------------|------------|----------|----------|
| High-quality curated | 1K-5K | Best quality, prevents catastrophic forgetting | GAIR/lima, Others |
| Large instruction | 10K-100K | General instruction-following | yahma/alpaca-cleaned, databricks/databricks-dolly-15k |
| Domain-specific | Varies | Specialized tasks | Medical, legal, code datasets |

Just set in `.env`:
```bash
DATASET_NAME=your-dataset/name  # Any HuggingFace dataset
```

### Using Your Own Dataset

Upload your dataset to HuggingFace, then use its name. Your dataset should be in one of these formats:

**Format 1: Conversations** (Recommended for chat models)
```json
{
  "conversations": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

**Format 2: Instruction-Output**
```json
{
  "instruction": "What is Python?",
  "output": "Python is a programming language..."
}
```

**Format 3: Input-Output with Context**
```json
{
  "instruction": "Translate to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

The pipeline handles format detection and conversion automatically.

### 🤖 Smart Auto-Detection

The pipeline automatically detects and configures many settings for you:

**Chat Template Auto-Detection**
- Detects model type from name (Llama, Qwen, Phi, Gemma, etc.)
- Automatically sets correct chat template:
  - Llama-3.1/3.2: Uses `llama-3.1` template (official Unsloth format)
  - Qwen2.5: Uses `qwen2.5` template
  - Phi-3: Uses `phi-3` template
  - Gemma: Uses `gemma` template
- Base models without templates are configured automatically

**Dataset Format Detection**
- Alternating string conversations (LIMA format)
- ShareGPT format (`{"from": "human", "value": "..."}`)
- Standard messages format (`{"role": "user", "content": "..."}`)
- Instruction-output formats
- Automatically converts between formats

**Model Type Detection**
- Reasoning models (Qwen3 with `<think>` tags)
- Instruct vs Base models
- Gated datasets (prompts for HuggingFace login)

**Smart Recommendations**
Running `python scripts/preprocess.py` analyzes your dataset and recommends:
- Optimal `BATCH_SIZE` based on GPU memory
- Recommended `MAX_STEPS` for 1-3 epochs
- Sequence length statistics
- Dataset compatibility warnings

### Dataset Tips

1. **Quality over quantity:** 1,000 high-quality examples > 10,000 poor examples
2. **Consistency:** Keep formatting consistent across all examples
3. **Diversity:** Include varied examples covering your use case
4. **Length:** Most examples should fit in MAX_SEQ_LENGTH (default: 4096 tokens)
5. **Model selection:** Use Instruct models for small datasets (<5K), Base models for large datasets (>10K)

See [docs/TRAINING.md](docs/TRAINING.md) for detailed dataset guidance.

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) - Detailed local setup instructions
- [Configuration Guide](docs/CONFIGURATION.md) - .env options explained
- [Training Guide](docs/TRAINING.md) - Training tips and troubleshooting
- [Distribution Guide](docs/DISTRIBUTION.md) - Sharing models on HuggingFace
- [FAQ](docs/FAQ.md) - Common questions and solutions

## ☁️ Google Colab Setup

### Why Use Colab?

Google Colab provides access to enterprise-grade GPUs suitable for LLM fine-tuning:

| Tier | GPU | VRAM | Cost | Best For |
|------|-----|------|------|----------|
| **Free** | Tesla T4 | 15GB | Free | 2B-4B models, testing |
| **Colab Pro** | A100 | 40GB | $10/month | 8B-14B models, production |
| **Colab Pro+** | A100 | 40GB | $50/month | Extended sessions, heavy use |

**Compared to local setup:**
- Access to enterprise-grade GPUs (T4, A100)
- No hardware purchase required
- Reduced training time on equivalent workloads
- Simplified notebook sharing
- Managed dependencies

### Quick Colab Guide

**Step 1: Open the Notebook**

Click here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farhan-syah/unsloth-finetuning/blob/main/colab_train.ipynb)

Or manually:
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Open notebook → GitHub
3. Enter: `farhan-syah/unsloth-finetuning`
4. Select `colab_train.ipynb`

**Step 2: Enable GPU**

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. GPU type → **T4** (free) or **A100** (Pro)
4. Save

**Step 3: Configure Your Training**

In the notebook, edit Step 3 configuration:

```python
# Quick test (30 minutes - full LIMA dataset)
LORA_BASE_MODEL = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
DATASET_NAME = "GAIR/lima"
MAX_STEPS = 0              # Full epoch
DATASET_MAX_SAMPLES = 0    # All 1K samples

# Full production training
# MAX_STEPS = 0          # Full epochs
# DATASET_MAX_SAMPLES = 0  # All samples
# LORA_RANK = 64         # Better quality
```

**Step 4: Run Training**

1. Click Runtime → Run all (or press Ctrl+F9)
2. Wait for setup (~5 minutes)
3. Training will start automatically
4. Monitor progress in real-time

**Step 5: Download Your Model**

Three options:

**Option A: Google Drive** (Recommended)
```python
# In Step 7 of notebook
from google.colab import drive
drive.mount('/content/drive')
!cp -r outputs/* /content/drive/MyDrive/unsloth-models/
```

**Option B: Download directly**
```python
# For small files (LoRA adapters)
from google.colab import files
!zip -r model.zip outputs/
files.download('model.zip')
```

**Option C: Push to HuggingFace**
```python
# In Step 7 of notebook, uncomment HuggingFace section
# Requires HF token from https://huggingface.co/settings/tokens
```

### Colab Tips

**💡 Maximize Free Tier:**
- Use quick tests (100 samples) during development
- Full training only when ready
- Download models immediately (sessions time out)
- Don't leave notebook idle

**⚡ Speed Up Training:**
- Use smaller datasets for testing
- Increase `BATCH_SIZE` if VRAM allows
- Set `CHECK_SEQ_LENGTH=false`
- Use `SAVE_ONLY_FINAL=true`

**🔥 Avoid Disconnects:**
- Keep browser tab open
- Use Colab Pro for longer sessions
- Save checkpoints regularly
- Backup to Google Drive frequently

**🐛 Common Colab Issues:**

**"Runtime disconnected"**
- Solution: Your session timed out. Restart and resume from checkpoint

**"Out of memory"**
- Solution: Reduce `BATCH_SIZE` to 1, or use smaller model

**"GPU not available"**
- Solution: Check Runtime → Change runtime type → GPU is selected

**"Installation failed"**
- Solution: Runtime → Restart runtime, then run installation cell again

### Colab vs Local Comparison

| Feature | Google Colab Free | Google Colab Pro | Local (GTX 1660) |
|---------|-------------------|------------------|------------------|
| GPU | T4 (15GB) | A100 (40GB) | GTX 1660 (6GB) |
| 2B Model Training | 5-10 min | 3-5 min | 20-30 min |
| Max Model Size | 4B models | 14B models | 2B models |
| Cost | Free | $10/month | One-time hardware |
| Session Limit | ~12 hours | ~24 hours | Unlimited |
| Internet Required | Yes | Yes | No |
| Setup Time | 5 min | 5 min | 10-15 min |

**Recommendation:**
- **Starting out?** Use Colab Free
- **Limited local VRAM?** Train in Colab, convert GGUF locally (GGUF is CPU-only)
- **Serious about fine-tuning?** Get Colab Pro ($10/month is cheaper than buying a GPU)
- **Privacy sensitive?** Use local setup
- **No internet?** Use local setup

### Hybrid Workflow (Best for Limited VRAM)

If your local GPU has limited VRAM but you want GGUF models:

1. **Train in Colab** - Use powerful T4/A100 GPUs
2. **Merge in Colab** - Create merged safetensors model
3. **Download to Google Drive** - Save the merged model
4. **Convert to GGUF locally** - CPU-only conversion (no GPU needed)

This workflow lets you use Colab's powerful GPUs for training while doing GGUF conversion on your local machine without needing GPU VRAM.

## 🎯 Advanced Usage

### Production Training

For full training on the entire dataset (not just a quick test):

```bash
# Edit .env and adjust these settings:
vim .env
```

Change from test mode to production:
```bash
# Test settings (default in .env.example)
MAX_STEPS=50              # ← Change to 0 for full training
DATASET_MAX_SAMPLES=100   # ← Change to 0 to use all samples
LORA_RANK=16              # ← Change to 64 for better quality
LORA_ALPHA=32             # ← Change to 128 for better quality

# Production settings
MAX_STEPS=0               # Train for full epochs
DATASET_MAX_SAMPLES=0     # Use entire dataset
LORA_RANK=64              # Higher quality adapters
LORA_ALPHA=128            # Stronger adaptation
SAVE_ONLY_FINAL=false     # Save checkpoints during training
```

Then run:
```bash
python scripts/train.py  # Hours (depends on dataset size)
python scripts/build.py  # Minutes per quantization
```

### Using Different Models

The pipeline supports any Unsloth-compatible model. Choose based on your GPU:

```bash
# In .env, change LORA_BASE_MODEL:

# 6GB+ VRAM (GTX 1660, RTX 3050, RTX 4060)
LORA_BASE_MODEL=unsloth/Llama-3.2-1B-Instruct-bnb-4bit
# Typical usage: 4-6GB depending on dataset size

# 8GB+ VRAM (RTX 3060, RTX 4060)
LORA_BASE_MODEL=unsloth/Llama-3.2-3B-Instruct-bnb-4bit
# Typical usage: 6-8GB

# 16GB+ VRAM (RTX 3090, RTX 4080)
LORA_BASE_MODEL=unsloth/Llama-3.1-8B-Instruct-bnb-4bit
# Typical usage: 12-16GB

# Browse more at: https://huggingface.co/unsloth
```

**Note:** VRAM usage varies based on:
- Dataset size (more samples = more VRAM during data loading)
- Batch size and gradient accumulation
- MAX_SEQ_LENGTH (longer sequences = more VRAM)
- LoRA rank (higher rank = more VRAM)

## 🔧 Configuration Highlights

Key settings in `.env`:

```bash
# Model Selection (choose based on VRAM)
LORA_BASE_MODEL=unsloth/Llama-3.2-1B-Instruct-bnb-4bit  # 4-6GB VRAM
# Other options: Llama-3.2-3B, Phi-3.5, Qwen2.5-1.5B, etc.

# Optional: Use unquantized base for merging (better quality, more VRAM during build)
INFERENCE_BASE_MODEL=  # Empty = use LORA_BASE_MODEL (4-bit)
# INFERENCE_BASE_MODEL=unsloth/Llama-3.2-1B-Instruct  # Unquantized (requires 15-20GB VRAM for build)

# Output name for directories and files
OUTPUT_MODEL_NAME=auto  # Auto = model + dataset name
# OUTPUT_MODEL_NAME=my-chatbot-v1  # Custom name for better organization

# Dataset
DATASET_NAME=your-dataset/name  # Any HuggingFace dataset

# Training (Quick Test)
MAX_STEPS=50              # Train for 50 steps only
DATASET_MAX_SAMPLES=100   # Use 100 samples only

# Training (Full)
MAX_STEPS=0               # Train for full epochs
DATASET_MAX_SAMPLES=0     # Use all samples

# Output Formats
OUTPUT_FORMATS=gguf_q4_k_m,gguf_q5_k_m  # Create Q4_K_M and Q5_K_M GGUF

# Author Attribution
AUTHOR_NAME=Your Name      # Your name for model cards and citations

# Performance
CHECK_SEQ_LENGTH=false    # Skip length checking (faster preprocessing)
FORCE_PREPROCESS=false    # Use cached dataset if available
```

See [Configuration Guide](docs/CONFIGURATION.md) for all options.

## 📊 Output Structure

```
outputs/Qwen3-VL-2B-Instruct-alpaca-cleaned/  # Auto-generated: {model}-{dataset}
├── lora/              # LoRA adapters (~50-100MB)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer files
├── merged_16bit/      # Full merged model (~5GB)
│   ├── model-*.safetensors
│   ├── Modelfile
│   └── tokenizer files
└── gguf/              # GGUF quantizations
    ├── model.Q4_K_M.gguf   (~1.5GB)
    ├── model.Q5_K_M.gguf   (~1.8GB)
    └── tokenizer files
```

**Note:** Output directory name is controlled by `OUTPUT_MODEL_NAME` in `.env`. Set to `auto` for automatic naming or provide a custom name (e.g., `my-chatbot-v1`).

## 🔧 Common Issues & Solutions

### "Out of Memory" Error

**Problem:** GPU runs out of VRAM during training

**Solutions:**
1. Reduce `MAX_SEQ_LENGTH` from 4096 to 1024 or 2048 in `.env`
2. Reduce `BATCH_SIZE` from 2 to 1
3. Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
4. Use a smaller model (e.g., 1.7B instead of 4B)

### "Model not found" Error

**Problem:** Dataset name is incorrect or private

**Solutions:**
1. Check the exact dataset name on HuggingFace
2. For gated datasets like LIMA, run `python scripts/preprocess.py` (auto-prompts login)
3. Try ungated alternative: `yahma/alpaca-cleaned`

### Training is Too Slow

**Problem:** Training takes too long

**Solutions:**
1. Set `CHECK_SEQ_LENGTH=false` in `.env` (skip length checking)
2. Reduce `DATASET_MAX_SAMPLES` for testing
3. Increase `MAX_STEPS` to limit training duration
4. Ensure you're using a 4-bit quantized model (`-bnb-4bit`)

### "Flash Attention failed" Warning

**Problem:** Flash Attention installation failed during setup

**Solution:** The pipeline will automatically fall back to xformers, which provides stable performance with approximately 10% slower training speed.

For more troubleshooting, see [docs/FAQ.md](docs/FAQ.md) and [docs/TRAINING.md](docs/TRAINING.md).

## 🤝 Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

## 👤 Author

**Farhan Syah**
- GitHub: [@farhan-syah](https://github.com/farhan-syah)
- Repository: [farhan-syah/unsloth-finetuning](https://github.com/farhan-syah/unsloth-finetuning)

## 🙏 Acknowledgments

Built with:
- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster LLM fine-tuning
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF quantization
- [Transformers](https://github.com/huggingface/transformers) - Model library

## 📬 Support

- **Issues:** [GitHub Issues](https://github.com/farhan-syah/unsloth-finetuning/issues)
- **Discussions:** [GitHub Discussions](https://github.com/farhan-syah/unsloth-finetuning/discussions)
- **Documentation:** [docs/](docs/)

---
