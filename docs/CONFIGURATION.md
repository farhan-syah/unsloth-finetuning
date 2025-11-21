# Configuration Guide

Complete reference for `.env` configuration options.

## Quick Setup

```bash
cp .env.example .env
vim .env  # Edit settings
```

## Model Selection

```bash
# Training base model (required)
LORA_BASE_MODEL=unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit

# Optional: Different base for merging (for best quality)
INFERENCE_BASE_MODEL=  # Empty = use LORA_BASE_MODEL
# INFERENCE_BASE_MODEL=Qwen/Qwen2.5-VL-2B-Instruct  # 16-bit for best quality

# Output naming (optional)
OUTPUT_MODEL_NAME=auto  # Auto-generates from model + dataset
# OUTPUT_MODEL_NAME=my-custom-name  # Or set custom name
```

Choose LORA_BASE_MODEL based on your VRAM:

| Model | VRAM Required | GPU Examples |
|-------|---------------|--------------|
| Qwen3-VL-2B-Instruct | 6-8GB | GTX 1660, RTX 3050 |
| Qwen3-4B | 12GB+ | RTX 3060, RTX 4060 Ti |
| Qwen3-8B | 24GB+ | RTX 3090, RTX 4090 |
| Qwen3-14B | 40GB+ | A100, H100 |

## Dataset Configuration

```bash
DATASET_NAME=yahma/alpaca-cleaned
DATASET_MAX_SAMPLES=0    # 0 = use all, >0 = limit for testing
```

Popular datasets:
- `yahma/alpaca-cleaned` - Cleaned Alpaca instructions (52K samples)
- `OpenAssistant/oasst1` - Multilingual conversations (88K samples)
- `tatsu-lab/alpaca` - Original Alpaca dataset
- `timdettmers/openassistant-guanaco` - High-quality instruction dataset
- Your own HuggingFace dataset

## Training Parameters

### LoRA Settings

```bash
LORA_RANK=64        # LoRA rank (r): 16, 32, 64, 128
LORA_ALPHA=128      # LoRA alpha: typically 2x rank
```

Higher rank = more trainable parameters = better quality but larger adapter size.

### Training Control

```bash
NUM_TRAIN_EPOCHS=1              # Number of epochs
MAX_STEPS=0                     # 0 = use epochs, >0 = train for N steps only
BATCH_SIZE=2                    # Per-device batch size
GRADIENT_ACCUMULATION_STEPS=4   # Effective batch = BATCH_SIZE × this
LEARNING_RATE=2e-4              # Learning rate
WARMUP_STEPS=5                  # Warmup steps
```

**Effective Batch Size:** `BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS`
- Example: `2 × 4 = 8` effective batch size

### Sequence Length

```bash
MAX_SEQ_LENGTH=2048  # Maximum tokens per sample
```

Common values: 512, 1024, 2048, 4096, 8192
- Higher = longer context but more VRAM needed

### Optimization

```bash
USE_GRADIENT_CHECKPOINTING=true  # Save VRAM (slower but uses less memory)
MAX_GRAD_NORM=1.0                # Gradient clipping
OPTIM=adamw_8bit                 # Optimizer (adamw_8bit recommended)
PACKING=false                    # Pack multiple samples (not recommended for chat)
```

## Preprocessing Control

```bash
CHECK_SEQ_LENGTH=false    # Set true to filter samples exceeding MAX_SEQ_LENGTH
FORCE_PREPROCESS=false    # Set true to reprocess dataset (ignores cache)
```

**When to use `CHECK_SEQ_LENGTH=true`:**
- First run with a new dataset
- Want to see length statistics
- Dataset has very long samples

**When to set `FORCE_PREPROCESS=true`:**
- Changed MAX_SEQ_LENGTH
- Changed dataset chat template
- Preprocessed data is corrupted

## Training Control Flags

```bash
FORCE_RETRAIN=false   # Set true to retrain even if LoRA adapters exist
FORCE_REBUILD=false   # Set true to rebuild formats even if they exist
```

## Output Formats

```bash
OUTPUT_FORMATS=gguf_q4_k_m,gguf_q5_k_m
```

Available formats:

| Format | Description | Size (Qwen3-4B) | Use Case |
|--------|-------------|-----------------|----------|
| *(empty)* | LoRA + merged_16bit only | 7.7GB | Just training |
| `merged_4bit` | 4-bit safetensors | 3.4GB | HuggingFace inference |
| `gguf_q4_k_m` | Q4_K_M quantization | 2.4GB | **Ollama (recommended)** |
| `gguf_q5_k_m` | Q5_K_M quantization | 3.0GB | Better quality |
| `gguf_q8_0` | Q8_0 quantization | 4.0GB | High quality |
| `gguf_f16` | F16 (no quantization) | 7.6GB | Full precision |

**Multiple formats:**
```bash
OUTPUT_FORMATS=gguf_q4_k_m,gguf_q5_k_m,gguf_q8_0  # All go to gguf/ folder
```

## Output Paths

```bash
OUTPUT_DIR_BASE=./outputs                    # Base output directory
PREPROCESSED_DATA_DIR=./data/preprocessed    # Cached preprocessed data
CACHE_DIR=./cache                            # Model/dataset cache
```

Output structure:
```
outputs/Qwen3-4B/
├── lora/          # Always created by train.py
├── merged_16bit/  # Always created by train.py
└── gguf/          # Created by build.py if OUTPUT_FORMATS has gguf_*
```

## Logging

```bash
LOGGING_STEPS=10          # Log every N steps
SAVE_STEPS=500            # Save checkpoint every N steps
SAVE_TOTAL_LIMIT=3        # Keep only last N checkpoints
SAVE_ONLY_FINAL=false     # Set true to skip intermediate checkpoints (saves disk)
```

## Weights & Biases (Optional)

```bash
WANDB_ENABLED=false
WANDB_PROJECT=unsloth-finetuning
WANDB_RUN_NAME=auto  # Auto-generates from model name
```

Set `WANDB_ENABLED=true` to enable training monitoring with W&B.

## HuggingFace Hub (Optional)

```bash
PUSH_TO_HUB=false
HF_USERNAME=your_username
HF_MODEL_NAME=auto  # Auto-generates from model name
HF_TOKEN=           # Get from https://huggingface.co/settings/tokens
```

## Author Attribution

```bash
AUTHOR_NAME=Your Name
```

Your name for model cards and citations. This appears in:
- Generated README files in all output formats (lora/, merged_16bit/, gguf/)
- BibTeX citations for academic use
- Model metadata

The generated README will credit:
1. **You** (AUTHOR_NAME) - as the person who trained the model
2. **Training pipeline** - farhan-syah/unsloth-finetuning
3. **Technology** - Unsloth (2x faster fine-tuning)

Example generated credit section:
```markdown
## Credits

**Trained by:** John Doe

**Training pipeline:**
- [unsloth-finetuning](https://github.com/farhan-syah/unsloth-finetuning) by [@farhan-syah](https://github.com/farhan-syah)
- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster LLM fine-tuning
```

## Random Seed

```bash
SEED=3407  # For reproducibility
```

## Example Configurations

### Quick Test (1-2 minutes)

```bash
LORA_BASE_MODEL=unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit
INFERENCE_BASE_MODEL=
OUTPUT_MODEL_NAME=auto
DATASET_NAME=yahma/alpaca-cleaned
MAX_STEPS=50
DATASET_MAX_SAMPLES=100
LORA_RANK=16
LORA_ALPHA=32
CHECK_SEQ_LENGTH=false
OUTPUT_FORMATS=gguf_q4_k_m
```

### Full Training (Production)

```bash
LORA_BASE_MODEL=unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit
INFERENCE_BASE_MODEL=
OUTPUT_MODEL_NAME=auto
DATASET_NAME=yahma/alpaca-cleaned
MAX_STEPS=0
DATASET_MAX_SAMPLES=0
NUM_TRAIN_EPOCHS=1
LORA_RANK=64
LORA_ALPHA=128
CHECK_SEQ_LENGTH=false
OUTPUT_FORMATS=gguf_q4_k_m,gguf_q5_k_m,gguf_q8_0
```

### Memory-Constrained (6GB VRAM)

```bash
LORA_BASE_MODEL=unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit  # Lightweight 2B model
INFERENCE_BASE_MODEL=
OUTPUT_MODEL_NAME=auto
MAX_SEQ_LENGTH=1024              # Reduce from 2048
BATCH_SIZE=1                     # Reduce from 2
GRADIENT_ACCUMULATION_STEPS=8    # Increase to maintain effective batch
LORA_RANK=32                     # Reduce from 64
USE_GRADIENT_CHECKPOINTING=true
```

## Troubleshooting

### Out of Memory (OOM)

Try these in order:
1. Reduce `MAX_SEQ_LENGTH` (2048 → 1024)
2. Reduce `BATCH_SIZE` (2 → 1)
3. Increase `GRADIENT_ACCUMULATION_STEPS` (maintain effective batch size)
4. Enable `USE_GRADIENT_CHECKPOINTING=true`
5. Reduce `LORA_RANK` (64 → 32)

### Training Too Slow

- Set `CHECK_SEQ_LENGTH=false` (skip length checking)
- Increase `BATCH_SIZE` if VRAM allows
- Reduce `GRADIENT_ACCUMULATION_STEPS`
- Use `PACKING=true` (experimental, not for chat models)

### Length/Token Errors

```bash
CHECK_SEQ_LENGTH=true
FORCE_PREPROCESS=true
```

Run `python train.py` to filter out long samples.

## See Also

- [Training Guide](TRAINING.md) - Training tips and best practices
- [FAQ](FAQ.md) - Common questions
