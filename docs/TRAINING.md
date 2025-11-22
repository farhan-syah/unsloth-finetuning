# Training Guide

Tips, best practices, and troubleshooting for fine-tuning with Unsloth.

## Quick Reference

```bash
# Test training (2 minutes)
cp .env.example .env
# The example is already configured for quick testing
python scripts/train.py

# Full training
cp .env.example .env
# Edit .env (set MAX_STEPS=0, DATASET_MAX_SAMPLES=0)
python scripts/train.py
```

## Understanding the Workflow

### Step 1: Preprocessing

The first time you run `train.py`:
1. Downloads dataset from HuggingFace
2. Applies chat template
3. Filters invalid samples
4. Optionally checks sequence lengths
5. Caches preprocessed data to disk

**Subsequent runs:** Uses cached data (unless `FORCE_PREPROCESS=true`)

### Step 2: Training

1. Loads model with 4-bit quantization
2. Adds LoRA adapters
3. Trains for specified epochs/steps
4. Saves LoRA adapters
5. Merges and saves full 16-bit model

### Step 3: Building (Optional)

Run `python scripts/build.py` to convert merged_16bit to other formats.

## Training Output

After training completes, you'll see:

```
============================================================
✅ TRAINING COMPLETE
============================================================

📊 Training Summary:

🔧 Training Configuration:
  Model: Qwen3-4B
  Dataset: yahma/alpaca-cleaned
  Samples Trained: 100
  Training Mode: 50 steps (early stop)
  Total Steps: 50
  Batch Size: 2 × 2 = 4 (effective)
  Max Seq Length: 4096
  LoRA: r=16, alpha=32

⏱️  Performance:
  Training Time: 2.7 minutes (164 seconds)
  Final Loss: 0.7647
  Max VRAM Used: 11.23 GB

💾 Output:
  LoRA Adapters: 141.2 MB
  Location: ./outputs/Qwen3-4B/lora
  Merged 16-bit: 7.62 GB (actual size shown by build.py)
  Location: ./outputs/Qwen3-4B/merged_16bit

💡 Next Steps:
  Run: python scripts/build.py
  Will create: gguf_q4_k_m,gguf_q5_k_m
============================================================
```

## Best Practices

### 1. Always Test First

The `.env.example` is pre-configured for quick validation:
```bash
cp .env.example .env
python scripts/train.py  # ~2 minutes
```

Verify:
- Training completes without errors
- VRAM usage is acceptable
- Loss decreases during training

### 2. Monitor VRAM Usage

The training summary shows max VRAM used. If you're close to limit:
- Reduce `MAX_SEQ_LENGTH`
- Reduce `BATCH_SIZE`
- Reduce `LORA_RANK`

### 3. Choose Appropriate LoRA Rank

| LoRA Rank | Adapter Size | Quality | Use Case |
|-----------|--------------|---------|----------|
| 8-16 | Small (~70MB) | Basic | Testing, simple tasks |
| 32-64 | Medium (~140MB) | Good | Most use cases |
| 128+ | Large (~280MB+) | Best | Complex tasks, large datasets |

### 4. Effective Batch Size Matters

`Effective Batch Size = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS`

Examples:
- `2 × 4 = 8` (good balance)
- `1 × 8 = 8` (same effective, less VRAM)
- `4 × 2 = 8` (same effective, faster if VRAM allows)

Target: 4-16 effective batch size

### 5. Sequence Length Checking

**Default: `CHECK_SEQ_LENGTH=false`** (fast)
- Trainer handles truncation automatically
- Good for most datasets

**Set `CHECK_SEQ_LENGTH=true`** when:
- First time with new dataset
- Want to see length statistics
- Dataset has extremely long samples

## Training on Different Hardware

### NVIDIA GPU (12GB)

```bash
LORA_BASE_MODEL=unsloth/Qwen3-1.7B-unsloth-bnb-4bit
MAX_SEQ_LENGTH=4096
BATCH_SIZE=2
LORA_RANK=64
```

### NVIDIA GPU (8GB)

```bash
LORA_BASE_MODEL=unsloth/Qwen3-1.7B-unsloth-bnb-4bit
MAX_SEQ_LENGTH=2048        # Reduce from 4096
BATCH_SIZE=1               # Reduce
GRADIENT_ACCUMULATION_STEPS=8
LORA_RANK=32               # Reduce
USE_GRADIENT_CHECKPOINTING=true
```

### NVIDIA GPU (24GB+)

```bash
LORA_BASE_MODEL=unsloth/Qwen3-8B-unsloth-bnb-4bit
MAX_SEQ_LENGTH=4096        # Increase
BATCH_SIZE=4               # Increase
LORA_RANK=128              # Increase
```

### AMD GPU (ROCm)

Same as NVIDIA, but ensure ROCm-compatible PyTorch is installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Apple Silicon (M1/M2/M3)

```bash
LORA_BASE_MODEL=unsloth/Qwen3-1.7B-unsloth-bnb-4bit
MAX_SEQ_LENGTH=4096
BATCH_SIZE=1               # MPS has lower memory
LORA_RANK=32
```

## Common Issues

### Out of Memory (OOM)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions (try in order):**
1. Reduce `MAX_SEQ_LENGTH` (4096 → 1024 or 2048)
2. Reduce `BATCH_SIZE` (2 → 1)
3. Increase `GRADIENT_ACCUMULATION_STEPS` (maintain effective batch)
4. Enable `USE_GRADIENT_CHECKPOINTING=true`
5. Reduce `LORA_RANK` (64 → 32)

### Sequence Length Errors

**Error:**
```
RuntimeError: sequence length exceeds maximum
```

**Solution:**
```bash
CHECK_SEQ_LENGTH=true
FORCE_PREPROCESS=true
```

Run `python scripts/train.py` - it will filter out samples exceeding `MAX_SEQ_LENGTH`.

Or increase `MAX_SEQ_LENGTH` (requires more VRAM).

### Training is Slow

**Preprocessing slow:**
- Set `CHECK_SEQ_LENGTH=false` (skip tokenization step)
- Preprocessed data is cached after first run

**Training slow:**
- Increase `BATCH_SIZE` if VRAM allows
- Reduce `GRADIENT_ACCUMULATION_STEPS`
- Use `PACKING=true` (experimental, not for chat)

### Loss Not Decreasing

**Possible causes:**
1. **Learning rate too high/low** - Try `2e-4` or `5e-5`
2. **Too few steps** - Increase `NUM_TRAIN_EPOCHS` or `MAX_STEPS`
3. **Bad dataset** - Check dataset quality
4. **LoRA rank too low** - Increase to 64 or 128

### Model Output is Gibberish

After training, test the merged model:
```bash
cd outputs/Qwen3-4B/merged_16bit
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('.', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('.')
messages = [{'role': 'user', 'content': 'Hello!'}]
inputs = tokenizer.apply_chat_template(messages, return_tensors='pt').to('cuda')
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
"
```

If output is good but GGUF is gibberish, it's a conversion issue (not training).

## Advanced Tips

### Resume Training

Training can be resumed from checkpoints:
```bash
# Checkpoints saved every SAVE_STEPS
outputs/Qwen3-4B/checkpoint-500/
outputs/Qwen3-4B/checkpoint-1000/
```

To resume, modify train.py to load from checkpoint (future feature).

### Multi-GPU Training

Currently single-GPU only. Multi-GPU support planned.

### Custom Datasets

Your dataset should have conversation format:
```json
{
  "conversations": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

Or adapt train.py's `convert_to_text()` function for your format.

### Hyperparameter Tuning

Key parameters to tune:
1. **Learning rate** (2e-4, 5e-5, 1e-4)
2. **LoRA rank** (32, 64, 128)
3. **Batch size** (effective: 4, 8, 16)
4. **Training steps/epochs**

Use W&B for systematic tracking:
```bash
WANDB_ENABLED=true
WANDB_PROJECT=my-finetuning
```

## Performance Tips

### Maximize Throughput

1. **Increase batch size** as much as VRAM allows
2. **Reduce gradient accumulation** if batch size is high
3. **Disable gradient checkpointing** if VRAM allows (faster)
4. **Use shorter sequences** if your data allows

### Minimize VRAM Usage

1. **Enable gradient checkpointing**
2. **Reduce batch size, increase accumulation**
3. **Use shorter max sequence length**
4. **Reduce LoRA rank**

### Balance Quality vs Speed

| Priority | Settings |
|----------|----------|
| **Speed** | Small batch, low rank, short sequences |
| **Quality** | Large effective batch, high rank, full sequences |
| **VRAM** | Gradient checkpointing, small batch, gradient accumulation |

## Monitoring Training

### Loss Curve

Good training:
- Loss decreases steadily
- No sudden spikes
- Plateaus at low value

Bad training:
- Loss increases or oscillates
- Very slow decrease
- Stays at high value

### VRAM Usage

Monitor during training:
```bash
watch -n 1 nvidia-smi
```

### Training Logs

Saved to: `outputs/Qwen3-4B/lora/training_metrics.json`

Contains:
- Training time
- Final loss
- VRAM usage
- All hyperparameters

## See Also

- [Configuration Guide](CONFIGURATION.md) - All `.env` options
- [FAQ](FAQ.md) - Common questions
- [Distribution Guide](DISTRIBUTION.md) - Sharing your models
