# Training Guide

Tips, best practices, and troubleshooting for fine-tuning with Unsloth.

## Quick Reference

```bash
# Test training (2 minutes) - uses quick_test.yaml
cp .env.example .env
python scripts/train.py --config quick_test.yaml

# Full training - uses training_params.yaml
cp .env.example .env
# Edit training_params.yaml to customize model, dataset, or parameters
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

1. Checks for existing LoRA adapters
   - If found: Automatically backs up to `lora_bak/{timestamp}_rank{X}_lr{Y}_loss{Z}/`
   - Backup naming includes training parameters for easy identification
2. Loads model with 4-bit quantization
3. Adds LoRA adapters
4. Trains for specified epochs/steps
5. Saves LoRA adapters with training metrics
6. Merges and saves full 16-bit model

**Retraining:** You can safely retrain with different parameters - previous versions are automatically preserved. Use `python scripts/restore_trained_data.py` to restore any previous training.

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

Use the quick test configuration for initial validation:
```bash
cp .env.example .env
python scripts/train.py --config quick_test.yaml  # Quick test with 50 steps
```

Verify:
- Training completes without errors
- VRAM usage is acceptable
- Loss decreases during training

### 2. Monitor VRAM Usage

The training summary shows max VRAM used. If you're close to limit, edit `training_params.yaml`:
- Reduce `training.data.max_seq_length` (from 2048 to 1024)
- Reduce `training.batch.size` (from 4 to 2 or 1)
- Reduce `training.lora.rank` (from 64 to 32 or 16)

### 3. Choose Appropriate LoRA Rank

Configure in `training_params.yaml`:

| LoRA Rank | Adapter Size | Quality | Use Case |
|-----------|--------------|---------|----------|
| 8-16 | Small (~70MB) | Basic | Testing, simple tasks |
| 32-64 | Medium (~140MB) | Good | Most use cases |
| 128+ | Large (~280MB+) | Best | Complex tasks, large datasets |

### 4. Effective Batch Size Matters

Configure in `training_params.yaml`:
```yaml
training:
  batch:
    size: 4                         # Per-device batch size
    gradient_accumulation_steps: 2  # Gradient accumulation
```

`Effective Batch Size = size × gradient_accumulation_steps`

Examples:
- `2 × 4 = 8` (good balance)
- `1 × 8 = 8` (same effective, less VRAM)
- `4 × 2 = 8` (same effective, faster if VRAM allows)

Target: 4-16 effective batch size

### 5. Sequence Length Checking

**Default: `CHECK_SEQ_LENGTH=false`** (fast) - Set in `.env`
- Trainer handles truncation automatically
- Good for most datasets

**Set `CHECK_SEQ_LENGTH=true`** when:
- First time with new dataset
- Want to see length statistics
- Dataset has extremely long samples

## Training on Different Hardware

Edit `training_params.yaml` based on your GPU:

### NVIDIA GPU (12GB)

```yaml
model:
  base_model: unsloth/Llama-3.2-3B-Instruct-bnb-4bit

training:
  lora:
    rank: 64
  batch:
    size: 2
    gradient_accumulation_steps: 4
  data:
    max_seq_length: 2048
```

### NVIDIA GPU (8GB)

```yaml
model:
  base_model: unsloth/Llama-3.2-1B-Instruct-bnb-4bit

training:
  lora:
    rank: 32               # Reduced
  batch:
    size: 1                # Reduced
    gradient_accumulation_steps: 8
  optimization:
    use_gradient_checkpointing: true
  data:
    max_seq_length: 1024   # Reduced
```

### NVIDIA GPU (24GB+)

```yaml
model:
  base_model: unsloth/Llama-3.1-8B-Instruct-bnb-4bit

training:
  lora:
    rank: 128              # Increased
  batch:
    size: 4                # Increased
    gradient_accumulation_steps: 2
  data:
    max_seq_length: 4096   # Increased
```

### AMD GPU (ROCm)

Same as NVIDIA, but ensure ROCm-compatible PyTorch is installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Apple Silicon (M1/M2/M3)

```yaml
model:
  base_model: unsloth/Llama-3.2-1B-Instruct-bnb-4bit

training:
  lora:
    rank: 32
  batch:
    size: 1                # MPS has lower memory
    gradient_accumulation_steps: 8
  data:
    max_seq_length: 2048
```

## Common Issues

### Out of Memory (OOM)

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions (try in order) - Edit `training_params.yaml`:**
1. Reduce `training.data.max_seq_length` (2048 → 1024)
2. Reduce `training.batch.size` (4 → 2 or 1)
3. Increase `training.batch.gradient_accumulation_steps` (maintain effective batch)
4. Enable `training.optimization.use_gradient_checkpointing: true`
5. Reduce `training.lora.rank` (64 → 32)

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

Key parameters to tune in `training_params.yaml`:
1. **Learning rate** - `training.optimization.learning_rate` (2e-4, 5e-5, 1e-4)
2. **LoRA rank** - `training.lora.rank` (32, 64, 128)
3. **Batch size** - `training.batch.size` and `gradient_accumulation_steps` (effective: 4, 8, 16)
4. **Training epochs** - `training.epochs.num_train_epochs`

Use W&B for systematic tracking (configure in `.env`):
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
- [Benchmarking Guide](BENCHMARK.md) - Optional validation and testing
