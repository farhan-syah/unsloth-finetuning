# Configuration Guide

Complete reference for configuring the Unsloth fine-tuning pipeline.

## Overview

The pipeline uses **two configuration files**:

1. **training_params.yaml** - All training hyperparameters and model/dataset selection
   - Model selection (base_model, inference_model, output_name)
   - Dataset configuration (name, max_samples)
   - LoRA settings (rank, alpha, dropout)
   - Batch size and gradient accumulation
   - Learning rate and optimizer
   - Number of epochs and max steps
   - Sequence length and data processing
   - Logging and checkpoints
   - Output formats (GGUF quantizations)
   - Benchmark settings

2. **.env** - Credentials, paths, and operational flags
   - HuggingFace credentials (HF_TOKEN, HF_USERNAME)
   - Author attribution (AUTHOR_NAME)
   - Directory paths (OUTPUT_DIR_BASE, CACHE_DIR, PREPROCESSED_DATA_DIR)
   - W&B integration (WANDB_ENABLED, WANDB_PROJECT)
   - HuggingFace Hub (PUSH_TO_HUB, HF_MODEL_NAME)
   - Operational flags (FORCE_PREPROCESS, CHECK_SEQ_LENGTH, FORCE_REBUILD)
   - Ollama configuration (OLLAMA_BASE_URL)

**Why this split?**
- **YAML files are shareable** - Commit `training_params.yaml` to git, share with team
- **.env stays private** - Contains secrets (HF_TOKEN), never commit to git
- **Better organization** - Related parameters grouped hierarchically
- **Type safety** - YAML config validated with Pydantic (catches errors early)

## Quick Setup

```bash
# 1. Copy .env template (for credentials and paths)
cp .env.example .env
vim .env  # Edit HF_TOKEN, HF_USERNAME, AUTHOR_NAME

# 2. Configure model, dataset, and training (optional - defaults work for testing)
vim training_params.yaml  # Edit model, dataset, training parameters

# 3. Run training
python scripts/train.py  # Uses training_params.yaml (default)
python scripts/train.py --config quick_test.yaml  # For testing
python scripts/train.py --config my_custom_config.yaml  # Custom
```

---

## Training Configuration (YAML)

### File: training_params.yaml

Complete training hyperparameters reference.

### LoRA Configuration

```yaml
training:
  lora:
    rank: 64              # Number of rank dimensions
                          # Higher = more parameters = better quality (but slower, more VRAM)
                          # Typical values:
                          #   8  = Very fast, minimal VRAM, lower quality
                          #   16 = Fast, good for quick experiments
                          #   32 = Balanced for medium datasets
                          #   64 = Recommended for most use cases
                          #   128 = High quality, use with large datasets

    alpha: 128            # Scaling factor for LoRA updates
                          # Often set to 2x the rank value
                          # Affects magnitude of updates during training
                          # Higher alpha = stronger adaptation

    dropout: 0.0          # Regularization to prevent overfitting
                          # 0.0 = no dropout (recommended for most cases)
                          # 0.05-0.1 = light regularization if overfitting
                          # Note: Unsloth optimizes training when dropout=0

    use_rslora: false     # Rank-Stabilized LoRA
                          # Improves training stability for high ranks
                          # Recommended: false for rank <= 64, true for rank >= 128
```

**Examples:**
```yaml
# Quick experimentation (fast, lower quality)
lora:
  rank: 16
  alpha: 32

# Standard production (balanced)
lora:
  rank: 64
  alpha: 128

# High quality (slow, more VRAM)
lora:
  rank: 128
  alpha: 256
  use_rslora: true
```

---

### Batch Configuration

```yaml
training:
  batch:
    size: 4                         # Per-device batch size
                                     # Samples processed simultaneously
                                     # Higher = faster, but more VRAM
                                     # Reduce if "CUDA out of memory" errors
                                     # Typical values: 1, 2, 4, 8

    gradient_accumulation_steps: 2  # Simulate larger batches
                                     # Effective batch = size × this value
                                     # Example: 4 × 2 = 8 effective batch
                                     # Higher = more stable training, slower
```

**Effective Batch Size** = `size` × `gradient_accumulation_steps`

**VRAM vs Speed Trade-offs** (for effective batch = 16):

| Configuration | VRAM Usage | Speed | Notes |
|--------------|------------|-------|-------|
| size=16, accum=1 | Highest | Fastest | Ideal if VRAM allows |
| size=8, accum=2 | High | Fast | Good balance |
| size=4, accum=4 | Medium | Medium | Recommended default |
| size=2, accum=8 | Low | Slow | For limited VRAM |
| size=1, accum=16 | Lowest | Slowest | Last resort for OOM |

**Recommendations:**
- **Target effective batch size:** 8-16 (good for most use cases)
- **OOM errors?** Reduce `size`, increase `gradient_accumulation_steps`
- **Plenty of VRAM?** Increase `size` for faster training

---

### Optimization Configuration

```yaml
training:
  optimization:
    learning_rate: 3e-4   # Learning rate (how fast model learns)
                          # Too high = unstable training
                          # Too low = slow convergence
                          # Typical range: 1e-5 to 5e-4
                          # Recommended: 2e-4 to 3e-4 for LoRA

    optimizer: adamw_8bit # Optimizer algorithm
                          # Options:
                          #   adamw_8bit   - Memory-efficient (recommended)
                          #   adamw_torch  - Standard PyTorch (more VRAM)
                          #   sgd          - Simple, rarely used for LLMs
                          #   adafactor    - Alternative to AdamW

    warmup_ratio: 0.1     # Warmup as percentage of total steps
                          # Gradually increases LR from 0 to learning_rate
                          # Stabilizes training in early steps
                          # 0.1 = 10% of total steps (recommended)
                          # 0.05 = 5% (shorter warmup)
                          # 0.0 = no warmup (not recommended)

    warmup_steps: 0       # Fixed number of warmup steps
                          # If > 0, overrides warmup_ratio
                          # If 0, uses warmup_ratio instead
                          # Recommended: Keep at 0, use warmup_ratio

    max_grad_norm: 1.0    # Gradient clipping threshold
                          # Prevents exploding gradients
                          # 1.0 is standard, rarely needs changing

    use_gradient_checkpointing: true  # Trade compute for memory
                                       # true = Lower VRAM, ~20% slower
                                       # false = Faster, more VRAM
                                       # Recommended: true for most cases
```

**Examples:**
```yaml
# Conservative (stable training)
optimization:
  learning_rate: 2e-4
  warmup_ratio: 0.1

# Aggressive (faster convergence, may be unstable)
optimization:
  learning_rate: 5e-4
  warmup_ratio: 0.05

# Maximum speed (if VRAM allows)
optimization:
  use_gradient_checkpointing: false
```

---

### Training Duration

```yaml
training:
  epochs:
    num_train_epochs: 3   # Number of epochs (full passes through dataset)
                          # Used when max_steps = 0
                          # Typical values:
                          #   1 = One pass (large/synthetic datasets)
                          #   3 = Standard (most use cases)
                          #   5+ = High-quality curated datasets only

    max_steps: 0          # Maximum training steps
                          # 0 = train for num_train_epochs
                          # >0 = stop after N steps (ignores epochs)
                          # Use for precise control or quick testing

    dataset_max_samples: 0  # Limit dataset size
                            # 0 = use full dataset
                            # >0 = use only first N samples
                            # Useful for quick testing
```

**Which to use?**
- **Production:** `max_steps: 0, dataset_max_samples: 0` (full training)
- **Testing:** `max_steps: 50, dataset_max_samples: 100` (quick validation)
- **Experimentation:** Adjust based on dataset size and quality

**Overfitting Prevention:**
- **Small datasets (<1K):** Use 1 epoch
- **Medium datasets (1K-10K):** Use 1-3 epochs
- **Large datasets (>10K):** Use 1 epoch, monitor loss

---

### Data Processing

```yaml
training:
  data:
    max_seq_length: 2048  # Maximum sequence length in tokens
                          # Must not exceed model's context window
                          # Longer = more context, but more VRAM
                          # Common values: 512, 1024, 2048, 4096, 8192
                          # Recommended: Match your data's typical length

    packing: false        # Pack multiple short sequences into one
                          # true = Better GPU utilization, faster
                          # false = Simpler, changes loss less
                          # Recommended: false for most use cases

    seed: 3407            # Random seed for reproducibility
                          # Fixed number = identical results across runs
                          # Useful for debugging and comparing experiments
```

**max_seq_length Guidelines:**
- Check your model's maximum: Most are 2048-8192
- Analyze your dataset: Run `python scripts/preprocess.py`
- **Shorter is faster:** 2048 uses ~50% less VRAM than 4096
- **OOM errors?** Reduce this first

---

### Logging & Checkpoints

```yaml
logging:
  logging_steps: 5        # Log metrics every N steps
                          # Lower = more frequent logging
                          # Typical values: 1 (debugging), 5 (standard), 10+ (long runs)

  save_steps: 25          # Save checkpoint every N steps
                          # Only used if save_only_final = false
                          # Lower = more checkpoints, more disk space

  save_total_limit: 2     # Maximum checkpoints to keep
                          # Only used if save_only_final = false
                          # 0 = keep all checkpoints
                          # Higher values use more disk space

  save_only_final: true   # Only save final checkpoint
                          # true = Faster, saves disk space (recommended for testing)
                          # false = Save checkpoints during training (safer for long runs)
```

**Recommendations:**
- **Testing/prototyping:** `save_only_final: true`
- **Production/long runs:** `save_only_final: false, save_total_limit: 3`
- **Debugging:** `logging_steps: 1, save_only_final: false`

---

### Output Formats

```yaml
output:
  formats:                # List of formats to generate after training
    - gguf_f16            # 16-bit GGUF (largest, highest quality)
    - gguf_q8_0           # 8-bit quantization (excellent quality)
    - gguf_q6_k           # 6-bit quantization (good quality)
    - gguf_q4_k_m         # 4-bit quantization (balanced size/quality)
```

**Available Formats:**

| Format | Size | Quality | Use Case |
|--------|------|---------|----------|
| `gguf_f16` | Largest | Best | Reference, highest quality inference |
| `gguf_f32` | Even larger | Best | Rarely needed |
| `gguf_q8_0` | Large | Excellent | High-quality production |
| `gguf_q6_k` | Medium | Very Good | Good balance |
| `gguf_q5_k_m` | Medium | Good | Balanced |
| `gguf_q4_k_m` | Small | Good | **Recommended for most users** |
| `gguf_q3_k_m` | Very Small | Okay | Resource-constrained |
| `gguf_q2_k` | Smallest | Poor | Not recommended |
| `merged_16bit` | Largest | Best | HuggingFace safetensors |
| `merged_4bit` | Small | Good | 4-bit safetensors |

**Note:** Build time varies based on hardware and number of quantizations

---

### Benchmark Configuration

```yaml
benchmark:
  max_tokens: 512         # Maximum tokens to generate during benchmarking
                          # Lower = faster, prevents repetition loops
                          # Higher = allows longer responses

  batch_size: 8           # Batch size for benchmark evaluation
                          # Adjust based on GPU memory

  default_backend: ollama # Default backend for benchmarking
                          # Options: ollama, vllm, transformers

  default_tasks:          # Default benchmark tasks
    - ifeval              # Instruction-following
    - gsm8k               # Math reasoning
    - hellaswag           # Commonsense reasoning
```

**Available Benchmark Tasks:**
- `ifeval` - Instruction-following evaluation (recommended)
- `gsm8k` - Math reasoning with chain-of-thought
- `hellaswag` - Commonsense reasoning
- `mmlu` - Knowledge across 57 subjects
- `truthfulqa` - Truthfulness evaluation
- `arc` - Science questions

---

## Environment Configuration (.env)

**Note:** Model and dataset selection have moved to YAML files. The .env file now only contains credentials, paths, and operational flags.

---

### Directory Paths

```bash
# Base directory for all outputs
OUTPUT_DIR_BASE=./outputs

# Cache directory for preprocessed datasets
PREPROCESSED_DATA_DIR=./data/preprocessed

# HuggingFace cache directory
CACHE_DIR=./cache
```

**Default Structure:**
```
./outputs/{model}-{dataset}/
├── lora/           # LoRA adapters
├── merged_16bit/   # Merged model
├── gguf/           # GGUF quantizations
└── benchmarks/     # Benchmark results
```

---

### HuggingFace Hub Integration

```bash
# Enable/disable pushing to HuggingFace Hub
PUSH_TO_HUB=false

# Your HuggingFace username
HF_USERNAME=your-username

# Model name on HuggingFace (auto = use OUTPUT_MODEL_NAME)
HF_MODEL_NAME=auto

# HuggingFace access token
# Get from: https://huggingface.co/settings/tokens
# IMPORTANT: Keep this secret! Never commit .env to git
HF_TOKEN=your-token-here
```

**To push models after training:**
1. Set `PUSH_TO_HUB=true`
2. Set your `HF_USERNAME` and `HF_TOKEN`
3. Run: `python scripts/push.py`

---

### Weights & Biases Integration

```bash
# Enable/disable W&B tracking
WANDB_ENABLED=false

# W&B project name
WANDB_PROJECT=unsloth-finetuning

# W&B run name (auto = generate from model name + lora rank)
WANDB_RUN_NAME=auto
```

**To enable W&B tracking:**
1. Sign up at https://wandb.ai
2. Run `wandb login`
3. Set `WANDB_ENABLED=true` in `.env`

---

### Operational Flags

```bash
# Check sequence lengths during preprocessing
CHECK_SEQ_LENGTH=true

# Force preprocessing even if cached data exists
FORCE_PREPROCESS=false

# Force rebuilding merged/GGUF models even if they exist
# Note: Training now uses automatic backup system - no force flag needed
FORCE_REBUILD=false
```

**When to use these flags:**
- `CHECK_SEQ_LENGTH=true` - First run with new dataset (analyzes lengths)
- `CHECK_SEQ_LENGTH=false` - Subsequent runs (faster preprocessing)
- `FORCE_PREPROCESS=true` - After changing max_seq_length or dataset
- `FORCE_REBUILD=true` - After changing output formats

---

### Ollama Configuration

```bash
# Ollama server URL for benchmarking with GGUF models
OLLAMA_BASE_URL=http://localhost:11434
```

---

### Author Attribution

```bash
# Your name for model card and citations
AUTHOR_NAME=Your Name
```

Used in generated README files and model cards to credit you as the fine-tuner.

---

## Example Configurations

### Quick Test (1-2 minutes)

**File: quick_test.yaml** (included)
```yaml
training:
  lora:
    rank: 32
    alpha: 64

  batch:
    size: 2
    gradient_accumulation_steps: 1

  epochs:
    num_train_epochs: 1
    dataset_max_samples: 100  # Only 100 samples

  data:
    max_seq_length: 2048

logging:
  save_only_final: true

output:
  formats:
    - gguf_f16  # Only F16 for quick validation
```

**Run with:**
```bash
python scripts/train.py --config quick_test.yaml
```

---

### Production Training (Hours)

**File: training_params.yaml** (default)
```yaml
training:
  lora:
    rank: 64
    alpha: 128

  batch:
    size: 4
    gradient_accumulation_steps: 2

  optimization:
    learning_rate: 3e-4
    optimizer: adamw_8bit
    use_gradient_checkpointing: true

  epochs:
    num_train_epochs: 3
    dataset_max_samples: 0  # All samples

  data:
    max_seq_length: 2048
    packing: false

logging:
  save_only_final: false  # Save checkpoints
  save_total_limit: 3

output:
  formats:
    - gguf_f16
    - gguf_q8_0
    - gguf_q4_k_m
```

**Run with:**
```bash
python scripts/train.py  # Uses training_params.yaml by default
```

---

### High-Quality Training (Large datasets)

**File: high_quality.yaml** (custom)
```yaml
training:
  lora:
    rank: 128
    alpha: 256
    use_rslora: true

  batch:
    size: 8  # If VRAM allows
    gradient_accumulation_steps: 2

  optimization:
    learning_rate: 2e-4  # More conservative
    warmup_ratio: 0.1
    use_gradient_checkpointing: true

  epochs:
    num_train_epochs: 1  # One pass on large dataset
    dataset_max_samples: 0

  data:
    max_seq_length: 4096  # Longer context

logging:
  logging_steps: 10
  save_only_final: false
  save_total_limit: 5

output:
  formats:
    - gguf_f16
    - gguf_q8_0
    - gguf_q6_k
    - gguf_q4_k_m
```

---

### Low-VRAM Training (4GB GPU)

**File: low_vram.yaml** (custom)
```yaml
training:
  lora:
    rank: 16  # Smaller rank
    alpha: 32

  batch:
    size: 1   # Minimum batch
    gradient_accumulation_steps: 8  # Maintain effective batch

  optimization:
    learning_rate: 3e-4
    use_gradient_checkpointing: true  # Essential for low VRAM

  epochs:
    num_train_epochs: 3
    dataset_max_samples: 0

  data:
    max_seq_length: 1024  # Shorter sequences

logging:
  save_only_final: true

output:
  formats:
    - gguf_q4_k_m  # Only one format to save disk
```

---

## Migrating from Old .env Configuration

If you have an old `.env` file with training parameters, here's how to migrate:

### Parameter Mapping

| Old (.env) | New (YAML) | Notes |
|------------|------------|-------|
| `LORA_RANK` | `training.lora.rank` | Moved to YAML |
| `LORA_ALPHA` | `training.lora.alpha` | Moved to YAML |
| `LORA_DROPOUT` | `training.lora.dropout` | Moved to YAML |
| `USE_RSLORA` | `training.lora.use_rslora` | Moved to YAML |
| `BATCH_SIZE` | `training.batch.size` | Moved to YAML |
| `GRADIENT_ACCUMULATION_STEPS` | `training.batch.gradient_accumulation_steps` | Moved to YAML |
| `LEARNING_RATE` | `training.optimization.learning_rate` | Moved to YAML |
| `WARMUP_RATIO` | `training.optimization.warmup_ratio` | Moved to YAML |
| `WARMUP_STEPS` | `training.optimization.warmup_steps` | Moved to YAML |
| `MAX_GRAD_NORM` | `training.optimization.max_grad_norm` | Moved to YAML |
| `OPTIM` | `training.optimization.optimizer` | Moved to YAML |
| `USE_GRADIENT_CHECKPOINTING` | `training.optimization.use_gradient_checkpointing` | Moved to YAML |
| `NUM_TRAIN_EPOCHS` | `training.epochs.num_train_epochs` | Moved to YAML |
| `MAX_STEPS` | `training.epochs.max_steps` | Moved to YAML |
| `DATASET_MAX_SAMPLES` | `training.epochs.dataset_max_samples` | Moved to YAML |
| `MAX_SEQ_LENGTH` | `training.data.max_seq_length` | Moved to YAML |
| `PACKING` | `training.data.packing` | Moved to YAML |
| `SEED` | `training.data.seed` | Moved to YAML |
| `LOGGING_STEPS` | `logging.logging_steps` | Moved to YAML |
| `SAVE_STEPS` | `logging.save_steps` | Moved to YAML |
| `SAVE_TOTAL_LIMIT` | `logging.save_total_limit` | Moved to YAML |
| `SAVE_ONLY_FINAL` | `logging.save_only_final` | Moved to YAML |
| `OUTPUT_FORMATS` | `output.formats` | Moved to YAML (now a list) |
| `LORA_BASE_MODEL` | `model.base_model` | **Moved to YAML** |
| `INFERENCE_BASE_MODEL` | `model.inference_model` | **Moved to YAML** |
| `OUTPUT_MODEL_NAME` | `model.output_name` | **Moved to YAML** |
| `DATASET_NAME` | `dataset.name` | **Moved to YAML** |
| `DATASET_MAX_SAMPLES` | `dataset.max_samples` | **Moved to YAML** (was also in training.epochs) |
| `HF_TOKEN` | Stays in `.env` | Credentials |
| `HF_USERNAME` | Stays in `.env` | Credentials |
| `AUTHOR_NAME` | Stays in `.env` | Attribution |
| `OUTPUT_DIR_BASE` | Stays in `.env` | Paths |
| `CHECK_SEQ_LENGTH` | Stays in `.env` | Operational flag |
| `FORCE_PREPROCESS` | Stays in `.env` | Operational flag |
| `FORCE_REBUILD` | Stays in `.env` | Operational flag |

### Migration Example

**OLD (.env):**
```bash
# Model and dataset (now in YAML)
LORA_BASE_MODEL=unsloth/Llama-3.2-1B-Instruct-bnb-4bit
DATASET_NAME=GAIR/lima

# Training parameters (now in YAML)
LORA_RANK=64
LORA_ALPHA=128
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=3e-4
NUM_TRAIN_EPOCHS=3
MAX_STEPS=0
MAX_SEQ_LENGTH=2048
OUTPUT_FORMATS=gguf_f16,gguf_q4_k_m
```

**NEW (training_params.yaml):**
```yaml
# Model and dataset selection
model:
  base_model: unsloth/Llama-3.2-1B-Instruct-bnb-4bit
  inference_model: unsloth/Llama-3.2-1B-Instruct
  output_name: auto

dataset:
  name: GAIR/lima
  max_samples: 0

# Training parameters
training:
  lora:
    rank: 64
    alpha: 128

  batch:
    size: 4
    gradient_accumulation_steps: 2

  optimization:
    learning_rate: 3e-4

  epochs:
    num_train_epochs: 3
    max_steps: 0

  data:
    max_seq_length: 2048

output:
  formats:
    - gguf_f16
    - gguf_q4_k_m
```

---

## Tips & Best Practices

### Configuration Management

1. **Version control YAML, not .env**
   - Commit `training_params.yaml` to git
   - Add `.env` to `.gitignore`
   - Share YAML configs with team

2. **Create profiles for different scenarios**
   - `training_params.yaml` - Production
   - `quick_test.yaml` - Fast validation
   - `experiment_01.yaml` - Custom experiments

3. **Document your configurations**
   - Add comments explaining why you chose values
   - Note the dataset size and characteristics
   - Record benchmark results in comments

### Tuning Strategy

1. **Start with quick_test.yaml**
   - Validate setup works
   - Test preprocessing pipeline
   - Estimate training time

2. **Use preprocessing recommendations**
   - Run `python scripts/preprocess.py`
   - Get smart recommendations for your dataset
   - Adjust training_params.yaml accordingly

3. **Monitor training loss**
   - Watch `loss_history.csv` in `lora/` folder
   - Stop early if loss plateaus
   - Adjust learning rate if unstable

4. **Iterate on hyperparameters**
   - Start with rank=64, alpha=128
   - Increase rank if underfitting
   - Reduce epochs if overfitting

---

## Troubleshooting

### Configuration Errors

**Error:** `Invalid configuration in training_params.yaml`

**Solution:** The YAML config is validated with Pydantic. Check the error message for specific field issues:
- Ensure rank is positive: `rank: 64` not `rank: -5`
- Ensure dropout is 0-1: `dropout: 0.1` not `dropout: 2.0`
- Check optimizer is valid: `adamw_8bit`, `adamw_torch`, `sgd`, or `adafactor`

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'config_schema'`

**Solution:** Make sure you're running scripts from the project root:
```bash
python scripts/train.py  # Correct
cd scripts && python train.py  # Incorrect
```

### Missing Config File

**Error:** `Configuration file not found: training_params.yaml`

**Solution:** Create the file or specify a different config:
```bash
cp quick_test.yaml my_config.yaml
python scripts/train.py --config my_config.yaml
```

---

## See Also

- [Training Guide](TRAINING.md) - Detailed training walkthrough
- [Understanding Fine-tuning](UNDERSTANDING_FINETUNING.md) - When and why to fine-tune
- [FAQ](FAQ.md) - Frequently asked questions
