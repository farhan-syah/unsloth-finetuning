# Frequently Asked Questions

## General

### Q: What is Unsloth?

Unsloth is a library that makes LLM fine-tuning 2x faster and use 60% less VRAM through optimized implementations of attention mechanisms and training loops.

### Q: What models are supported?

Any model supported by Unsloth, including:
- Qwen/Qwen2/Qwen3 series
- Llama 2/3 series
- Mistral series
- Gemma series

Check [Unsloth's supported models](https://github.com/unslothai/unsloth#-supported-models).

### Q: Do I need a GPU?

Yes, GPU is required for training. Supported:
- NVIDIA (CUDA)
- AMD (ROCm)
- Apple Silicon (MPS)

Minimum: 8GB VRAM for 4B models

## Installation

### Q: Setup.sh fails - what do I do?

Try manual installation:
```bash
# Install dependencies
bash install_dependencies.sh

# Install llama.cpp (if you need GGUF)
bash install_llama_cpp.sh

# Configure
cp .env.example .env
```

### Q: Do I need llama.cpp?

Only if you want GGUF format for Ollama/llama.cpp. For just LoRA training, you don't need it.

### Q: Can I use CPU only?

No, Unsloth requires GPU. For CPU inference of trained models, use llama.cpp with GGUF format.

## Training

### Q: How long does training take?

Depends on:
- Dataset size
- Model size
- GPU speed
- Hyperparameters

Examples (Qwen3-4B on RTX 3060):
- Test (100 samples, 50 steps): ~2 minutes
- Small dataset (10K samples): ~30-60 minutes
- Large dataset (100K+ samples): Several hours

### Q: How much VRAM do I need?

| Model | Minimum VRAM | Recommended |
|-------|--------------|-------------|
| 2B | 6GB | 8GB |
| 4B | 8GB | 12GB |
| 8B | 16GB | 24GB |
| 14B+ | 24GB | 40GB+ |

Tips to reduce VRAM:
- Reduce `MAX_SEQ_LENGTH`
- Reduce `BATCH_SIZE`
- Reduce `LORA_RANK`
- Enable `USE_GRADIENT_CHECKPOINTING`

### Q: Training failed with OOM error

See [Training Guide - Out of Memory](TRAINING.md#out-of-memory-oom).

Quick fix - Edit `training_params.yaml`:
```yaml
training:
  lora:
    rank: 32                       # Reduce from 64
  batch:
    size: 1                        # Reduce from 4
    gradient_accumulation_steps: 8 # Maintain effective batch
  optimization:
    use_gradient_checkpointing: true
  data:
    max_seq_length: 1024           # Reduce from 2048
```

### Q: My model outputs gibberish after training

**If merged model works but GGUF doesn't:**
- This was a known issue with Unsloth's GGUF conversion
- We now use llama.cpp directly, which works correctly

**If merged model also outputs gibberish:**
- Training failed - check loss curve
- Dataset might be corrupted
- Learning rate too high
- Try retraining with different hyperparameters

### Q: Can I resume interrupted training?

Currently not directly supported. Future feature.

Workaround: Checkpoints are saved every `SAVE_STEPS` - you can manually load from checkpoint (requires code modification).

### Q: How do I know if training is working?

Check the loss in training output:
```
Step 10: Loss: 2.345
Step 20: Loss: 1.876  ← Should decrease
Step 30: Loss: 1.543
```

Good training: Loss decreases steadily
Bad training: Loss stays flat or increases

## Configuration

### Q: What's the difference between max_steps and num_train_epochs?

Configure in `training_params.yaml`:
```yaml
training:
  epochs:
    max_steps: 0              # 0 = use num_train_epochs
    num_train_epochs: 3       # Number of epochs
```

- `max_steps: 0`: Train for `num_train_epochs` epochs
- `max_steps: 50`: Train for exactly 50 steps (ignores epochs)

Use `max_steps` for quick testing, `num_train_epochs` for full training.

### Q: What LoRA rank should I use?

Configure in `training_params.yaml`:

| Use Case | LoRA Rank (training.lora.rank) |
|----------|--------------------------------|
| Testing | 16-32 |
| General | 64 |
| Complex tasks | 128+ |

Higher rank = better quality but larger adapters and more VRAM.

### Q: Should I enable CHECK_SEQ_LENGTH?

**Default: `false`** (faster)
- Trainer truncates long samples automatically
- Good for most cases

**Set to `true`** when:
- First time with new dataset
- Want to see statistics
- Training fails with length errors

### Q: What output formats should I use?

Configure in `training_params.yaml`:

```yaml
# Just training, no conversion
output:
  formats: []

# For Ollama (most common)
output:
  formats:
    - gguf_q4_k_m

# Multiple GGUF quantizations
output:
  formats:
    - gguf_q4_k_m
    - gguf_q5_k_m
    - gguf_q8_0

# High quality formats
output:
  formats:
    - gguf_f16
    - gguf_q8_0
    - gguf_q4_k_m
```

## Output & Formats

### Q: What's the difference between lora, merged_16bit, and gguf?

| Format | Size | Purpose |
|--------|------|---------|
| `lora/` | ~140MB | Adapters only, needs base model |
| `merged_16bit/` | Varies by model | Full model, safetensors (1.7B ~3.4GB, 4B ~7.6GB) |
| `gguf/` | Varies by quant | Quantized for Ollama/llama.cpp |

**Important:** `merged_16bit/` is saved in 16-bit precision format, but since we train on a 4-bit quantized base (e.g., `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`), the model quality reflects 4-bit-trained weights. This is standard practice and produces excellent results while keeping VRAM requirements low during training.

### Q: Which GGUF quantization should I use?

| Quant | Size (4B model) | Quality | Use Case |
|-------|-----------------|---------|----------|
| Q4_K_M | ~2.4GB | Good | **Recommended** - best balance |
| Q5_K_M | ~3.0GB | Better | More quality, acceptable size |
| Q8_0 | ~4.0GB | Best | Maximum quality |
| Q2_K | ~1.5GB | Lower | Smallest size |

### Q: Can I delete intermediate files?

**Keep:**
- `lora/` - Small, useful for continued training
- `merged_16bit/` - Needed for building other formats

**Can delete:**
- `gguf/` - If you don't use Ollama
- `merged_4bit/` - If you don't need 4-bit safetensors
- `checkpoint-*/` - Intermediate checkpoints (auto-deleted)

### Q: How do I use the trained model?

**With Python (transformers):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("outputs/Qwen3-4B/merged_16bit")
tokenizer = AutoTokenizer.from_pretrained("outputs/Qwen3-4B/merged_16bit")
```

**With Ollama:**
```bash
cd outputs/Qwen3-4B/gguf
cat > Modelfile <<EOF
FROM ./Qwen3-4B-Q4_K_M.gguf
EOF
ollama create my-model -f Modelfile
ollama run my-model "Hello!"
```

**With llama.cpp:**
```bash
llama-cli -m outputs/Qwen3-4B/gguf/Qwen3-4B-Q4_K_M.gguf -p "Hello"
```

## Datasets

### Q: What dataset format is required?

Supports datasets with `conversations` field:
```json
{
  "conversations": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
  ]
}
```

Other formats need modification to `convert_to_text()` in train.py.

### Q: Can I use my own dataset?

Yes! Either:
1. Upload to HuggingFace and set `DATASET_NAME=your-username/your-dataset`
2. Modify train.py to load from local files

### Q: How do I combine multiple datasets?

Currently not supported directly. Workarounds:
1. Combine datasets manually and upload to HuggingFace
2. Modify train.py to load multiple datasets

## Performance

### Q: Why is preprocessing slow?

First run:
- Downloads dataset
- Applies chat template
- Tokenizes all samples (if `CHECK_SEQ_LENGTH=true`)

Subsequent runs: Uses cached data (much faster)

Speed up:
```bash
CHECK_SEQ_LENGTH=false  # Skip tokenization
```

### Q: Training is slower than expected

Check in `training_params.yaml`:
- `training.optimization.use_gradient_checkpointing: true` - Saves VRAM but slower
- `training.batch.size: 1` - Increase if VRAM allows
- `dataset.max_samples` - Large datasets take time
- GPU utilization - Run `nvidia-smi` to check

### Q: GGUF conversion is slow

Normal - quantization is CPU-intensive. Q4_K_M takes ~5-10 minutes for 4B model.

Multiple quantizations run sequentially, so `gguf_q4_k_m,gguf_q5_k_m,gguf_q8_0` takes 15-30 minutes.

## Distribution

### Q: How do I share my model?

Use the push script to upload to HuggingFace:

```bash
HF_USERNAME=your-username
HF_TOKEN=your-token
PUSH_TO_HUB=true
```

### Q: What should I upload to HuggingFace?

Minimum:
- `lora/` - Small, others can merge themselves
- `merged_16bit/` - Full model

Recommended:
- `lora/`
- `merged_16bit/`
- `gguf/` folder with Q4_K_M and Q5_K_M

### Q: How do I create a model card?

README is auto-generated in each output folder with model card information. Edit as needed before uploading.

## Troubleshooting

### Q: ImportError: No module named 'unsloth'

```bash
pip install unsloth
# Or run setup.sh
```

### Q: CUDA out of memory

See [Training Guide](TRAINING.md#out-of-memory-oom).

### Q: Flash Attention version warning

Can be safely ignored. Unsloth works with Flash Attention 2.8.3+.

### Q: llama.cpp not found

Only needed for GGUF. Install:
```bash
bash install_llama_cpp.sh
```

Or use system package manager.

### Q: Model not downloading

Check:
- Internet connection
- HuggingFace status
- Disk space

Try manual download:
```bash
huggingface-cli download unsloth/Qwen3-4B-unsloth-bnb-4bit
```

## Advanced

### Q: Can I use LoRA adapters with the base model?

Yes:
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-4B-unsloth-bnb-4bit")
model = PeftModel.from_pretrained(base, "outputs/Qwen3-4B/lora")
```

### Q: Can I retrain with different parameters?

Yes! Just run `train.py` again. The system automatically backs up your existing LoRA adapters to `lora_bak/{timestamp}_rank{X}_lr{Y}_loss{Z}/` before training with new parameters.

To restore a previous version, use: `python scripts/restore_trained_data.py`

### Q: How do I merge multiple LoRA adapters?

Not currently supported. Feature request welcome!

### Q: We train on 4-bit, but save as 16-bit. Is the quality actually 16-bit?

**Short answer:** No, the quality is based on the 4-bit base we trained on. The "16-bit" refers to the precision format, not the quality level.

**Explanation:**

```
Training: 4-bit base + LoRA → Quality limited by 4-bit base
Saving: Saved in 16-bit format → Just the file format
Result: 4-bit-trained quality in 16-bit format
```

**Why this is fine:**
- Standard practice across the industry
- Keeps VRAM low during training (12GB vs 30GB+)
- Quality is excellent for fine-tuned models
- Most users convert to GGUF anyway (which is quantized)

**If you need true 16-bit quality:**
You'd need to:
1. Train LoRA on 4-bit base (memory efficient)
2. Load the ORIGINAL unquantized 16-bit base model
3. Apply LoRA adapters to 16-bit base
4. Merge and save

This requires ~30GB+ VRAM just for merging and adds complexity. For most use cases, the current approach (4-bit training → 16-bit format) produces excellent results.

### Q: Can I quantize to other formats?

GGUF quantizations available: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q2_K

For other formats (AWQ, GPTQ), use external tools after getting merged_16bit.

## Getting Help

### Q: Where do I report bugs?

GitHub Issues: [https://github.com/yourusername/unsloth-finetuning/issues](https://github.com/yourusername/unsloth-finetuning/issues)

### Q: Where can I ask questions?

- GitHub Discussions
- Unsloth Discord
- This documentation

### Q: How do I contribute?

See main README's Contributing section. Pull requests welcome!

## See Also

- [Configuration Guide](CONFIGURATION.md) - All settings explained
- [Training Guide](TRAINING.md) - Training tips and best practices
- [Installation Guide](INSTALLATION.md) - Detailed setup instructions
- [Benchmarking Guide](BENCHMARK.md) - Optional validation and testing
