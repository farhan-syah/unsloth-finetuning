# Distribution Best Practices

## Recommended Structure

### For Training/Development (Local)
```
outputs/Qwen3-4B/
├── lora/              # LoRA adapters (continued training)
├── merged_16bit/      # Source model (base for all conversions)
└── gguf/              # All GGUF quantizations together
    ├── Q4_K_M.gguf
    ├── Q5_K_M.gguf
    ├── Q8_0.gguf
    ├── Modelfile      # Ollama config
    └── README.md      # GGUF usage guide
```

### For HuggingFace Hub Distribution

**Option A: Separate Repos (Recommended for large models)**
```
username/Qwen3-4B-Alpaca          # Main repo (merged_16bit)
username/Qwen3-4B-Alpaca-GGUF     # GGUF repo (all quants)
username/Qwen3-4B-Alpaca-LoRA     # LoRA adapters only
```

**Option B: Single Repo with Subdirs (Simpler)**
```
username/Qwen3-4B-Alpaca/
├── model-00001.safetensors   # Main model (merged_16bit)
├── *.json, *.txt
├── README.md
├── lora/                     # LoRA adapters
└── gguf/                     # GGUF quantizations
    ├── Q4_K_M.gguf
    ├── Q5_K_M.gguf
    └── README.md
```

## What to Share Publicly

### Minimum (Space-conscious):
```
✅ merged_16bit/      # Full model - users can quantize themselves
✅ lora/              # Tiny, useful for continued training
✅ gguf/Q4_K_M.gguf   # Most popular quant for Ollama
```

### Recommended:
```
✅ merged_16bit/      # Full model (transformers)
✅ lora/              # LoRA adapters
✅ gguf/
    ├── Q4_K_M.gguf   # Good quality/size balance (most popular)
    ├── Q5_K_M.gguf   # Better quality
    └── Q8_0.gguf     # Best quality
```

### Complete:
```
✅ merged_16bit/      # Full precision
✅ merged_4bit/       # 4-bit safetensors (for specific use cases)
✅ lora/              # LoRA adapters
✅ gguf/
    ├── F16.gguf      # Full precision GGUF
    ├── Q8_0.gguf     # 8-bit
    ├── Q6_K.gguf     # 6-bit
    ├── Q5_K_M.gguf   # 5-bit medium
    ├── Q4_K_M.gguf   # 4-bit medium (most popular)
    └── Q2_K.gguf     # 2-bit (very small, lower quality)
```

## File Sizes (Qwen3-4B example)

| Format | Size | Use Case |
|--------|------|----------|
| lora/ | 130 MB | Continued training |
| merged_16bit/ | 7.6 GB | Base for everything |
| merged_4bit/ | 3.4 GB | HF inference |
| gguf/F16.gguf | 7.6 GB | GGUF full precision |
| gguf/Q8_0.gguf | 4.0 GB | Best GGUF quality |
| gguf/Q5_K_M.gguf | 3.0 GB | Good quality |
| gguf/Q4_K_M.gguf | 2.4 GB | **Most popular** |
| gguf/Q2_K.gguf | 1.5 GB | Smallest |

## Recommended: Group GGUF Together

### Current .env setting:
```bash
# Build multiple GGUF quants - they'll go to separate folders
OUTPUT_FORMATS=gguf_q4_k_m,gguf_q5_k_m,gguf_q8_0
```

### Better: Update build.py to group GGUF

All GGUF quantizations should go into:
```
outputs/Qwen3-4B/gguf/
├── Q4_K_M.gguf
├── Q5_K_M.gguf
├── Q8_0.gguf
├── Modelfile
├── tokenizer files
└── README.md
```

This makes it easier to:
1. Share on HuggingFace (one gguf/ folder with all quants)
2. Download (users pick the quant they want)
3. Maintain (all GGUF versions in one place)

## HuggingFace Hub Examples

**Popular pattern:**
```
TheBloke/Llama-2-7B-GGUF       # Separate GGUF repo
TheBloke/Llama-2-7B-fp16       # Separate FP16 repo
TheBloke/Llama-2-7B-AWQ        # Separate AWQ repo
```

**Alternative (what we recommend):**
```
yourname/Qwen3-4B-Alpaca/
├── Main branch: merged_16bit (safetensors)
├── gguf/ folder: All GGUF quants
└── lora/ folder: LoRA adapters
```

## Summary

**Best practice**:
- Keep `lora/` and `merged_16bit/` separate (different use cases)
- **Group all GGUF quants in `gguf/` folder** (easier sharing)
- Optionally have `merged_4bit/` for specific use cases

This balances:
- ✅ Developer needs (separate lora, merged_16bit)
- ✅ User needs (all GGUF quants together)
- ✅ Disk space (grouped, easy to delete)
- ✅ HuggingFace Hub standards
