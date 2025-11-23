# Benchmarking Guide

Validate your fine-tuned models using industry-standard benchmarks.

## Table of Contents

- [Quick Start](#quick-start)
- [Why Benchmark?](#why-benchmark)
- [Setup](#setup)
- [Running Benchmarks](#running-benchmarks)
- [Understanding Results](#understanding-results)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Install dependencies
pip install lm-eval langdetect

# Run benchmark (no timeout by default)
python scripts/benchmark.py

# Optional: Enable watchdog timeout to auto-kill if stuck
python scripts/benchmark.py --timeout 30min
```

The interactive tool will guide you through model selection, backend choice, and benchmark configuration.

## Why Benchmark?

Benchmarking helps you:

1. **Validate training effectiveness** - Did fine-tuning actually improve the model?
2. **Compare datasets** - Understand which datasets work well with which models
3. **Detect issues** - Catch catastrophic forgetting (knowledge loss) early
4. **Track improvements** - Measure progress across training experiments

**Note:** Benchmarking is optional but recommended before deploying models.

## Setup

### Prerequisites

```bash
# Install lm-evaluation-harness and dependencies
pip install lm-eval langdetect sacrebleu rouge-score scikit-learn immutabledict sqlitedict pycountry

# Verify installation
python -m lm_eval --help
```

### Optional: Ollama Backend

If testing GGUF models via Ollama:

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve
```

### Build Models First

Benchmarking requires merged models:

```bash
# Build merged model
python scripts/build.py

# This creates merged_16bit/ directory needed for benchmarking
```

## Running Benchmarks

### Interactive Mode

```bash
python scripts/benchmark.py
```

You'll be prompted for:

1. **Model Selection** - Auto-detects trained models from `outputs/`
2. **Comparison Mode** - Compare with base model? (Recommended: Yes)
3. **Backend** - HuggingFace (merged models) or Ollama (GGUF)
4. **Benchmark Suite** - Test mode, Quick, or Full
5. **Batch Size** - Based on your GPU VRAM

### Benchmark Suites

| Suite | Tasks | Time | Best For |
|-------|-------|------|----------|
| **Test Mode** | IFEval (5 samples) | ~30 sec | Testing the benchmark workflow |
| **Quick** | IFEval | ~15 min | Quick validation |
| **Core** | IFEval + GSM8K + HellaSwag | ~30 min | Balanced evaluation |
| **Full** | Core + MMLU | ~60 min | Complete validation before deployment |

**Currently Supported:** Test mode and IFEval are fully tested. Other benchmarks (GSM8K, HellaSwag, MMLU) are available but may require additional testing.

### Test Mode

Perfect for trying out the benchmark system:

```bash
python scripts/benchmark.py
# Select: Test Mode (5 samples only)
```

Runs quickly (~30 seconds) on 5 samples instead of the full dataset. Use this while learning the system.

### Backends

#### HuggingFace Backend (Recommended)

Tests merged models directly:

```bash
# Requires merged_16bit/ or merged_4bit/ directory
# Auto-selected if available
```

**Pros:**
- Direct evaluation
- No external dependencies
- Accurate results

**Cons:**
- Uses more VRAM
- Slower than Ollama

#### Ollama Backend

Tests GGUF models via llama.cpp:

```bash
# Requires ollama serve running
# Requires model loaded in Ollama
```

**Pros:**
- Memory efficient
- Fast inference

**Cons:**
- Requires Ollama setup
- Manual model loading

## Understanding Results

### Comparison Mode (Recommended)

Benchmarking both base and fine-tuned models shows training impact:

**Example Output:**

```
Base vs Fine-tuned Comparison:
──────────────────────────────────────────────────────────────
Benchmark                      Base         Fine-tuned   Improvement
──────────────────────────────────────────────────────────────
Instruction-Following          40.00%       60.00%       ↗ +20.00% (+50.0%)
```

**Interpretation:**
- **Base:** Base model score before fine-tuning
- **Fine-tuned:** Your model after fine-tuning
- **Improvement:** Both absolute (+20.00%) and relative (+50.0%) gains

### Visual Indicators

| Symbol | Meaning |
|--------|---------|
| **↗** | Improvement (good) |
| **↘** | Regression (may indicate forgetting) |
| **→** | No change |

### IFEval Detailed Metrics

IFEval provides 4 detailed metrics:

| Metric | Tests |
|--------|-------|
| **Strict Prompt** | All instructions followed exactly |
| **Strict Inst** | Each instruction followed exactly |
| **Loose Prompt** | Instructions followed approximately |
| **Loose Inst** | Each instruction followed approximately |

These appear in generated README files for detailed analysis.

### What's a Good Result?

**Instruction-Following (IFEval):**
- **+10% or more:** Excellent - Clear improvement
- **+5-10%:** Good - Noticeable improvement
- **+2-5%:** Marginal - Minor improvement
- **<+2%:** Weak - Consider adjusting hyperparameters

**General Guidelines:**
- Focus on improvements in your target domain
- Small regressions (<3%) on other tasks are acceptable
- Large drops (>5%) may indicate catastrophic forgetting

### Output Files

Results are saved in your model directory:

```
outputs/model-name/
└── benchmarks/
    ├── benchmark.json              # Summary with comparison results
    ├── base/                       # Base model benchmark (run once)
    │   └── lm-eval/
    │       ├── results_*.json
    │       └── samples_*.jsonl
    └── fine-tuned/                 # Fine-tuned model benchmarks
        └── 2025-11-23T10-30-15/   # Timestamp of each run
            └── lm-eval/
                ├── results_*.json
                └── samples_*.jsonl
```

**Key files:**
- `benchmark.json` - Used by README generation, contains summary
- `results_*.json` - Full lm-eval output with all metrics
- `samples_*.jsonl` - Individual predictions for each test sample

### README Integration

Benchmark results automatically appear in generated README files:

```bash
# Regenerate READMEs with benchmark results
python scripts/generate_readme_build.py
```

Results appear in:
- `outputs/model-name/README.md`
- `outputs/model-name/lora/README.md`
- `outputs/model-name/merged_16bit/README.md`

### Regenerate Benchmark Summary

If you need to regenerate the benchmark.json summary without re-running benchmarks:

```bash
# Regenerate summary from existing benchmark results
python scripts/generate_benchmark_summary.py [model_path]
```

This combines base + latest fine-tuned results into `benchmarks/benchmark.json`. Useful when:
- You want to update the summary after editing benchmark results
- You need to regenerate README files with correct benchmark data
- You're troubleshooting benchmark display issues

## Troubleshooting

### Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size when prompted (try 4 or 1)
2. Use Ollama backend (more memory efficient)
3. Close other GPU applications
4. Use merged_4bit instead of merged_16bit

### Ollama Connection Failed

**Error:** `Connection refused`

**Solution:**
```bash
# Start Ollama in another terminal
ollama serve

# Verify it's running
ollama list
```

### Model Not Found

**Error:** `No merged model found`

**Solution:**
```bash
# Build merged model first
python scripts/build.py
```

### lm-eval Not Found

**Error:** `lm_eval: command not found`

**Solution:**
```bash
# Install lm-evaluation-harness
pip install lm-eval langdetect

# Verify
python -m lm_eval --help
```

### Benchmarks Too Slow

**Solutions:**
1. Use Test Mode (5 samples only)
2. Use Quick Suite (IFEval only)
3. Increase batch size if VRAM allows
4. Use Ollama backend (faster)

### Process Hangs at 100%

**Symptoms:** Progress shows 100% complete but process hangs with no output

**Why it happens:**

Common causes:
1. **Multiprocessing deadlock** - lm-eval workers stall during "finalizing results"
2. **Long generation** - Model loops forever on specific samples
3. **Bad samples** - Dataset has problematic formatting/encoding
4. **Backend timeout** - Model server (if using Ollama) doesn't respond

**What happens (auto-recovery):**
- Watchdog is **disabled by default** (no timeout)
- Enable with `--timeout`: e.g., `--timeout 30min` (or "10m", "600s", "600")
- When enabled:
  - After 60s of no output: Warning message appears
  - Every 30s: Progress update showing time elapsed and remaining
  - After timeout expires: Auto-terminates and attempts to recover partial results

**Solutions:**

Try in order:

1. **Increase timeout** - Give the process more time if needed
   ```bash
   python scripts/benchmark.py --timeout 30min
   ```

2. **Use Test Mode first** - Avoids most problematic samples (5 samples only)
   ```bash
   python scripts/benchmark.py
   # Select: Test Mode (5 samples only)
   ```

3. **Disable parallelism** - Fixes multiprocessing deadlocks (already done automatically)
   ```bash
   export TOKENIZERS_PARALLELISM=false
   export OMP_NUM_THREADS=1
   python scripts/benchmark.py
   ```

4. **Reduce batch size** - Use batch_size=1 when prompted

5. **Try different backend** - If HuggingFace hangs, try Ollama (or vice versa)

**Advanced debugging:**

If hangs persist, run lm-eval manually with debug mode:
```bash
lm_eval \
  --model hf \
  --model_args pretrained=./outputs/model/merged_16bit \
  --tasks ifeval \
  --batch_size 1 \
  --limit 10 \
  --debug
```

This shows exactly which sample causes the hang.

### Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'langdetect'`

**Solution:**
```bash
# Install all task dependencies
pip install langdetect sacrebleu rouge-score scikit-learn immutabledict sqlitedict pycountry
```

## Best Practices

### 1. Always Use Comparison Mode

Compare with base model to see training impact:
- ❌ "My model scores 58% on IFEval" - Is this good?
- ✅ "My model improved IFEval by +18%" - Training worked!

### 2. Start with Test Mode

When learning the system:
```bash
python scripts/benchmark.py
# Select: Test Mode (5 samples only)
```

This runs in ~30 seconds and helps you understand the workflow.

### 3. Use Full Suite Before Deployment

During development:
```bash
# Quick iteration: Test Mode or Quick Suite
python scripts/benchmark.py
```

Before deployment:
```bash
# Complete validation: Full Suite
python scripts/benchmark.py  # Select: Full
```

### 4. Track Experiments

Save results for comparison:
```bash
# After each experiment
mkdir -p experiments/
cp outputs/model/benchmarks/benchmark.json experiments/exp1-lr2e4.json
cp outputs/model/benchmarks/benchmark.json experiments/exp2-lr5e4.json
```

### 5. Document Results

Keep a log of what works:

| Experiment | Dataset | Learning Rate | LoRA Rank | IFEval Δ | Notes |
|------------|---------|---------------|-----------|----------|-------|
| 1 | LIMA | 2e-4 | 16 | +18.0% | Baseline |
| 2 | LIMA | 5e-4 | 32 | +22.5% | Better! |
| 3 | Alpaca | 2e-4 | 16 | +8.0% | Lower quality |

## FAQ

**Q: How long do benchmarks take?**

- Test Mode: ~30 seconds
- Quick (IFEval): ~15 minutes
- Core: ~30 minutes
- Full: ~60 minutes
- With comparison: 2x time (runs twice)

**Q: Can I skip comparison mode?**

Yes, but you won't know if training helped. Comparison is highly recommended.

**Q: Do I need to benchmark?**

No, it's optional. Use it to validate training and compare datasets.

**Q: Can I benchmark on CPU?**

Technically yes, but very slow (10-20x slower). Not recommended.

**Q: Which backend should I use?**

- GGUF models → Ollama
- Merged models → HuggingFace
- Both formats → Test both

**Q: What if I get different scores each run?**

Small variations (<1%) are normal. Use consistent batch sizes for comparability.

**Q: Can I add custom benchmarks?**

Yes, lm-eval supports many tasks. See [lm-evaluation-harness docs](https://github.com/EleutherAI/lm-evaluation-harness).

## Resources

- **lm-evaluation-harness:** https://github.com/EleutherAI/lm-evaluation-harness
- **IFEval paper:** https://arxiv.org/abs/2311.07911
- **Available tasks:** Run `lm_eval --tasks list`

## Support

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Verify all dependencies installed: `pip install lm-eval langdetect`
3. Check GPU memory: `nvidia-smi`
4. Try Test Mode first to verify setup
5. Open an issue with error logs if problems persist

---

**Remember:** Benchmarking helps you understand training effectiveness and dataset quality. It's a validation tool, not a requirement. Start with Test Mode to learn the system, then use comparison mode to measure real training impact.
