"""
Generate proper README.md files for model outputs during BUILD
Reads actual training configuration from training_metrics.json checkpoint

This script is called by build.py when creating output formats.
It reads from training_metrics.json (saved during training) to ensure
READMEs reflect the actual training parameters, not local .env values.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env for HuggingFace links and author name (non-training config)
load_dotenv(override=True)

# Try to load Unsloth's template mapper FIRST (before transformers)
try:
    from unsloth.models.mapper import MODEL_TO_OLLAMA_TEMPLATE_MAPPER
    UNSLOTH_MAPPER_AVAILABLE = True
except ImportError:
    UNSLOTH_MAPPER_AVAILABLE = False
    MODEL_TO_OLLAMA_TEMPLATE_MAPPER = {}

# Try to load transformers for chat template extraction (after unsloth)
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not available - chat template will not be included")

def get_template_for_model(model_name):
    """Get Ollama template name for a given model using Unsloth's mapper."""
    if model_name in MODEL_TO_OLLAMA_TEMPLATE_MAPPER:
        return MODEL_TO_OLLAMA_TEMPLATE_MAPPER[model_name]
    return None

# Helper functions
def get_bool_env(key, default=False):
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')

# Load HuggingFace configuration from .env (for cross-linking repos)
HF_USERNAME = os.getenv("HF_USERNAME", "")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "auto")
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Your Name")
OUTPUT_DIR_BASE = os.getenv("OUTPUT_DIR_BASE", "./outputs")

# First, find the LoRA directory
# We need to detect which model was trained by looking at existing output directories
lora_dirs = []
if os.path.exists(OUTPUT_DIR_BASE):
    for item in os.listdir(OUTPUT_DIR_BASE):
        lora_path = os.path.join(OUTPUT_DIR_BASE, item, "lora")
        if os.path.exists(lora_path):
            lora_dirs.append((item, lora_path))

if not lora_dirs:
    print(f"❌ No LoRA directories found in {OUTPUT_DIR_BASE}")
    print("   Run 'python scripts/train.py' first")
    exit(1)

# Use the most recently modified LoRA directory
lora_dirs.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
output_model_name, LORA_DIR = lora_dirs[0]
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, output_model_name)

print(f"📂 Found model output: {output_model_name}")

# Load training configuration from training_metrics.json
metrics_path = os.path.join(LORA_DIR, "training_metrics.json")
if not os.path.exists(metrics_path):
    print(f"❌ training_metrics.json not found: {metrics_path}")
    print("   This file should be created by train.py during training")
    exit(1)

print(f"📊 Loading training configuration from: {metrics_path}")

with open(metrics_path) as f:
    metrics_data = json.load(f)

# Extract training configuration from metrics (these are the ACTUAL values used during training)
LORA_BASE_MODEL = metrics_data.get("model_name", "unknown")
DATASET_NAME = metrics_data.get("dataset_name", "unknown")
MAX_SEQ_LENGTH = metrics_data.get("max_seq_length", 4096)
LORA_RANK = metrics_data.get("lora_rank", 16)
LORA_ALPHA = metrics_data.get("lora_alpha", 32)
BATCH_SIZE = metrics_data.get("batch_size", 2)
GRADIENT_ACCUMULATION_STEPS = metrics_data.get("gradient_accumulation_steps", 4)
LEARNING_RATE = metrics_data.get("learning_rate", 0.0002)
NUM_TRAIN_EPOCHS = metrics_data.get("num_train_epochs", 1)
MAX_STEPS = metrics_data.get("max_steps", 0)
DATASET_MAX_SAMPLES = metrics_data.get("dataset_max_samples", 0)
PACKING = metrics_data.get("packing", False)

# Extract training results
training_time = f"{metrics_data.get('training_time_minutes', 0):.1f} minutes"
final_loss = f"{metrics_data.get('final_loss', 0):.4f}"
total_steps = metrics_data.get("total_steps", "Unknown")
samples_trained = metrics_data.get("dataset_samples", "Unknown")

print(f"✅ Loaded configuration:")
print(f"   Base model: {LORA_BASE_MODEL}")
print(f"   Dataset: {DATASET_NAME}")
print(f"   Max seq length: {MAX_SEQ_LENGTH}")
print(f"   LoRA rank: {LORA_RANK}")
print(f"   Training steps: {total_steps}")

# Load chat template - will be populated after finding LoRA directory
chat_template_name = None
chat_template_format = None

# Load chat template from chat_template.json (created by build.py)
chat_template_json_path = os.path.join(OUTPUT_DIR, "merged_16bit", "chat_template.json")
if os.path.exists(chat_template_json_path):
    try:
        print(f"\n📝 Loading chat template from {chat_template_json_path}...")
        with open(chat_template_json_path) as f:
            chat_template_info = json.load(f)
            chat_template_name = chat_template_info.get("template_name")
            chat_template_format = chat_template_info.get("template_format")
            if chat_template_name and chat_template_format:
                print(f"✅ Chat template loaded: {chat_template_name}")
    except Exception as e:
        print(f"⚠️  Could not load chat template: {e}")

# Load benchmark results if available
benchmark_results = None

# Try new location first: benchmarks/benchmark.json
new_benchmark_path = os.path.join(OUTPUT_DIR, "benchmarks", "benchmark.json")
old_benchmark_path = os.path.join(OUTPUT_DIR, "benchmark.json")

if os.path.exists(new_benchmark_path):
    try:
        print(f"\n📊 Loading benchmark results from {new_benchmark_path}...")
        with open(new_benchmark_path) as f:
            benchmark_results = json.load(f)
            print(f"✅ Benchmark results loaded (new format)")
    except Exception as e:
        print(f"⚠️  Could not load benchmark results: {e}")
elif os.path.exists(old_benchmark_path):
    try:
        print(f"\n📊 Loading benchmark results from {old_benchmark_path} (legacy)...")
        with open(old_benchmark_path) as f:
            benchmark_results = json.load(f)
            print(f"✅ Benchmark results loaded (legacy format)")
    except Exception as e:
        print(f"⚠️  Could not load benchmark results: {e}")

# Parse model and dataset names
dataset_short_name = DATASET_NAME.split("/")[-1].lower().replace("_", "-")
dataset_author = DATASET_NAME.split("/")[0] if "/" in DATASET_NAME else "Unknown"

# Helper function to generate benchmark results section
def generate_benchmark_section(benchmark_data, format_type="markdown"):
    """
    Generate a simple benchmark table with model names as rows and tasks as columns.
    No opinions, no descriptions - just raw scores.

    Args:
        benchmark_data: Loaded benchmark.json data
        format_type: "markdown" or "table"

    Returns:
        Formatted markdown string with benchmark results
    """
    if not benchmark_data:
        return ""

    # Extract model names and results
    base_model_name = benchmark_data.get("base_model_name", "Base Model")
    # Use model_name for fine-tuned (this is the output directory name like "Llama-3.2-1B-Instruct-bnb-4bit-gsm8k")
    ft_model_name = benchmark_data.get("model_name", "Fine-tuned Model")

    # Normalize data structure (handle both old and new formats)
    base_model_data = benchmark_data.get("base_model", {})
    ft_model_data = benchmark_data.get("fine_tuned_model", {})

    # Parse results from benchmark data
    # New format: {results: {task_name: {metric: value}}}
    # Old format: {huggingface: {task_name: {metric: value}}}

    def get_task_scores(model_data):
        """Extract task scores from either format"""
        if "results" in model_data:
            return model_data["results"]
        elif "huggingface" in model_data:
            return model_data["huggingface"]
        return {}

    base_scores = get_task_scores(base_model_data)
    ft_scores = get_task_scores(ft_model_data)

    if not ft_scores:
        return ""

    # Get all unique tasks from both base and ft
    all_tasks = sorted(set(list(base_scores.keys()) + list(ft_scores.keys())))

    if not all_tasks:
        return ""

    # Build simple table: Model names as rows, tasks as columns
    section = "\n## Benchmark Results\n\n"

    # Add note about benchmark model
    section += "*Benchmarked on the merged 16-bit safetensor model*\n\n"

    # Add timestamp
    timestamp = benchmark_data.get("timestamp", "")
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M")
            section += f"*Evaluated: {timestamp}*\n\n"
        except:
            pass

    # Table header with Type column
    section += "| Model | Type |"
    for task in all_tasks:
        section += f" {task} |"
    section += "\n"

    # Table separator
    section += "|-------|------|"
    for _ in all_tasks:
        section += "--------|"
    section += "\n"

    # Helper function to get primary metric from task
    def get_primary_metric(task_metrics):
        """Get the main score from a task's metrics"""
        if not task_metrics or not isinstance(task_metrics, dict):
            return None

        # Priority order for metrics
        priority_metrics = [
            "prompt_level_strict_acc",
            "inst_level_strict_acc",
            "exact_match",
            "accuracy",
            "acc"
        ]

        # Try priority metrics first
        for metric in priority_metrics:
            clean_metric = metric.replace(",none", "")
            if clean_metric in task_metrics:
                return task_metrics[clean_metric]
            if f"{metric},none" in task_metrics:
                return task_metrics[f"{metric},none"]

        # Fallback: first numeric value
        for value in task_metrics.values():
            if isinstance(value, (int, float)):
                return value

        return None

    # Base model row (if available)
    if base_scores:
        section += f"| {base_model_name} | Base |"
        for task in all_tasks:
            if task in base_scores:
                score = get_primary_metric(base_scores[task])
                if score is not None:
                    section += f" {score:.4f} |"
                else:
                    section += " - |"
            else:
                section += " - |"
        section += "\n"

    # Fine-tuned model row
    section += f"| {ft_model_name} | Fine-tuned |"
    for task in all_tasks:
        if task in ft_scores:
            score = get_primary_metric(ft_scores[task])
            if score is not None:
                section += f" {score:.4f} |"
            else:
                section += " - |"
        else:
            section += " - |"
    section += "\n\n"

    return section

def get_training_scope():
    """
    Calculate and display actual samples used during training.
    Shows "Samples Used: X/Total" format.
    """
    # Calculate actual samples used
    if MAX_STEPS > 0:
        # Limited by steps: samples_used = steps × effective_batch_size
        effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
        samples_used = MAX_STEPS * effective_batch
    else:
        # Full epochs
        if DATASET_MAX_SAMPLES > 0:
            # Limited by DATASET_MAX_SAMPLES
            samples_used = DATASET_MAX_SAMPLES * NUM_TRAIN_EPOCHS
        else:
            # Full dataset
            samples_used = samples_trained if samples_trained != "Unknown" else "Unknown"

    # Get total available
    total_samples = samples_trained if samples_trained != "Unknown" else "Unknown"

    # Format output
    if samples_used == "Unknown" or total_samples == "Unknown":
        return f"{NUM_TRAIN_EPOCHS} epoch(s)"

    # Show as fraction if limited, otherwise show full
    if samples_used < total_samples:
        return f"{samples_used:,}/{total_samples:,} samples ({NUM_TRAIN_EPOCHS} epoch(s))"
    else:
        return f"{total_samples:,} samples ({NUM_TRAIN_EPOCHS} epoch(s), full dataset)"

# Generate HuggingFace repo names (for cross-linking)
if HF_MODEL_NAME == "auto" or not HF_MODEL_NAME:
    hf_model_name = output_model_name
else:
    hf_model_name = HF_MODEL_NAME

# HuggingFace repository links (if username is provided)
if HF_USERNAME:
    HF_LORA_REPO = f"{HF_USERNAME}/{hf_model_name}-lora"
    HF_MERGED_REPO = f"{HF_USERNAME}/{hf_model_name}"
    HF_GGUF_REPO = f"{HF_USERNAME}/{hf_model_name}-GGUF"
    HF_LORA_URL = f"https://huggingface.co/{HF_LORA_REPO}"
    HF_MERGED_URL = f"https://huggingface.co/{HF_MERGED_REPO}"
    HF_GGUF_URL = f"https://huggingface.co/{HF_GGUF_REPO}"
else:
    HF_LORA_REPO = None
    HF_MERGED_REPO = None
    HF_GGUF_REPO = None
    HF_LORA_URL = None
    HF_MERGED_URL = None
    HF_GGUF_URL = None

# Generate LoRA adapter README
lora_readme = f"""---
base_model: {LORA_BASE_MODEL}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- transformers
- trl
- unsloth
- {dataset_short_name.lower()}
language:
- en
license: apache-2.0
---

# {output_model_name} - LoRA Adapters

Fine-tuned LoRA adapters for [{LORA_BASE_MODEL}](https://huggingface.co/{LORA_BASE_MODEL}) using supervised fine-tuning.

## Model Details

- **Base Model**: [{LORA_BASE_MODEL}](https://huggingface.co/{LORA_BASE_MODEL})
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: [{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME})
- **Training Framework**: Unsloth + TRL + Transformers
- **Adapter Type**: PEFT LoRA adapters only (requires base model)
"""

# Add chat template for LoRA README if available
if chat_template_format:
    lora_readme += f"""
## Prompt Format

This model uses the **{chat_template_name}** chat template.

Use the tokenizer's `apply_chat_template()` method:

```python
messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Your question here"}}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
```
"""

lora_readme += f"""
## Training Configuration

### LoRA Parameters
- **LoRA Rank (r)**: {LORA_RANK}
- **LoRA Alpha**: {LORA_ALPHA}
- **LoRA Dropout**: 0.0
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Hyperparameters
- **Learning Rate**: {LEARNING_RATE}
- **Batch Size**: {BATCH_SIZE}
- **Gradient Accumulation Steps**: {GRADIENT_ACCUMULATION_STEPS}
- **Effective Batch Size**: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}
- **Epochs**: {NUM_TRAIN_EPOCHS}
- **Max Sequence Length**: {MAX_SEQ_LENGTH}
- **Optimizer**: adamw_8bit
- **Packing**: {PACKING}
- **Weight Decay**: 0.01
- **Learning Rate Scheduler**: linear

### Training Results
- **Training Loss**: {final_loss}
- **Training Steps**: {total_steps if total_steps != "Unknown" else "Unknown"}
- **Dataset Samples**: {samples_trained if samples_trained != "Unknown" else "See dataset"}
- **Training Scope**: {get_training_scope()}
"""

# Add benchmark results if available
if benchmark_results:
    lora_readme += generate_benchmark_section(benchmark_results)

lora_readme += f"""
## Usage

### Load with Transformers + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{LORA_BASE_MODEL}",
    load_in_4bit=True,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "path/to/lora")
tokenizer = AutoTokenizer.from_pretrained("{LORA_BASE_MODEL}")

# Generate
messages = [{{"role": "user", "content": "Your question here"}}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Load with Unsloth (Recommended)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/lora",
    max_seq_length={MAX_SEQ_LENGTH},
    dtype=None,
    load_in_4bit=True,
)

# For inference
FastLanguageModel.for_inference(model)

# Generate
messages = [{{"role": "user", "content": "Your question here"}}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Related Models

{f'- **Merged Model**: [{HF_MERGED_REPO}]({HF_MERGED_URL}) - Ready-to-use merged model' if HF_MERGED_URL else '- **Merged Model**: Available locally in the output directory'}
{f'- **GGUF Quantized**: [{HF_GGUF_REPO}]({HF_GGUF_URL}) - GGUF format for llama.cpp/Ollama' if HF_GGUF_URL else ''}

## Dataset

Training dataset: [{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME})

Please refer to the dataset documentation for licensing and usage restrictions.

## Merge with Base Model

To create a standalone merged model:

```python
from unsloth import FastLanguageModel

# Load model with LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="path/to/lora",
    max_seq_length={MAX_SEQ_LENGTH},
    dtype=None,
    load_in_4bit=True,
)

# Save merged 16-bit model
model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")

# Or save as GGUF for llama.cpp/Ollama
model.save_pretrained_gguf("model.gguf", tokenizer, quantization_method="q4_k_m")
```

## Framework Versions

- **Unsloth**: 2025.11.3
- **Transformers**: 4.57.1
- **PyTorch**: 2.9.0+cu128
- **PEFT**: 0.18.0
- **TRL**: 0.22.2
- **Datasets**: 4.3.0

## License

This model is based on {LORA_BASE_MODEL} and trained on {DATASET_NAME}.
Please refer to the original model and dataset licenses for usage terms.

## Credits

**Trained by:** {AUTHOR_NAME}

**Training pipeline:**
- [unsloth-finetuning](https://github.com/farhan-syah/unsloth-finetuning) by [@farhan-syah](https://github.com/farhan-syah)
- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster LLM fine-tuning

**Base components:**
- Base model: [{LORA_BASE_MODEL}](https://huggingface.co/{LORA_BASE_MODEL})
- Training dataset: [{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME}) by {dataset_author}

## Citation

If you use this model, please cite:

```bibtex
@misc{{{output_model_name.lower().replace("-", "_")}_lora,
  author = {{{AUTHOR_NAME}}},
  title = {{{output_model_name} Fine-tuned with LoRA}},
  year = {{{datetime.now().year}}},
  note = {{Fine-tuned using Unsloth: https://github.com/unslothai/unsloth}},
  howpublished = {{\\url{{https://github.com/farhan-syah/unsloth-finetuning}}}}
}}
```
"""

# Generate main output directory README
main_readme = f"""# {output_model_name} Fine-tuning Output

This directory contains the fine-tuned model outputs from training {LORA_BASE_MODEL} on {DATASET_NAME}.

## Directory Structure

```
{OUTPUT_DIR}/
├── lora/                   # LoRA adapters (142MB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer files
│   └── README.md
├── merged_16bit/          # (Optional) Full merged 16-bit model
├── merged_4bit/           # (Optional) Full merged 4-bit model
├── gguf/                  # (Optional) All GGUF quantizations
│   ├── model.Q4_K_M.gguf
│   ├── model.Q5_K_M.gguf
│   ├── tokenizer files
│   └── README.md
└── README.md             # This file
```

## What's Included

### LoRA Adapters (`lora/`)
- Small adapter weights (~142MB)
- Requires base model to use
- Most efficient for storage/sharing
- See `lora/README.md` for usage

### Optional Formats
Depending on OUTPUT_FORMATS in .env, you may also have:
- **merged_16bit**: Full model with LoRA merged (8-16GB)
- **merged_4bit**: Quantized merged model (4-8GB)
- **gguf**: All GGUF quantizations for llama.cpp/Ollama (in one folder)

## Quick Start

### Using LoRA Adapters

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{LORA_DIR}",
    max_seq_length={MAX_SEQ_LENGTH},
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

messages = [{{"role": "user", "content": "Hello!"}}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### Using Merged Model (if available)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{OUTPUT_DIR}/merged_16bit",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_DIR}/merged_16bit")

messages = [{{"role": "user", "content": "Hello!"}}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### Using GGUF with Ollama (if available)

```bash
# Create Modelfile
cat > Modelfile <<EOF
FROM {OUTPUT_DIR}/gguf/model.Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create Ollama model
ollama create {output_model_name.lower()} -f Modelfile

# Run
ollama run {output_model_name.lower()} "Hello!"
```

## Training Details

- **Base Model**: {LORA_BASE_MODEL}
- **Dataset**: {DATASET_NAME}
- **LoRA Rank**: {LORA_RANK}
- **Training Steps**: {total_steps if total_steps != "Unknown" else "Unknown"}
- **Training Loss**: {final_loss}
- **Max Seq Length**: {MAX_SEQ_LENGTH}
- **Training Scope**: {get_training_scope()}

For complete training configuration, see the LoRA directory.
"""

# Add benchmark results if available
if benchmark_results:
    main_readme += generate_benchmark_section(benchmark_results)

main_readme += f"""
## Building Additional Formats

To create other formats from LoRA adapters:

```bash
# Edit .env
OUTPUT_FORMATS=lora_only,merged_16bit,gguf_q4_k_m

# Run build script
python scripts/build.py
```

Available formats:
- `lora_only` - Just LoRA adapters (default)
- `merged_16bit` - Full precision merged model
- `merged_4bit` - 4-bit quantized merged model
- `gguf_f16` - GGUF float16
- `gguf_q8_0` - GGUF 8-bit quantization
- `gguf_q4_k_m` - GGUF 4-bit quantization (recommended)
- `gguf_q5_k_m` - GGUF 5-bit quantization

## License

Please refer to the original model and dataset licenses:
- Base model: {LORA_BASE_MODEL}
- Dataset: {DATASET_NAME}

---

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Training pipeline: https://github.com/unslothai/unsloth
"""

# Generate format-specific READMEs
def generate_format_readme(format_name, format_path):
    """Generate README for merged/GGUF formats"""

    format_type = "Merged Model" if "merged" in format_name else "GGUF Format"

    if "merged_16bit" in format_name:
        description = "Full-precision (16-bit) merged model with LoRA adapters integrated."
        size_info = "~8-16GB"
        usage_lib = "transformers"
    elif "merged_4bit" in format_name:
        description = "4-bit quantized merged model with LoRA adapters integrated."
        size_info = "~4-8GB"
        usage_lib = "transformers with bitsandbytes"
    elif "gguf" == format_name:
        description = f"GGUF format quantizations for llama.cpp/Ollama."
        # Calculate actual size range from GGUF files (will be updated below if files exist)
        size_info = "Varies by quantization"
        usage_lib = "llama.cpp / Ollama"
    else:
        description = "Model format"
        size_info = "Unknown"
        usage_lib = "Unknown"

    # For GGUF, calculate actual size range from files
    if "gguf" == format_name and os.path.exists(format_path):
        gguf_sizes = []
        for f in Path(format_path).glob("*.gguf"):
            gguf_sizes.append(f.stat().st_size / (1024 * 1024 * 1024))  # Size in GB
        if gguf_sizes:
            min_size = min(gguf_sizes)
            max_size = max(gguf_sizes)
            size_info = f"{min_size:.2f} GB - {max_size:.2f} GB"

    readme = f"""---
base_model: {LORA_BASE_MODEL}
library_name: transformers
pipeline_tag: text-generation
tags:
- {format_name}
- fine-tuned
- {dataset_short_name.lower()}
language:
- en
license: apache-2.0
---

# {output_model_name} - {format_type}

{description}

## Model Details

- **Base Model**: [{LORA_BASE_MODEL}](https://huggingface.co/{LORA_BASE_MODEL})
- **Format**: {format_name}
- **Dataset**: [{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME})
- **Size**: {size_info}
- **Usage**: {usage_lib}

## Related Models

{f'- **LoRA Adapters**: [{HF_LORA_REPO}]({HF_LORA_URL}) - Smaller LoRA-only adapters' if HF_LORA_URL else '- **LoRA Adapters**: Available locally in the output directory'}
{f'- **Merged FP16 Model**: [{HF_MERGED_REPO}]({HF_MERGED_URL}) - Original unquantized model in FP16' if HF_MERGED_URL and format_name == 'gguf' else ''}
{f'- **GGUF Quantized**: [{HF_GGUF_REPO}]({HF_GGUF_URL}) - GGUF format for llama.cpp/Ollama' if HF_GGUF_URL and format_name == 'merged_16bit' else ''}
"""

    # Add chat template section based on format type
    if chat_template_format:
        if format_name == "gguf":
            # GGUF: Show Ollama template format
            readme += f"""
## Prompt Format

This model uses the **{chat_template_name}** chat template.

### Ollama Template Format

```
{chat_template_format}
```
"""
        else:
            # Merged models: Show Python usage
            readme += f"""
## Prompt Format

This model uses the **{chat_template_name}** chat template.

### Python Usage

Use the tokenizer's `apply_chat_template()` method:

```python
messages = [
    {{"role": "system", "content": "You are a helpful assistant."}},
    {{"role": "user", "content": "Your question here"}}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
```
"""

    readme += f"""

## Training Details

- **LoRA Rank**: {LORA_RANK}
- **Training Steps**: {total_steps if total_steps != "Unknown" else "Unknown"}
- **Training Loss**: {final_loss}
- **Max Seq Length**: {MAX_SEQ_LENGTH}
- **Training Scope**: {get_training_scope()}

For complete training configuration, see the LoRA adapters repository/directory.
"""

    # Add benchmark results if available
    # Show benchmarks in all formats (lora, merged, gguf)
    # Note: Benchmarks were run on the merged safetensor model
    if benchmark_results:
        readme += generate_benchmark_section(benchmark_results)

    readme += "\n"

    if "merged" in format_name:
        readme += f"""## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{format_path}",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{format_path}")

messages = [{{"role": "user", "content": "Your question here"}}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```
"""
    elif "gguf" == format_name:
        # Detect available GGUF files and generate table
        gguf_files = []
        gguf_dir_path = format_path  # format_path already points to gguf directory

        if os.path.exists(gguf_dir_path):
            for f in sorted(Path(gguf_dir_path).glob("*.gguf")):
                file_size = f.stat().st_size / (1024 * 1024 * 1024)  # Size in GB
                file_name = f.name
                # Extract quantization type from filename (e.g., Llama-3.2-1B-Instruct-bnb-4bit-lima-Q4_K_M.gguf -> Q4_K_M)
                # Format: {output_model_name}-{quant_type}.gguf
                quant_type = file_name.replace(".gguf", "").split("-")[-1]

                # Describe quantization quality
                if "Q2" in quant_type:
                    quality = "Smallest size, lower quality"
                elif "Q3" in quant_type:
                    quality = "Small size, moderate quality"
                elif "Q4" in quant_type:
                    quality = "Good balance (recommended)"
                elif "Q5" in quant_type:
                    quality = "Better quality, larger size"
                elif "Q6" in quant_type:
                    quality = "High quality"
                elif "Q8" in quant_type:
                    quality = "Very high quality, near original"
                elif "F16" in quant_type or "f16" in quant_type:
                    quality = "Full precision (largest)"
                else:
                    quality = "Unknown"

                gguf_files.append((file_name, quant_type, file_size, quality))

        if gguf_files:
            readme += f"""## Available Quantizations

| Quantization | File | Size | Quality |
|--------------|------|------|---------|
"""
            for file_name, quant_type, file_size, quality in gguf_files:
                readme += f"| **{quant_type}** | [{file_name}]({file_name}) | {file_size:.2f} GB | {quality} |\n"

            readme += f"""
**Usage:** Use the dropdown menu above to select a quantization, then follow HuggingFace's provided instructions.
"""
        else:
            # No GGUF files found - shouldn't happen but handle gracefully
            readme += f"""## Available Quantizations

No GGUF files found in this directory yet.
"""

    readme += f"""
## License

Based on {LORA_BASE_MODEL} and trained on {DATASET_NAME}.
Please refer to the original model and dataset licenses.

## Credits

**Trained by:** {AUTHOR_NAME}

**Training pipeline:**
- [unsloth-finetuning](https://github.com/farhan-syah/unsloth-finetuning) by [@farhan-syah](https://github.com/farhan-syah)
- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster LLM fine-tuning

**Base components:**
- Base model: [{LORA_BASE_MODEL}](https://huggingface.co/{LORA_BASE_MODEL})
- Training dataset: [{DATASET_NAME}](https://huggingface.co/datasets/{DATASET_NAME}) by {dataset_author}
"""

    return readme

# Write README files
print("\n📝 Generating README files...")

lora_readme_path = os.path.join(LORA_DIR, "README.md")
with open(lora_readme_path, "w") as f:
    f.write(lora_readme)
print(f"✅ Generated: {lora_readme_path}")

main_readme_path = os.path.join(OUTPUT_DIR, "README.md")
with open(main_readme_path, "w") as f:
    f.write(main_readme)
print(f"✅ Generated: {main_readme_path}")

# Generate READMEs for other formats if they exist
formats_to_check = ["merged_16bit", "merged_4bit", "gguf"]
for format_name in formats_to_check:
    format_path = os.path.join(OUTPUT_DIR, format_name)
    if os.path.exists(format_path):
        format_readme_path = os.path.join(format_path, "README.md")
        format_readme_content = generate_format_readme(format_name, format_path)
        with open(format_readme_path, "w") as f:
            f.write(format_readme_content)
        print(f"✅ Generated: {format_readme_path}")

print("\n✨ README files generated successfully!")
print(f"   Main: {main_readme_path}")
print(f"   LoRA: {lora_readme_path}")

# Show format-specific READMEs
format_readmes = [os.path.join(OUTPUT_DIR, fmt, "README.md") for fmt in formats_to_check
                  if os.path.exists(os.path.join(OUTPUT_DIR, fmt))]
if format_readmes:
    print(f"   Formats: {len(format_readmes)} additional READMEs")
