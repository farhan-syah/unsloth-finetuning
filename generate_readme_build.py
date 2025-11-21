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
from dotenv import load_dotenv

# Load .env for HuggingFace links and author name (non-training config)
load_dotenv(override=True)

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
    print("   Run 'python train.py' first")
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

# Parse model and dataset names
dataset_short_name = DATASET_NAME.split("/")[-1].lower().replace("_", "-")
dataset_author = DATASET_NAME.split("/")[0] if "/" in DATASET_NAME else "Unknown"

# Generate HuggingFace repo names (for cross-linking)
if HF_MODEL_NAME == "auto" or not HF_MODEL_NAME:
    hf_model_name = output_model_name
else:
    hf_model_name = HF_MODEL_NAME

# HuggingFace repository links (if username is provided)
if HF_USERNAME:
    HF_LORA_REPO = f"{HF_USERNAME}/{hf_model_name}-lora"
    HF_MERGED_REPO = f"{HF_USERNAME}/{hf_model_name}"
    HF_GGUF_REPO = f"{HF_USERNAME}/{hf_model_name}-gguf"
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
- **Training Time**: {training_time}
- **Training Steps**: {"~" + str(total_steps) if total_steps != "Unknown" else "Unknown"}
- **Dataset Samples**: {samples_trained if samples_trained != "Unknown" else "See dataset"}
- **Training Mode**: {"Quick test" if MAX_STEPS > 0 and MAX_STEPS < 500 else "Full training"}

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
- **Training Time**: {training_time}
- **Training Loss**: {final_loss}
- **Max Seq Length**: {MAX_SEQ_LENGTH}
- **Training Mode**: {"Quick test (limited steps/samples)" if MAX_STEPS > 0 and MAX_STEPS < 500 else "Full training"}

For complete training configuration, see the LoRA directory.

## Building Additional Formats

To create other formats from LoRA adapters:

```bash
# Edit .env
OUTPUT_FORMATS=lora_only,merged_16bit,gguf_q4_k_m

# Run build script
python build.py
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
        size_info = "Varies by quantization (2-8GB per file)"
        usage_lib = "llama.cpp / Ollama"
    else:
        description = "Model format"
        size_info = "Unknown"
        usage_lib = "Unknown"

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

## Training Details

- **LoRA Rank**: {LORA_RANK}
- **Training Time**: {training_time}
- **Training Loss**: {final_loss}
- **Max Seq Length**: {MAX_SEQ_LENGTH}
- **Training Mode**: {"Quick test" if MAX_STEPS > 0 and MAX_STEPS < 500 else "Full training"}

For complete training configuration, see the LoRA adapters repository/directory.

## Usage

"""

    if "merged" in format_name:
        readme += f"""### With Transformers

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
        readme += f"""### Available Quantizations

This folder contains multiple GGUF quantizations. Choose based on your needs:
- **Q4_K_M** (2.4GB): Best balance of quality and size (recommended)
- **Q5_K_M** (3.0GB): Better quality, larger size
- **Q8_0** (4.0GB): High quality, closer to original

### With Ollama

```bash
# Create Modelfile (using Q4_K_M as example)
cat > Modelfile <<EOF
FROM {format_path}/model.Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create model
ollama create {output_model_name.lower()} -f Modelfile

# Run
ollama run {output_model_name.lower()} "Hello!"
```

### With llama.cpp

```bash
# Run directly (using Q4_K_M as example)
llama-cli -m {format_path}/model.Q4_K_M.gguf -p "Hello!"
```
"""

    readme += f"""
## License

Based on {LORA_BASE_MODEL} and trained on {DATASET_NAME}.
Please refer to the original model and dataset licenses.

## Framework Versions

- Unsloth: 2025.11.3
- Transformers: 4.57.1
- PyTorch: 2.9.0+cu128

---

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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
