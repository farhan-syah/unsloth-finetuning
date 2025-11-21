"""
Fine-tune models using Unsloth with LoRA
Handles ONLY training - creates LoRA adapters
Use build.py to convert to merged/GGUF formats

Configure your dataset and model in .env file
"""

import torch
import os
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Helper functions
def get_bool_env(key, default=False):
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')

def get_int_env(key, default):
    return int(os.getenv(key, str(default)))

def get_float_env(key, default):
    return float(os.getenv(key, str(default)))

# Configuration from environment variables
LORA_BASE_MODEL = os.getenv("LORA_BASE_MODEL", "unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME", "auto")
MAX_SEQ_LENGTH = get_int_env("MAX_SEQ_LENGTH", 2048)
LORA_RANK = get_int_env("LORA_RANK", 64)
LORA_ALPHA = get_int_env("LORA_ALPHA", 128)
DATASET_NAME = os.getenv("DATASET_NAME", "yahma/alpaca-cleaned")
DATASET_MAX_SAMPLES = get_int_env("DATASET_MAX_SAMPLES", 0)  # 0 = use all
MAX_STEPS = get_int_env("MAX_STEPS", 0)  # 0 = use epochs
FORCE_PREPROCESS = get_bool_env("FORCE_PREPROCESS", False)
FORCE_RETRAIN = get_bool_env("FORCE_RETRAIN", False)
CHECK_SEQ_LENGTH = get_bool_env("CHECK_SEQ_LENGTH", False)  # Set true to filter out samples exceeding MAX_SEQ_LENGTH (slower)

# Generate output model name
dataset_short_name = DATASET_NAME.split("/")[-1].lower().replace("_", "-")
if OUTPUT_MODEL_NAME == "auto" or not OUTPUT_MODEL_NAME:
    # Auto-generate from base model + dataset
    model_base = LORA_BASE_MODEL.split("/")[-1].replace("-unsloth-bnb-4bit", "").replace("-unsloth", "")
    output_model_name = f"{model_base}-{dataset_short_name}"
else:
    output_model_name = OUTPUT_MODEL_NAME

# Base directories
OUTPUT_DIR_BASE = os.getenv("OUTPUT_DIR_BASE", "./outputs")
PREPROCESSED_DATA_DIR = os.getenv("PREPROCESSED_DATA_DIR", "./data/preprocessed")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")

# Auto-created paths
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, output_model_name)
LORA_DIR = os.path.join(OUTPUT_DIR, "lora")
PREPROCESSED_DATASET_PATH = os.path.join(PREPROCESSED_DATA_DIR, dataset_short_name)

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Training Configuration
BATCH_SIZE = get_int_env("BATCH_SIZE", 2)
GRADIENT_ACCUMULATION_STEPS = get_int_env("GRADIENT_ACCUMULATION_STEPS", 4)
LEARNING_RATE = get_float_env("LEARNING_RATE", 2e-4)
NUM_TRAIN_EPOCHS = get_int_env("NUM_TRAIN_EPOCHS", 1)
WARMUP_STEPS = get_int_env("WARMUP_STEPS", 5)
PACKING = get_bool_env("PACKING", False)
MAX_GRAD_NORM = get_float_env("MAX_GRAD_NORM", 1.0)
OPTIM = os.getenv("OPTIM", "adamw_8bit")
LOGGING_STEPS = get_int_env("LOGGING_STEPS", 10)
SAVE_STEPS = get_int_env("SAVE_STEPS", 500)
SAVE_TOTAL_LIMIT = get_int_env("SAVE_TOTAL_LIMIT", 3)
SAVE_ONLY_FINAL = get_bool_env("SAVE_ONLY_FINAL", False)  # Skip intermediate checkpoints
USE_GRADIENT_CHECKPOINTING = get_bool_env("USE_GRADIENT_CHECKPOINTING", True)
SEED = get_int_env("SEED", 3407)

# Wandb Configuration
WANDB_ENABLED = get_bool_env("WANDB_ENABLED", False)
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "unsloth-finetuning")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "auto")
if WANDB_RUN_NAME == "auto":
    WANDB_RUN_NAME = f"{output_model_name}-r{LORA_RANK}"

print("\n" + "="*60)
print("🦥 UNSLOTH TRAINING - LoRA Adapter Creation")
print("="*60)

# Check if LoRA adapters already exist
lora_exists = os.path.exists(os.path.join(LORA_DIR, "adapter_config.json"))
if lora_exists and not FORCE_RETRAIN:
    print(f"\n✅ LoRA adapters already exist at: {LORA_DIR}")
    print("   Skipping training.")
    print("   To retrain, set FORCE_RETRAIN=true in .env")
    print("\n💡 Use build.py to convert to merged/GGUF formats")
    exit(0)
elif lora_exists and FORCE_RETRAIN:
    print(f"\n⚠️  LoRA adapters exist but FORCE_RETRAIN=true")
    print(f"   Will overwrite: {LORA_DIR}")

print("\n📋 Configuration:")
print(f"  Model: {LORA_BASE_MODEL}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  LoRA Output: {LORA_DIR}")
print(f"  Preprocessed Data: {PREPROCESSED_DATASET_PATH}")
print(f"\n  Max Seq Length: {MAX_SEQ_LENGTH}")
print(f"  LoRA Rank: {LORA_RANK}, Alpha: {LORA_ALPHA}")
print(f"  Batch Size: {BATCH_SIZE}, Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Packing: {PACKING}")
print(f"  Learning Rate: {LEARNING_RATE}")

# Show training length
if MAX_STEPS > 0:
    print(f"  ⚡ TEST MODE: Max Steps: {MAX_STEPS}")
elif DATASET_MAX_SAMPLES > 0:
    print(f"  ⚡ TEST MODE: Dataset Samples: {DATASET_MAX_SAMPLES}")
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
else:
    print(f"  Epochs: {NUM_TRAIN_EPOCHS}")
print()

print("🦥 Loading model with 4-bit quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    trust_remote_code=True,
)

print("🔧 Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth" if USE_GRADIENT_CHECKPOINTING else False,
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

# Load or preprocess dataset
preprocess_exists = os.path.exists(PREPROCESSED_DATASET_PATH)

if preprocess_exists and not FORCE_PREPROCESS:
    print(f"\n📂 Loading preprocessed dataset from {PREPROCESSED_DATASET_PATH}")
    from datasets import load_from_disk
    dataset = load_from_disk(PREPROCESSED_DATASET_PATH)
    print(f"✅ Loaded {len(dataset)} samples (already filtered by MAX_SEQ_LENGTH)")
    print("   To reprocess, set FORCE_PREPROCESS=true in .env")
elif preprocess_exists and FORCE_PREPROCESS:
    print(f"\n⚠️  Preprocessed dataset exists but FORCE_PREPROCESS=true")
    print(f"   Will reprocess: {PREPROCESSED_DATASET_PATH}")
    import shutil
    shutil.rmtree(PREPROCESSED_DATASET_PATH)
    print("\n📚 Loading and preprocessing dataset...")
    from datasets import get_dataset_split_names, concatenate_datasets
else:
    print("\n📚 Loading and preprocessing dataset...")
    from datasets import get_dataset_split_names, concatenate_datasets

if not preprocess_exists or FORCE_PREPROCESS:
    # Only do preprocessing if needed
    splits = get_dataset_split_names(DATASET_NAME)
    print(f"Found {len(splits)} splits")

    all_datasets = []
    for split_name in splits:
        split_data = load_dataset(DATASET_NAME, split=split_name)
        all_datasets.append(split_data)

    dataset = concatenate_datasets(all_datasets)
    print(f"Total samples: {len(dataset)}")

    def convert_to_text(example):
        try:
            prompt_input = example.get("prompt_input", "") or ""
            user_input = example.get("input", "") or ""
            output = example.get("output", "") or ""

            if not user_input or not output:
                return {"text": ""}

            if prompt_input and prompt_input.strip():
                full_input = f"{prompt_input}\n{user_input}"
            else:
                full_input = user_input

            messages = [
                {"role": "user", "content": full_input},
                {"role": "assistant", "content": output}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            return {"text": text}
        except:
            return {"text": ""}

    print("Applying chat template...")
    dataset = dataset.map(convert_to_text, num_proc=4, desc="Formatting")

    print("Filtering invalid samples...")
    original_size = len(dataset)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    print(f"✅ Kept {len(dataset)}/{original_size} valid samples")

    # Filter samples by sequence length DURING preprocessing (OPTIONAL)
    if CHECK_SEQ_LENGTH:
        print(f"\n📏 Checking sequence lengths (max: {MAX_SEQ_LENGTH})...")
        print(f"   (This may take a while for large datasets - set CHECK_SEQ_LENGTH=false to skip)")

        def check_length(example):
            """Check if sample fits within MAX_SEQ_LENGTH"""
            text = example.get("text", "")
            if not text:
                return {"length": 0, "text": text}

            # Tokenize to get actual length
            tokens = tokenizer(text, truncation=False, add_special_tokens=True)
            length = len(tokens["input_ids"])

            return {"length": length, "text": text}

        # Add length field to dataset (parallel processing for speed)
        dataset = dataset.map(check_length, num_proc=4, desc="Checking lengths")

        # Show length statistics
        lengths = dataset["length"]
        total_samples = len(dataset)
        too_long = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)
        max_length = max(lengths) if lengths else 0
        avg_length = sum(lengths) / len(lengths) if lengths else 0

        print(f"  Total samples: {total_samples}")
        print(f"  Max length in dataset: {max_length} tokens")
        print(f"  Average length: {avg_length:.0f} tokens")
        print(f"  Samples > {MAX_SEQ_LENGTH}: {too_long} ({too_long/total_samples*100:.1f}%)")

        # Filter out samples that are too long
        if too_long > 0:
            print(f"\n⚠️  Filtering out {too_long} samples exceeding MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
            dataset = dataset.filter(lambda x: x["length"] <= MAX_SEQ_LENGTH, desc="Filtering by length")
            print(f"✅ Kept {len(dataset)}/{total_samples} samples")

            if too_long > total_samples * 0.1:  # If >10% filtered
                print(f"\n💡 TIP: {too_long} samples ({too_long/total_samples*100:.1f}%) were skipped.")
                print(f"   To include them, increase MAX_SEQ_LENGTH to at least {max_length}")

        # Remove the length field (not needed for training)
        dataset = dataset.remove_columns(["length"])
    else:
        print(f"\n⚡ Skipping sequence length checking (CHECK_SEQ_LENGTH=false)")
        print(f"   Trainer will handle truncation automatically at MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
        print(f"\n   ℹ️  If training fails with length/token errors:")
        print(f"   • Set CHECK_SEQ_LENGTH=true to filter out long samples")
        print(f"   • Or increase MAX_SEQ_LENGTH in .env")

    print(f"\n💾 Saving to {PREPROCESSED_DATASET_PATH}")
    dataset.save_to_disk(PREPROCESSED_DATASET_PATH)
    print("✅ Saved for future use")

# Limit dataset for testing if requested
if DATASET_MAX_SAMPLES > 0 and len(dataset) > DATASET_MAX_SAMPLES:
    print(f"\n⚡ TEST MODE: Limiting dataset to {DATASET_MAX_SAMPLES} samples (from {len(dataset)})")
    dataset = dataset.select(range(DATASET_MAX_SAMPLES))
    print(f"✅ Using {len(dataset)} samples for quick testing")

print("\n🚀 Setting up trainer...")

# Build training arguments based on MAX_STEPS
training_args = {
    "output_dir": OUTPUT_DIR,
    "logging_steps": LOGGING_STEPS,
    "per_device_train_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "learning_rate": LEARNING_RATE,
    "fp16": not is_bfloat16_supported(),
    "bf16": is_bfloat16_supported(),
    "optim": OPTIM,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "max_grad_norm": MAX_GRAD_NORM,
    "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "report_to": "wandb" if WANDB_ENABLED else "none",
    "run_name": WANDB_RUN_NAME if WANDB_ENABLED else None,
    "seed": SEED,
}

# Configure checkpoint saving
if SAVE_ONLY_FINAL:
    # Disable intermediate checkpoints - only save final model
    training_args["save_strategy"] = "no"
else:
    # Save intermediate checkpoints
    training_args["save_steps"] = SAVE_STEPS
    training_args["save_total_limit"] = SAVE_TOTAL_LIMIT
    training_args["save_strategy"] = "steps"

# Set either max_steps OR num_train_epochs (not both)
if MAX_STEPS > 0:
    training_args["max_steps"] = MAX_STEPS
else:
    training_args["num_train_epochs"] = NUM_TRAIN_EPOCHS

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=4,
    packing=PACKING,
    args=TrainingArguments(**training_args),
)

print("\n📊 Model info:")
model.print_trainable_parameters()

print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60 + "\n")

import time
start_time = time.time()

try:
    trainer_output = trainer.train()
    training_time = time.time() - start_time
except (RuntimeError, ValueError) as e:
    error_msg = str(e).lower()

    # Check for sequence length / token related errors
    if any(keyword in error_msg for keyword in ['length', 'token', 'sequence', 'size', 'exceed', 'overflow']):
        print("\n" + "="*60)
        print("❌ TRAINING FAILED - Sequence Length Error")
        print("="*60)
        print(f"\nError: {e}")
        print("\n💡 SOLUTIONS:")
        print("\n1. Enable length filtering (recommended for first run):")
        print("   • Set CHECK_SEQ_LENGTH=true in .env")
        print("   • Set FORCE_PREPROCESS=true in .env")
        print("   • Re-run: python train.py")
        print("   • This will filter out samples exceeding MAX_SEQ_LENGTH")
        print("\n2. Increase context length:")
        print(f"   • Current: MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
        print("   • Increase MAX_SEQ_LENGTH in .env (e.g., 4096, 8192)")
        print("   • Note: Higher values require more VRAM")
        print("\n3. Check your dataset:")
        print("   • Some samples may be extremely long")
        print("   • Consider filtering or truncating in your dataset")
        print("\n" + "="*60)
        exit(1)
    else:
        # Re-raise if it's a different error
        raise

print("\n" + "="*60)
print("💾 SAVING MODEL")
print("="*60)

# Save LoRA adapters (small, for continued training)
print(f"\n💾 Saving LoRA adapters to: {LORA_DIR}")
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"✅ LoRA adapters saved!")
print(f"\n💡 To create merged models and other formats, run: python build.py")

# Save training metrics for README generation
import json

# Get VRAM usage for metrics
max_vram_bytes = None
try:
    import torch
    if torch.cuda.is_available():
        max_vram_bytes = torch.cuda.max_memory_allocated()
except:
    pass

# Get dataset size
dataset_samples = len(dataset) if 'dataset' in locals() else None
total_steps = trainer_output.global_step if hasattr(trainer_output, 'global_step') else None

metrics_path = os.path.join(LORA_DIR, "training_metrics.json")
metrics = {
    "training_time_seconds": training_time,
    "training_time_minutes": training_time / 60,
    "final_loss": trainer_output.metrics.get('train_loss', None) if hasattr(trainer_output, 'metrics') else None,
    "model_name": LORA_BASE_MODEL,
    "dataset_name": DATASET_NAME,
    "dataset_samples": dataset_samples,
    "total_steps": total_steps,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "max_seq_length": MAX_SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "max_steps": MAX_STEPS,
    "max_vram_bytes": max_vram_bytes,
    "max_vram_gb": max_vram_bytes / (1024**3) if max_vram_bytes else None,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ All models saved!")

# Generate proper README files with actual configuration
print(f"\n📝 Generating README documentation...")
try:
    import subprocess
    result = subprocess.run(
        ["python", "generate_readme.py"],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode == 0:
        print(f"✅ README files generated")
    else:
        print(f"⚠️  README generation failed (non-critical)")
except Exception as e:
    print(f"⚠️  Could not generate READMEs (non-critical): {e}")

# Clean up intermediate checkpoints to save disk space
checkpoint_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
if checkpoint_dirs:
    print(f"\n🧹 Cleaning up {len(checkpoint_dirs)} intermediate checkpoints...")
    total_saved = 0
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = os.path.join(OUTPUT_DIR, checkpoint_dir)
        import shutil
        # Get size before deletion
        checkpoint_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(checkpoint_path)
            for filename in filenames
        )
        total_saved += checkpoint_size
        shutil.rmtree(checkpoint_path)

    # Convert bytes to MB
    total_saved_mb = total_saved / (1024 * 1024)
    print(f"✅ Freed {total_saved_mb:.1f} MB of disk space")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE")
print("="*60)

# Get VRAM usage if available
max_vram_gb = "N/A"
try:
    import torch
    if torch.cuda.is_available():
        max_vram_bytes = torch.cuda.max_memory_allocated()
        max_vram_gb = f"{max_vram_bytes / (1024**3):.2f} GB"
except:
    pass

# Get dataset info
dataset_samples = len(dataset) if 'dataset' in locals() else 'N/A'
total_steps = trainer_output.global_step if hasattr(trainer_output, 'global_step') else 'N/A'

# Show training summary
print(f"\n📊 Training Summary:")
print(f"\n🔧 Training Configuration:")
print(f"  Base Model: {LORA_BASE_MODEL}")
print(f"  Output Name: {output_model_name}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Samples Trained: {dataset_samples}")
if MAX_STEPS > 0:
    print(f"  Training Mode: {MAX_STEPS} steps (early stop)")
else:
    print(f"  Training Mode: {NUM_TRAIN_EPOCHS} epoch(s)")
print(f"  Total Steps: {total_steps}")
print(f"  Batch Size: {BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} (effective)")
print(f"  Max Seq Length: {MAX_SEQ_LENGTH}")
print(f"  LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}")

print(f"\n⏱️  Performance:")
print(f"  Training Time: {training_time/60:.1f} minutes ({training_time:.0f} seconds)")
if hasattr(trainer_output, 'metrics'):
    final_loss = trainer_output.metrics.get('train_loss', 'N/A')
    print(f"  Final Loss: {final_loss}")
print(f"  Max VRAM Used: {max_vram_gb}")

print(f"\n💾 Output:")
# Show output info
from pathlib import Path
lora_size = sum(f.stat().st_size for f in Path(LORA_DIR).rglob('*') if f.is_file())
lora_size_mb = lora_size / (1024 * 1024)
print(f"  LoRA Adapters: {lora_size_mb:.1f} MB")
print(f"  Location: {LORA_DIR}")

# Show next steps
print(f"\n💡 Next Steps:")
print(f"  Run: python build.py")
print(f"\n  This will create:")
print(f"    • merged_16bit/ (merged model + Modelfile)")

# Show what base model will be used for merging
INFERENCE_BASE_MODEL = os.getenv("INFERENCE_BASE_MODEL", "")
if INFERENCE_BASE_MODEL:
    print(f"      Using INFERENCE_BASE_MODEL: {INFERENCE_BASE_MODEL}")
    print(f"      Quality: 16-bit (requires 15-30GB VRAM during build)")
else:
    print(f"      Using LORA_BASE_MODEL: {LORA_BASE_MODEL}")
    print(f"      Quality: Same as training (4-bit)")

# Show additional output formats
OUTPUT_FORMATS = os.getenv("OUTPUT_FORMATS", "")
if OUTPUT_FORMATS:
    format_names = {
        "merged_4bit": "4-bit safetensors",
        "gguf_f16": "GGUF 16-bit",
        "gguf_q8_0": "GGUF Q8_0",
        "gguf_q5_k_m": "GGUF Q5_K_M",
        "gguf_q4_k_m": "GGUF Q4_K_M",
        "gguf_q2_k": "GGUF Q2_K",
    }
    formats = [f.strip() for f in OUTPUT_FORMATS.split(",") if f.strip()]
    if formats:
        print(f"\n    • Additional formats:")
        for fmt in formats:
            friendly_name = format_names.get(fmt, fmt)
            print(f"      - {friendly_name}")

        # Estimate build time
        gguf_count = sum(1 for f in formats if f.startswith("gguf_"))
        if gguf_count > 0:
            est_time = 3 + (gguf_count * 1.5)  # ~3 min merge + ~1.5 min per GGUF
            print(f"\n  ⏱️  Estimated build time: ~{est_time:.0f} minutes")
        else:
            print(f"\n  ⏱️  Estimated build time: ~3 minutes")
else:
    print(f"\n  💡 Tip: Set OUTPUT_FORMATS in .env to create GGUF/other formats")

print("\n" + "="*60 + "\n")
