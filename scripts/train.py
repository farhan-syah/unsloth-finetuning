"""
Fine-tune models using Unsloth with LoRA
Handles ONLY training - creates LoRA adapters
Use build.py to convert to merged/GGUF formats

Configure your dataset and model in .env file
"""

import torch
import os
import json
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

def save_tokenizer_with_template(tokenizer, output_dir, token=False):
    """
    Save tokenizer and preserve chat_template in tokenizer_config.json.

    Fixes issue where tokenizer.save_pretrained() doesn't always preserve
    the chat_template attribute in the config file, causing benchmarks to fail.
    """
    # Save tokenizer normally
    tokenizer.save_pretrained(output_dir, token=token)

    # Ensure chat_template is in tokenizer_config.json
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        config_path = os.path.join(output_dir, "tokenizer_config.json")

        # Load existing config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Add chat_template if missing
        if 'chat_template' not in config:
            config['chat_template'] = tokenizer.chat_template

            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

# Configuration from environment variables
LORA_BASE_MODEL = os.getenv("LORA_BASE_MODEL", "unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME", "auto")
MAX_SEQ_LENGTH = get_int_env("MAX_SEQ_LENGTH", 2048)
LORA_RANK = get_int_env("LORA_RANK", 64)
LORA_ALPHA = get_int_env("LORA_ALPHA", 128)
LORA_DROPOUT = get_float_env("LORA_DROPOUT", 0.0)
USE_RSLORA = get_bool_env("USE_RSLORA", False)
DATASET_NAME = os.getenv("DATASET_NAME", "yahma/alpaca-cleaned")
DATASET_MAX_SAMPLES = get_int_env("DATASET_MAX_SAMPLES", 0)  # 0 = use all
MAX_STEPS = get_int_env("MAX_STEPS", 0)  # 0 = use epochs

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

# Set HuggingFace cache to project directory for consistency
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

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
NUM_TRAIN_EPOCHS = get_float_env("NUM_TRAIN_EPOCHS", 1)  # Allow fractional epochs (0.5, 1.5, etc.)
WARMUP_RATIO = get_float_env("WARMUP_RATIO", 0.1)  # 10% of total steps
WARMUP_STEPS = get_int_env("WARMUP_STEPS", 0)  # Overrides warmup_ratio if > 0
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

# Automatic backup of existing LoRA adapters
lora_exists = os.path.exists(os.path.join(LORA_DIR, "adapter_config.json"))
if lora_exists:
    print(f"\n📦 Existing LoRA adapters detected at: {LORA_DIR}")
    print("   Creating backup before retraining...")

    # Read existing training metrics for backup naming
    metrics_path = os.path.join(LORA_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                old_metrics = json.load(f)

            # Use trained_date if available, otherwise timestamp
            trained_date = old_metrics.get('trained_date', old_metrics.get('timestamp', ''))
            # Convert timestamp format if needed: "2025-11-23 20:22:55" -> "20251123_202255"
            if ' ' in trained_date:
                trained_date = trained_date.replace(' ', '_').replace(':', '').replace('-', '')[:15]

            # Create descriptive backup folder name
            rank = old_metrics.get('lora_rank', 'unknown')
            lr = old_metrics.get('learning_rate', 'unknown')
            final_loss = old_metrics.get('final_loss', 'unknown')
            if isinstance(final_loss, float):
                final_loss = f"{final_loss:.4f}"

            backup_name = f"{trained_date}_rank{rank}_lr{lr}_loss{final_loss}"
        except Exception as e:
            print(f"   ⚠️  Could not read training metrics: {e}")
            from datetime import datetime
            backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        # No metrics file, use current timestamp
        from datetime import datetime
        backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create backup directory
    backup_base = os.path.join(OUTPUT_DIR, "lora_bak")
    os.makedirs(backup_base, exist_ok=True)
    backup_dir = os.path.join(backup_base, backup_name)

    # Check if backup with same name already exists
    if os.path.exists(backup_dir):
        print(f"   ⚠️  Backup already exists: {backup_name}")
        print(f"   Skipping backup (already preserved)")
        print(f"   Removing current lora to proceed with retraining...")
        import shutil
        shutil.rmtree(LORA_DIR)
    else:
        # Move old lora to backup
        import shutil
        shutil.move(LORA_DIR, backup_dir)
        print(f"   ✅ Backed up to: {backup_dir}")
        print(f"   💡 Use scripts/restore_trained_data.py to restore if needed")

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
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth" if USE_GRADIENT_CHECKPOINTING else False,
    random_state=SEED,
    use_rslora=USE_RSLORA,
    loftq_config=None,
)

# Load preprocessed dataset
# NOTE: Run 'python scripts/preprocess.py' first to preprocess your dataset
preprocess_exists = os.path.exists(PREPROCESSED_DATASET_PATH)
metadata_path = os.path.join(PREPROCESSED_DATASET_PATH, "preprocessing_metadata.json")

if not preprocess_exists:
    print("\n" + "="*60)
    print("❌ ERROR: Preprocessed dataset not found!")
    print("="*60)
    print(f"\nExpected location: {PREPROCESSED_DATASET_PATH}")
    print(f"\n🔧 REQUIRED: Run preprocessing first:")
    print(f"   python scripts/preprocess.py")
    print(f"\nThis will:")
    print(f"   • Preprocess and analyze your dataset")
    print(f"   • Provide smart configuration recommendations")
    print(f"   • Save preprocessed data for training")
    print(f"\nAfter preprocessing, run 'python scripts/train.py' again")
    print("="*60 + "\n")
    exit(1)

print(f"\n📂 Loading preprocessed dataset from {PREPROCESSED_DATASET_PATH}")
from datasets import load_from_disk
dataset = load_from_disk(PREPROCESSED_DATASET_PATH)

# Load and show metadata
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✅ Loaded {len(dataset)} samples")
    print(f"   Dataset: {metadata.get('dataset_name', DATASET_NAME)}")
    print(f"   Max length: {metadata.get('max_length', 'unknown')} tokens")
    print(f"   Avg length: {metadata.get('avg_length', 'unknown')} tokens")
    if metadata.get('filtered_count', 0) > 0:
        print(f"   Filtered out: {metadata['filtered_count']} samples (exceeded MAX_SEQ_LENGTH)")
else:
    print(f"✅ Loaded {len(dataset)} samples")
    print(f"   ⚠️  No metadata found - run 'python scripts/preprocess.py' to regenerate with stats")

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

# Add warmup configuration (warmup_steps overrides warmup_ratio if set)
if WARMUP_STEPS > 0:
    training_args["warmup_steps"] = WARMUP_STEPS
else:
    training_args["warmup_ratio"] = WARMUP_RATIO

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
        print("   • Re-run: python scripts/train.py")
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
# Save with token=False to avoid unnecessary network calls
# The base model config is already in local cache
model.save_pretrained(LORA_DIR, token=False)
save_tokenizer_with_template(tokenizer, LORA_DIR, token=False)
print(f"✅ LoRA adapters saved!")
print(f"\n💡 To create merged models and other formats, run: python scripts/build.py")

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

# Save loss history as separate CSV file
loss_history_path = os.path.join(LORA_DIR, "loss_history.csv")
if hasattr(trainer.state, 'log_history'):
    import csv
    with open(loss_history_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'epoch', 'loss', 'learning_rate', 'grad_norm'])

        last_logged_step = 0
        for entry in trainer.state.log_history:
            if 'loss' in entry:
                writer.writerow([
                    entry.get('step', ''),
                    entry.get('epoch', ''),
                    entry.get('loss', ''),
                    entry.get('learning_rate', ''),
                    entry.get('grad_norm', '')
                ])
                last_logged_step = entry.get('step', 0)

        # Append final step if not already logged (happens when total_steps % logging_steps != 0)
        if total_steps and last_logged_step < total_steps:
            # Find the final training entry in log_history
            final_entry = None
            for entry in reversed(trainer.state.log_history):
                if 'loss' in entry:
                    final_entry = entry
                    break

            if final_entry:
                # Write the final step
                writer.writerow([
                    total_steps,  # Use actual final step
                    final_entry.get('epoch', ''),
                    final_entry.get('loss', ''),
                    final_entry.get('learning_rate', ''),
                    final_entry.get('grad_norm', '')
                ])

metrics_path = os.path.join(LORA_DIR, "training_metrics.json")

# Calculate actual warmup steps used
actual_warmup_steps = None
if WARMUP_STEPS > 0:
    actual_warmup_steps = WARMUP_STEPS
elif total_steps:
    actual_warmup_steps = int(total_steps * WARMUP_RATIO)

# Get final loss from loss_history.csv (more accurate than trainer metrics)
final_loss = None
loss_history_path = os.path.join(LORA_DIR, "loss_history.csv")
if os.path.exists(loss_history_path):
    try:
        import csv
        with open(loss_history_path, 'r') as f:
            # Read all rows and get the last one
            rows = list(csv.DictReader(f))
            if rows:
                final_loss = float(rows[-1]['loss'])
    except Exception as e:
        print(f"⚠️  Could not read final loss from CSV: {e}")
        # Fallback to trainer metrics
        final_loss = trainer_output.metrics.get('train_loss', None) if hasattr(trainer_output, 'metrics') else None
else:
    # Fallback to trainer metrics if CSV doesn't exist
    final_loss = trainer_output.metrics.get('train_loss', None) if hasattr(trainer_output, 'metrics') else None

metrics = {
    "training_time_seconds": training_time,
    "training_time_minutes": training_time / 60,
    "final_loss": final_loss,
    "model_name": LORA_BASE_MODEL,
    "dataset_name": DATASET_NAME,
    "dataset_samples": dataset_samples,
    "dataset_max_samples": DATASET_MAX_SAMPLES,
    "total_steps": total_steps,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "use_rslora": USE_RSLORA,
    "max_seq_length": MAX_SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "warmup_steps": WARMUP_STEPS,
    "actual_warmup_steps": actual_warmup_steps,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "max_steps": MAX_STEPS,
    "packing": PACKING,
    "max_grad_norm": MAX_GRAD_NORM,
    "optimizer": OPTIM,
    "use_gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "seed": SEED,
    "max_vram_bytes": max_vram_bytes,
    "max_vram_gb": max_vram_bytes / (1024**3) if max_vram_bytes else None,
    "trained_date": time.strftime("%Y%m%d_%H%M%S"),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ All models saved!")

# Clean up LoRA directory - remove redundant files
print(f"\n🧹 Cleaning up LoRA directory...")
try:
    from cleanup_utils import cleanup_lora_directory
    removed_count, removed_size = cleanup_lora_directory(LORA_DIR, verbose=True)
    if removed_count > 0:
        print(f"✅ Cleaned up {removed_count} redundant file(s) ({removed_size / (1024 * 1024):.1f} MB)")
    else:
        print(f"✅ LoRA directory already clean")
except Exception as e:
    print(f"⚠️  Could not clean up LoRA directory: {e}")

# Generate proper README files with actual configuration from .env
print(f"\n📝 Generating README documentation...")
try:
    import subprocess
    result = subprocess.run(
        ["python", "generate_readme_train.py"],
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
print(f"  Run: python scripts/build.py")
print(f"\n  This will create:")
print(f"    • merged_16bit/ (merged model + Modelfile)")

# Show what base model will be used for merging
INFERENCE_BASE_MODEL = os.getenv("INFERENCE_BASE_MODEL", "")
if INFERENCE_BASE_MODEL:
    print(f"      Using INFERENCE_BASE_MODEL: {INFERENCE_BASE_MODEL}")
    print(f"      Quality: 16-bit (best quality)")
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
