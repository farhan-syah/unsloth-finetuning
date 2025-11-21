"""
Preprocess dataset and provide smart configuration recommendations

This script:
1. Loads and preprocesses datasets (applies chat template, filters invalid samples)
2. Analyzes dataset size and sequence lengths
3. Detects available GPU memory
4. Provides smart recommendations for BATCH_SIZE, MAX_STEPS, NUM_TRAIN_EPOCHS

Run this before training to get optimal configuration suggestions.
"""

import torch
import os
import json
from datetime import datetime
from unsloth import FastLanguageModel
from datasets import load_dataset, get_dataset_split_names, concatenate_datasets
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

# Helper functions
def get_bool_env(key, default=False):
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')

def get_int_env(key, default):
    return int(os.getenv(key, str(default)))

def get_float_env(key, default):
    return float(os.getenv(key, str(default)))

# Configuration
LORA_BASE_MODEL = os.getenv("LORA_BASE_MODEL", "unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
DATASET_NAME = os.getenv("DATASET_NAME", "yahma/alpaca-cleaned")
MAX_SEQ_LENGTH = get_int_env("MAX_SEQ_LENGTH", 4096)
PREPROCESSED_DATA_DIR = os.getenv("PREPROCESSED_DATA_DIR", "./data/preprocessed")
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME", "auto")
FORCE_PREPROCESS = get_bool_env("FORCE_PREPROCESS", False)
CHECK_SEQ_LENGTH = get_bool_env("CHECK_SEQ_LENGTH", True)  # Default true for analysis

# Current training config (for comparison)
CURRENT_BATCH_SIZE = get_int_env("BATCH_SIZE", 2)
CURRENT_GRAD_ACCUM = get_int_env("GRADIENT_ACCUMULATION_STEPS", 4)
CURRENT_MAX_STEPS = get_int_env("MAX_STEPS", 0)
CURRENT_NUM_EPOCHS = get_int_env("NUM_TRAIN_EPOCHS", 1)

# Generate names
dataset_short_name = DATASET_NAME.split("/")[-1].lower().replace("_", "-")
if OUTPUT_MODEL_NAME == "auto" or not OUTPUT_MODEL_NAME:
    model_base = LORA_BASE_MODEL.split("/")[-1].replace("-unsloth-bnb-4bit", "").replace("-unsloth", "")
    output_model_name = f"{model_base}-{dataset_short_name}"
else:
    output_model_name = OUTPUT_MODEL_NAME

PREPROCESSED_DATASET_PATH = os.path.join(PREPROCESSED_DATA_DIR, dataset_short_name)

# Create directories
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print("\n" + "="*70)
print("📊 DATASET PREPROCESSING & CONFIGURATION ANALYSIS")
print("="*70)

print(f"\n📚 Dataset: {DATASET_NAME}")
print(f"🤖 Model: {LORA_BASE_MODEL}")
print(f"📏 Max Sequence Length: {MAX_SEQ_LENGTH}")

# ============================================
# STEP 1: GPU Memory Detection
# ============================================
print("\n" + "-"*70)
print("🔍 STEP 1: GPU Memory Analysis")
print("-"*70)

gpu_available = torch.cuda.is_available()
if gpu_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = gpu_memory_total / (1024**3)

    print(f"✅ GPU Detected: {gpu_name}")
    print(f"   Total VRAM: {gpu_memory_gb:.1f} GB")

    # Estimate model size based on model name
    model_size_estimate = None
    if "1.7B" in LORA_BASE_MODEL or "1.5B" in LORA_BASE_MODEL:
        model_size_estimate = "1.5-2GB VRAM (4-bit quantized)"
    elif "2B" in LORA_BASE_MODEL:
        model_size_estimate = "2-3GB VRAM (4-bit quantized)"
    elif "4B" in LORA_BASE_MODEL or "3B" in LORA_BASE_MODEL:
        model_size_estimate = "4-5GB VRAM (4-bit quantized)"
    elif "7B" in LORA_BASE_MODEL:
        model_size_estimate = "7-9GB VRAM (4-bit quantized)"
    elif "13B" in LORA_BASE_MODEL:
        model_size_estimate = "13-16GB VRAM (4-bit quantized)"

    if model_size_estimate:
        print(f"   Estimated Model Size: {model_size_estimate}")

    # Available memory for training
    available_for_batch = gpu_memory_gb - 2  # Reserve 2GB for model overhead
    print(f"   Available for Batching: ~{available_for_batch:.1f} GB")
else:
    print("⚠️  No GPU detected - training will be very slow or fail")
    gpu_memory_gb = 0

# ============================================
# STEP 2: Load Model to Get Exact Size
# ============================================
print("\n" + "-"*70)
print("🔍 STEP 2: Loading Model (to measure actual VRAM usage)")
print("-"*70)

try:
    print(f"Loading {LORA_BASE_MODEL}...")

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        cache_dir=CACHE_DIR
    )

    # Measure actual VRAM used
    if torch.cuda.is_available():
        model_vram_bytes = torch.cuda.max_memory_allocated()
        model_vram_gb = model_vram_bytes / (1024**3)
        print(f"✅ Model loaded successfully")
        print(f"   Actual VRAM used: {model_vram_gb:.2f} GB")

        # Available for training batches
        available_vram = gpu_memory_gb - model_vram_gb - 1  # Reserve 1GB safety margin
        print(f"   Available for training: {available_vram:.2f} GB")
    else:
        model_vram_gb = 0
        available_vram = 0
        print(f"✅ Model loaded (CPU mode)")

except Exception as e:
    print(f"⚠️  Could not load model: {e}")
    model_vram_gb = 2  # Rough estimate
    available_vram = max(0, gpu_memory_gb - 3)
    tokenizer = None

# ============================================
# STEP 3: Dataset Loading & Preprocessing
# ============================================
print("\n" + "-"*70)
print("🔍 STEP 3: Dataset Preprocessing")
print("-"*70)

preprocess_exists = os.path.exists(PREPROCESSED_DATASET_PATH)
metadata_path = os.path.join(PREPROCESSED_DATASET_PATH, "preprocessing_metadata.json")

if preprocess_exists and not FORCE_PREPROCESS:
    print(f"📂 Loading preprocessed dataset from {PREPROCESSED_DATASET_PATH}")
    from datasets import load_from_disk
    dataset = load_from_disk(PREPROCESSED_DATASET_PATH)
    print(f"✅ Loaded {len(dataset)} samples (already preprocessed)")
    print(f"   To reprocess, set FORCE_PREPROCESS=true in .env")

    # Load metadata if available
    dataset_metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            dataset_metadata = json.load(f)
            print(f"\n   📊 Preprocessing Info (from metadata):")
            print(f"      Original samples: {dataset_metadata.get('original_samples', 'unknown')}")
            print(f"      Valid samples: {dataset_metadata.get('valid_samples', len(dataset))}")
            print(f"      Max length: {dataset_metadata.get('max_length', 'unknown')} tokens")
            print(f"      Avg length: {dataset_metadata.get('avg_length', 'unknown')} tokens")
            print(f"      Samples filtered: {dataset_metadata.get('filtered_count', 0)}")

elif preprocess_exists and FORCE_PREPROCESS:
    print(f"⚠️  Preprocessed dataset exists but FORCE_PREPROCESS=true")
    print(f"   Reprocessing...")
    import shutil
    shutil.rmtree(PREPROCESSED_DATASET_PATH)
    preprocess_exists = False

if not preprocess_exists:
    print(f"📚 Loading dataset: {DATASET_NAME}")

    # Load all splits
    splits = get_dataset_split_names(DATASET_NAME)
    print(f"   Found {len(splits)} splits: {splits}")

    all_datasets = []
    for split_name in splits:
        split_data = load_dataset(DATASET_NAME, split=split_name)
        all_datasets.append(split_data)
        print(f"   Loaded {split_name}: {len(split_data)} samples")

    dataset = concatenate_datasets(all_datasets)
    print(f"   Total samples (all splits): {len(dataset)}")

    # Convert to chat template format
    if tokenizer:
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

        print("\n   Applying chat template...")
        dataset = dataset.map(convert_to_text, num_proc=4, desc="Formatting")

        print("   Filtering invalid samples...")
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: len(x["text"]) > 0)
        filtered_out = original_size - len(dataset)
        print(f"   ✅ Kept {len(dataset)}/{original_size} valid samples ({filtered_out} filtered)")

    # Check sequence lengths using HYBRID APPROACH
    # Step 1: Fast character-based estimation (filters ~95% instantly)
    # Step 2: Only tokenize borderline cases for exact count
    if CHECK_SEQ_LENGTH and tokenizer:
        print(f"\n   📏 Analyzing sequence lengths (max: {MAX_SEQ_LENGTH})...")
        print(f"      Using hybrid approach: char estimation → selective tokenization")

        # Step 1: Fast character-based filter
        print(f"\n      [1/3] Fast character-length pre-filter...")

        def estimate_tokens_fast(example):
            """Estimate tokens using character count (avg 4 chars/token for GPT-style)"""
            text = example.get("text", "")
            if not text:
                return {"estimated_length": 0, "needs_tokenization": False, "text": text}

            # Estimate: ~4 characters per token (conservative for GPT/LLaMA tokenizers)
            estimated_tokens = len(text) // 4

            # Three zones:
            # 1. Definitely safe: < 80% of limit → skip tokenization
            # 2. Borderline: 80%-120% of limit → needs tokenization
            # 3. Definitely too long: > 120% of limit → skip tokenization, will be filtered
            safe_threshold = int(MAX_SEQ_LENGTH * 0.8)
            borderline_threshold = int(MAX_SEQ_LENGTH * 1.2)

            if estimated_tokens < safe_threshold:
                # Definitely safe, no need to tokenize
                return {"estimated_length": estimated_tokens, "needs_tokenization": False, "text": text}
            elif estimated_tokens > borderline_threshold:
                # Definitely too long, no need to tokenize
                return {"estimated_length": estimated_tokens, "needs_tokenization": False, "text": text}
            else:
                # Borderline - need exact count
                return {"estimated_length": estimated_tokens, "needs_tokenization": True, "text": text}

        dataset = dataset.map(estimate_tokens_fast, num_proc=4, desc="Char estimation")

        # Count how many need tokenization
        borderline_count = sum(dataset["needs_tokenization"])
        total_samples = len(dataset)

        print(f"      ✅ Pre-filtered {total_samples} samples")
        print(f"         • {total_samples - borderline_count} samples clearly safe/unsafe (skipped tokenization)")
        print(f"         • {borderline_count} borderline samples need exact tokenization ({borderline_count/total_samples*100:.1f}%)")

        # Step 2: Only tokenize borderline cases
        if borderline_count > 0:
            print(f"\n      [2/3] Tokenizing {borderline_count} borderline samples for exact count...")

            def tokenize_if_needed(example):
                """Only tokenize if needed (borderline cases)"""
                if not example["needs_tokenization"]:
                    # Use estimate
                    return {"length": example["estimated_length"], "text": example["text"]}

                # Tokenize for exact count
                text = example.get("text", "")
                if not text:
                    return {"length": 0, "text": text}

                tokens = tokenizer(text, truncation=False, add_special_tokens=True)
                return {"length": len(tokens["input_ids"]), "text": text}

            dataset = dataset.map(tokenize_if_needed, num_proc=4, desc="Exact tokenization")
            # Remove temporary columns
            if "estimated_length" in dataset.column_names:
                dataset = dataset.remove_columns(["estimated_length", "needs_tokenization"])
        else:
            print(f"\n      [2/3] No borderline samples - using character estimates only")
            # Just rename estimated_length to length
            dataset = dataset.rename_column("estimated_length", "length")
            dataset = dataset.remove_columns(["needs_tokenization"])

        # Step 3: Analyze and filter
        print(f"\n      [3/3] Analyzing results...")
        lengths = dataset["length"]
        too_long = sum(1 for l in lengths if l > MAX_SEQ_LENGTH)
        max_length = max(lengths) if lengths else 0
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        median_length = sorted(lengths)[len(lengths)//2] if lengths else 0

        print(f"\n   📊 Sequence Length Statistics:")
        print(f"      Total samples: {total_samples}")
        print(f"      Max length: {max_length} tokens")
        print(f"      Average length: {avg_length:.0f} tokens")
        print(f"      Median length: {median_length} tokens")
        print(f"      Samples > {MAX_SEQ_LENGTH}: {too_long} ({too_long/total_samples*100:.1f}%)")
        print(f"\n   ⚡ Performance: Only tokenized {borderline_count}/{total_samples} samples ({borderline_count/total_samples*100:.1f}%)")

        # Filter out too-long samples
        if too_long > 0:
            print(f"\n   ⚠️  Filtering out {too_long} samples exceeding MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}")
            dataset = dataset.filter(lambda x: x["length"] <= MAX_SEQ_LENGTH, desc="Filtering")
            print(f"   ✅ Final dataset: {len(dataset)} samples")

            if too_long > total_samples * 0.2:  # If >20% filtered
                print(f"\n   💡 TIP: {too_long/total_samples*100:.1f}% of samples were skipped.")
                print(f"      Consider increasing MAX_SEQ_LENGTH to {max_length} to use all data")

        # Remove length field
        dataset = dataset.remove_columns(["length"])

        # Save metadata for future reference
        dataset_metadata = {
            "original_samples": original_size,
            "valid_samples": len(dataset),
            "max_length": max_length,
            "avg_length": avg_length,
            "median_length": median_length,
            "filtered_count": too_long,
            "max_seq_length": MAX_SEQ_LENGTH,
            "dataset_name": DATASET_NAME,
            "preprocessing_date": str(datetime.now())
        }
    else:
        # No sequence length checking, save basic metadata
        dataset_metadata = {
            "original_samples": original_size,
            "valid_samples": len(dataset),
            "max_length": "not_checked",
            "avg_length": "not_checked",
            "filtered_count": 0,
            "max_seq_length": MAX_SEQ_LENGTH,
            "dataset_name": DATASET_NAME,
            "preprocessing_date": str(datetime.now())
        }

    # Save preprocessed dataset
    print(f"\n   💾 Saving to {PREPROCESSED_DATASET_PATH}")
    dataset.save_to_disk(PREPROCESSED_DATASET_PATH)

    # Save metadata
    metadata_path = os.path.join(PREPROCESSED_DATASET_PATH, "preprocessing_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)

    print(f"   ✅ Saved for future use")

# ============================================
# STEP 4: Smart Configuration Recommendations
# ============================================
print("\n" + "="*70)
print("💡 SMART CONFIGURATION RECOMMENDATIONS")
print("="*70)

final_dataset_size = len(dataset)
print(f"\n📊 Final Dataset Size: {final_dataset_size} samples")

# Current config
current_effective_batch = CURRENT_BATCH_SIZE * CURRENT_GRAD_ACCUM
print(f"\n📋 Current Configuration:")
print(f"   BATCH_SIZE: {CURRENT_BATCH_SIZE}")
print(f"   GRADIENT_ACCUMULATION_STEPS: {CURRENT_GRAD_ACCUM}")
print(f"   Effective Batch Size: {current_effective_batch}")
print(f"   MAX_STEPS: {CURRENT_MAX_STEPS} (0 = use epochs)")
print(f"   NUM_TRAIN_EPOCHS: {CURRENT_NUM_EPOCHS}")

# Calculate current training plan
if CURRENT_MAX_STEPS > 0:
    current_total_steps = CURRENT_MAX_STEPS
    current_samples_seen = CURRENT_MAX_STEPS * current_effective_batch
    current_epochs = current_samples_seen / final_dataset_size
    print(f"   → Will train for {current_total_steps} steps")
    print(f"   → ~{current_epochs:.2f} epochs ({current_samples_seen} samples seen)")
else:
    current_steps_per_epoch = math.ceil(final_dataset_size / current_effective_batch)
    current_total_steps = current_steps_per_epoch * CURRENT_NUM_EPOCHS
    print(f"   → {current_steps_per_epoch} steps/epoch × {CURRENT_NUM_EPOCHS} epochs = {current_total_steps} total steps")

# ============================================
# Recommendation Logic
# ============================================
print(f"\n" + "-"*70)
print("🎯 RECOMMENDED CONFIGURATION")
print("-"*70)

# Target: 1-3 epochs (prefer 1-2 for most cases)
TARGET_MIN_EPOCHS = 1
TARGET_MAX_EPOCHS = 3
IDEAL_EPOCHS = 1  # Most models benefit from 1 full pass

# Recommended effective batch size based on dataset size
if final_dataset_size < 500:
    recommended_effective_batch = 4  # Small dataset, small batches
elif final_dataset_size < 2000:
    recommended_effective_batch = 8
elif final_dataset_size < 10000:
    recommended_effective_batch = 16
else:
    recommended_effective_batch = 32  # Large dataset, larger batches

# Adjust based on available VRAM
if gpu_available and available_vram < 4:
    # Low VRAM: use smaller batch size with more accumulation
    recommended_batch_size = 1
    recommended_grad_accum = recommended_effective_batch
elif gpu_available and available_vram < 8:
    # Medium VRAM
    recommended_batch_size = 2
    recommended_grad_accum = recommended_effective_batch // 2
else:
    # High VRAM or CPU
    recommended_batch_size = 4
    recommended_grad_accum = recommended_effective_batch // 4

# Ensure minimum grad_accum of 1
recommended_grad_accum = max(1, recommended_grad_accum)

# Calculate steps for target epochs
steps_per_epoch = math.ceil(final_dataset_size / recommended_effective_batch)
recommended_max_steps_1_epoch = steps_per_epoch
recommended_max_steps_2_epochs = steps_per_epoch * 2
recommended_max_steps_3_epochs = steps_per_epoch * 3

print(f"\n📊 Why limit to 1-3 epochs?")
print(f"   • Modern LLMs with LoRA typically need only 1-2 full passes through data")
print(f"   • More epochs often lead to overfitting (model memorizes instead of learns)")
print(f"   • For {final_dataset_size} samples, 1 epoch = {steps_per_epoch} training steps")
print(f"   • If loss plateaus early, you can stop training (no need to complete epoch)")

print(f"\n💡 Recommended Settings for .env:")
print(f"   BATCH_SIZE={recommended_batch_size}")
print(f"   GRADIENT_ACCUMULATION_STEPS={recommended_grad_accum}")
print(f"   # Effective batch size: {recommended_effective_batch}")

print(f"\n   # Choose ONE of these MAX_STEPS options:")
print(f"   MAX_STEPS={recommended_max_steps_1_epoch}  # 1 epoch (recommended)")
print(f"   # MAX_STEPS={recommended_max_steps_2_epochs}  # 2 epochs (if 1 epoch underfits)")
print(f"   # MAX_STEPS={recommended_max_steps_3_epochs}  # 3 epochs (maximum recommended)")
print(f"   # MAX_STEPS=0  # Use NUM_TRAIN_EPOCHS instead (less precise)")

print(f"\n   NUM_TRAIN_EPOCHS=1  # Only used if MAX_STEPS=0")

# Explain the reasoning
print(f"\n📖 Explanation:")
print(f"   Dataset: {final_dataset_size} samples")
print(f"   Effective Batch: {recommended_effective_batch} samples/step")
print(f"   Steps per epoch: {steps_per_epoch}")
print(f"")
print(f"   1 epoch  = {recommended_max_steps_1_epoch} steps × {recommended_effective_batch} samples = {recommended_max_steps_1_epoch * recommended_effective_batch} samples seen")
print(f"   2 epochs = {recommended_max_steps_2_epochs} steps × {recommended_effective_batch} samples = {recommended_max_steps_2_epochs * recommended_effective_batch} samples seen")
print(f"   3 epochs = {recommended_max_steps_3_epochs} steps × {recommended_effective_batch} samples = {recommended_max_steps_3_epochs * recommended_effective_batch} samples seen")

# VRAM-based recommendations
if gpu_available:
    print(f"\n🎮 GPU Optimization:")
    print(f"   Available VRAM: {available_vram:.1f} GB")
    if available_vram < 4:
        print(f"   • Low VRAM: Using BATCH_SIZE=1 with high accumulation")
        print(f"   • Consider: USE_GRADIENT_CHECKPOINTING=true (saves VRAM)")
    elif available_vram < 8:
        print(f"   • Medium VRAM: Balanced settings")
        print(f"   • Optional: USE_GRADIENT_CHECKPOINTING=false for 20% speed boost")
    else:
        print(f"   • High VRAM: Can increase BATCH_SIZE for faster training")
        print(f"   • Recommended: USE_GRADIENT_CHECKPOINTING=false for maximum speed")

# Warning if current config will train too many epochs
if CURRENT_MAX_STEPS > 0:
    current_epochs_estimate = (CURRENT_MAX_STEPS * current_effective_batch) / final_dataset_size
    if current_epochs_estimate > TARGET_MAX_EPOCHS:
        print(f"\n⚠️  WARNING: Current MAX_STEPS={CURRENT_MAX_STEPS} will train for ~{current_epochs_estimate:.1f} epochs")
        print(f"   This is likely too much and may cause overfitting!")
        print(f"   Recommended: Use MAX_STEPS={recommended_max_steps_1_epoch} for 1 epoch instead")

print("\n" + "="*70)
print("✅ PREPROCESSING COMPLETE")
print("="*70)
print(f"\nNext steps:")
print(f"1. Update your .env file with recommended settings above")
print(f"2. Run: python train.py")
print(f"3. Monitor training loss - stop if it plateaus early")
print(f"\nPreprocessed data saved to: {PREPROCESSED_DATASET_PATH}")
print("="*70 + "\n")
