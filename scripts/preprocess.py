"""
Preprocess dataset and provide smart configuration recommendations

This script:
1. Loads and preprocesses datasets (applies chat template, filters invalid samples)
2. Analyzes dataset size and sequence lengths
3. Detects available GPU memory
4. Provides smart recommendations for batch size, steps, and epochs

Run this before training to get optimal configuration suggestions.
"""

import torch
import os
import json
import argparse
from datetime import datetime
from unsloth import FastLanguageModel, standardize_sharegpt
from unsloth.chat_templates import get_chat_template
from unsloth.ollama_template_mappers import MODEL_TO_OLLAMA_TEMPLATE_MAPPER
from datasets import load_dataset, get_dataset_split_names, concatenate_datasets
import math

# Load configuration
from config_loader import get_config_for_script

# Parse command line arguments
parser = argparse.ArgumentParser(description="Preprocess dataset and analyze configuration")
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to YAML config file (default: training_params.yaml)"
)
args = parser.parse_args()

# Load configuration from YAML and .env
config, env_config = get_config_for_script(args.config, verbose=False)

# Helper functions
def get_template_for_model(model_name):
    """
    Get chat template name for a given model using Unsloth's mapper.
    Returns template name (e.g., "llama-3.1", "qwen2.5") or None if not found.
    """
    if model_name in MODEL_TO_OLLAMA_TEMPLATE_MAPPER:
        return MODEL_TO_OLLAMA_TEMPLATE_MAPPER[model_name]
    return None

# Model and dataset from YAML config
LORA_BASE_MODEL = config.model.base_model
DATASET_NAME = config.dataset.name
DATASET_SUBSET = config.dataset.subset  # Optional subset/configuration name
DATASET_SPLIT = config.dataset.split or "all"  # Which split(s) to use
OUTPUT_MODEL_NAME = config.model.output_name

# Training parameters from YAML config
MAX_SEQ_LENGTH = config.training.data.max_seq_length

# Paths from .env config
PREPROCESSED_DATA_DIR = env_config['preprocessed_data_dir']
CACHE_DIR = env_config['cache_dir']
FORCE_PREPROCESS = env_config['force_preprocess']
CHECK_SEQ_LENGTH = env_config['check_seq_length']

# Set HuggingFace cache to project directory for consistency
# Don't set HF_HOME - it creates a nested cache/ subdirectory
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

# Current training config (for comparison)
CURRENT_BATCH_SIZE = config.training.batch.size
CURRENT_GRAD_ACCUM = config.training.batch.gradient_accumulation_steps
CURRENT_MAX_STEPS = config.training.epochs.max_steps
CURRENT_NUM_EPOCHS = config.training.epochs.num_train_epochs

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

    # Detect if model is a reasoning model by checking chat template
    print(f"\n   🔍 Detecting model capabilities...")
    is_reasoning_model = False
    supports_enable_thinking = False

    # Check if tokenizer has a chat template, if not set one
    if not tokenizer.chat_template:
        print(f"   ⚠️  Model has no chat template (typical for base models)")

        # Try to get template from Unsloth's official mapper first
        template_name = get_template_for_model(LORA_BASE_MODEL)

        if template_name:
            print(f"   Setting '{template_name}' template (from Unsloth mapper)...")
        else:
            # Fallback: Pattern matching for base models or unmapped variants
            # Base models aren't in the mapper, so we use pattern matching
            print(f"   Model not in mapper, using pattern matching...")
            model_name_lower = LORA_BASE_MODEL.lower()
            if "llama-3.2" in model_name_lower or "llama-3.1" in model_name_lower:
                template_name = "llama-3.1"
            elif "llama-3" in model_name_lower or "llama3" in model_name_lower:
                template_name = "llama3"
            elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
                template_name = "llama"
            elif "qwen3" in model_name_lower:
                template_name = "qwen3"
            elif "qwen2.5" in model_name_lower or "qwen-2.5" in model_name_lower:
                template_name = "qwen-2.5"
            elif "qwen2" in model_name_lower or "qwen-2" in model_name_lower:
                template_name = "qwen-2"
            elif "phi" in model_name_lower:
                template_name = "phi-3"
            elif "gemma-3" in model_name_lower or "gemma3" in model_name_lower:
                template_name = "gemma-3"
            elif "gemma-2" in model_name_lower or "gemma2" in model_name_lower:
                template_name = "gemma2"
            elif "gemma" in model_name_lower:
                template_name = "gemma"
            elif "mistral" in model_name_lower:
                template_name = "mistral"
            else:
                template_name = "chatml"  # Generic fallback
            print(f"   Setting '{template_name}' template (pattern matched)...")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=template_name,
        )
        print(f"   ✅ Chat template set")

    if tokenizer.chat_template:
        template_str = str(tokenizer.chat_template).lower()
        # Check for reasoning indicators in template
        reasoning_template_markers = ["<think>", "reasoning", "chain-of-thought", "cot"]
        is_reasoning_model = any(marker in template_str for marker in reasoning_template_markers)

        # Test if model supports enable_thinking parameter (Qwen3, etc.)
        try:
            test_msgs = [{"role": "user", "content": "test"}]
            tokenizer.apply_chat_template(test_msgs, tokenize=False, enable_thinking=False)
            supports_enable_thinking = True
        except (TypeError, AttributeError):
            supports_enable_thinking = False

    if is_reasoning_model:
        thinking_support = " (enable_thinking parameter)" if supports_enable_thinking else ""
        print(f"   ✅ Detected REASONING model{thinking_support}")
    else:
        print(f"   ✅ Detected STANDARD model (no reasoning template)")

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
    print(f"   To reprocess, set FORCE_PREPROCESS=true in .env (operational flag)")

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

    # Check if dataset requires authentication and handle login
    try:
        # Try to get dataset config (this will fail if gated and not logged in)
        from huggingface_hub import dataset_info, HfApi
        try:
            info = dataset_info(DATASET_NAME)
            if info.gated:
                print(f"⚠️  Dataset '{DATASET_NAME}' is gated and requires authentication")
                print(f"   Checking HuggingFace login status...")

                # Check if already logged in
                try:
                    api = HfApi()
                    api.whoami()
                    print(f"   ✅ Already logged in to HuggingFace")
                except Exception:
                    # Need to login
                    print(f"\n🔐 Please login to HuggingFace to access this dataset")
                    print(f"   Get your token from: https://huggingface.co/settings/tokens")
                    from huggingface_hub import login
                    try:
                        login()
                        print(f"   ✅ Login successful!")
                    except Exception as e:
                        print(f"\n❌ Login failed: {e}")
                        print(f"\nAlternatively, set HF_TOKEN in your .env file")
                        print(f"Or use an ungated dataset like 'yahma/alpaca-cleaned'")
                        exit(1)
        except Exception:
            # Dataset info fetch failed, might still work if public
            pass
    except ImportError:
        pass

    # Load dataset with split configuration
    try:
        # First try standard loading
        available_splits = get_dataset_split_names(DATASET_NAME, config_name=DATASET_SUBSET)
        print(f"   Found {len(available_splits)} available splits: {available_splits}")

        # Determine which splits to load
        if DATASET_SPLIT == "all":
            splits_to_load = available_splits
            print(f"   Loading all splits: {splits_to_load}")
        else:
            if DATASET_SPLIT not in available_splits:
                raise ValueError(
                    f"Requested split '{DATASET_SPLIT}' not found in dataset. "
                    f"Available splits: {available_splits}"
                )
            splits_to_load = [DATASET_SPLIT]
            print(f"   Loading split: {DATASET_SPLIT}")

        all_datasets = []
        for split_name in splits_to_load:
            split_data = load_dataset(DATASET_NAME, name=DATASET_SUBSET, split=split_name)
            all_datasets.append(split_data)
            print(f"   Loaded {split_name}: {len(split_data)} samples")

        if len(all_datasets) > 1:
            dataset = concatenate_datasets(all_datasets)
            print(f"   Total samples (merged): {len(dataset)}")
        else:
            dataset = all_datasets[0]
            print(f"   Total samples: {len(dataset)}")
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print(f"   ⚠️  Dataset uses legacy loading script")
            print(f"   Loading as generic JSON dataset...")
            # Load as generic json/jsonl without the dataset-specific script
            # This bypasses the deprecated lima.py script
            from huggingface_hub import hf_hub_download

            # Download the data file directly
            data_file = hf_hub_download(
                repo_id=DATASET_NAME,
                filename="train.jsonl",
                repo_type="dataset"
            )

            # Load as generic JSON
            dataset = load_dataset("json", data_files=data_file, split="train")
            print(f"   Loaded dataset: {len(dataset)} samples")
        else:
            raise

    # Detect dataset type (reasoning vs standard)
    print(f"\n   🔍 Analyzing dataset content...")
    sample_size = min(50, len(dataset))
    reasoning_count = 0

    # Reasoning indicators in dataset content
    reasoning_indicators = [
        "<think>", "</think>",  # Explicit thinking tags
        "let me think", "let's think", "thinking step by step",
        "step 1:", "step 2:", "step 3:",  # Step-by-step
        "reasoning:", "analysis:", "let's break this down",
        "first,", "second,", "finally,",  # Sequential reasoning
    ]

    for i in range(sample_size):
        sample_text = str(dataset[i]).lower()
        if any(indicator in sample_text for indicator in reasoning_indicators):
            reasoning_count += 1

    # Dataset is "reasoning" if >20% of samples show reasoning
    is_reasoning_dataset = (reasoning_count / sample_size) > 0.2

    if is_reasoning_dataset:
        print(f"   ✅ Detected REASONING dataset ({reasoning_count}/{sample_size} samples have chain-of-thought)")
    else:
        print(f"   ✅ Detected STANDARD dataset ({reasoning_count}/{sample_size} samples have reasoning)")

    # Validate model-dataset compatibility
    print(f"\n   🔍 Validating model-dataset compatibility...")
    print(f"      Model type: {'REASONING' if is_reasoning_model else 'STANDARD'}")
    print(f"      Dataset type: {'REASONING' if is_reasoning_dataset else 'STANDARD'}")

    # Store whether to enable thinking based on compatibility
    enable_thinking = True  # Default for reasoning models

    if is_reasoning_model and not is_reasoning_dataset:
        print(f"\n   ⚠️  WARNING: MISMATCH DETECTED!")
        print(f"      You are training a REASONING model on a STANDARD dataset.")
        print(f"      This will likely cause the model to:")
        print(f"      • Output empty <think></think> tags")
        print(f"      • Lose its reasoning capabilities")
        print(f"      • Produce degraded responses")
        print(f"\n   💡 Recommended actions:")
        print(f"      1. Use a reasoning dataset (with chain-of-thought examples)")
        if supports_enable_thinking:
            print(f"      2. OR disable reasoning (set enable_thinking=False)")
        else:
            print(f"      2. OR switch to 'chatml' template")
        print(f"\n   ❓ Continue with reasoning disabled? (y/N): ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            print(f"\n   ❌ Preprocessing cancelled by user")
            exit(0)
        else:
            if supports_enable_thinking:
                print(f"\n   ⚠️  Disabling thinking mode (enable_thinking=False)")
                enable_thinking = False
            else:
                print(f"\n   ⚠️  Switching to 'chatml' template (disabling reasoning)")
                tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    elif not is_reasoning_model and is_reasoning_dataset:
        print(f"\n   ⚠️  WARNING: MISMATCH DETECTED!")
        print(f"      You are training a STANDARD model on a REASONING dataset.")
        print(f"      The model doesn't support reasoning templates.")
        print(f"\n   💡 Recommended: Use a reasoning-capable model (e.g., Qwen, DeepSeek-R1)")
        print(f"\n   ❓ Continue anyway? (y/N): ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            print(f"\n   ❌ Preprocessing cancelled by user")
            exit(0)
    else:
        print(f"   ✅ Model and dataset are compatible!")
        if is_reasoning_model and is_reasoning_dataset:
            print(f"      Using default reasoning template")
        else:
            print(f"      Using standard chat template")

    # Convert to chat template format
    if tokenizer:
        # Check format of first sample
        first_sample = dataset[0]

        # Handle question/answer format (e.g., GSM8K, MATH, etc.)
        if "question" in first_sample and "answer" in first_sample:
            print(f"\n   Detected question/answer format")

            def qa_to_messages(example):
                """Convert Q&A format to chat messages."""
                messages = [
                    {"role": "user", "content": example.get("question", "")},
                    {"role": "assistant", "content": example.get("answer", "")}
                ]
                return {"conversations": messages}

            print(f"   Converting to messages format...")
            dataset = dataset.map(qa_to_messages, num_proc=4, desc="Converting Q&A to chat")
            print(f"   ✅ Converted to HuggingFace chat format")

            # Now standardize using Unsloth
            print(f"\n   🦥 Standardizing dataset format using Unsloth...")
            dataset = standardize_sharegpt(
                dataset,
                tokenizer=tokenizer,
                aliases_for_system=["system"],
                aliases_for_user=["user", "human", "input"],
                aliases_for_assistant=["gpt", "assistant", "output"],
                num_proc=4
            )
            print(f"   ✅ Dataset standardized to HuggingFace chat format")
        # Some datasets have conversations as a list of strings (alternating user/assistant)
        elif "conversations" in first_sample and isinstance(first_sample["conversations"], list):
            if len(first_sample["conversations"]) > 0 and isinstance(first_sample["conversations"][0], str):
                print(f"\n   Detected alternating string conversation format")

                # Convert alternating strings to messages format
                def strings_to_messages(example):
                    convos = example.get("conversations", [])
                    messages = []
                    for i, content in enumerate(convos):
                        role = "user" if i % 2 == 0 else "assistant"
                        messages.append({"role": role, "content": content})
                    return {"messages": messages}

                print(f"   Converting to messages format...")
                dataset = dataset.map(strings_to_messages, num_proc=4, desc="Converting format")
                print(f"   ✅ Converted to HuggingFace chat format")
            else:
                # Standard ShareGPT format with dicts
                print(f"\n   🦥 Standardizing dataset format using Unsloth...")
                dataset = standardize_sharegpt(
                    dataset,
                    tokenizer=tokenizer,
                    aliases_for_system=["system"],
                    aliases_for_user=["user", "human", "input"],
                    aliases_for_assistant=["gpt", "assistant", "output"],
                    num_proc=4
                )
                print(f"   ✅ Dataset standardized to HuggingFace chat format")
        else:
            # Use Unsloth's standardizer for other formats
            print(f"\n   🦥 Standardizing dataset format using Unsloth...")
            dataset = standardize_sharegpt(
                dataset,
                tokenizer=tokenizer,
                aliases_for_system=["system"],
                aliases_for_user=["user", "human", "input"],
                aliases_for_assistant=["gpt", "assistant", "output"],
                num_proc=4
            )
            print(f"   ✅ Dataset standardized to HuggingFace chat format")

        # Apply chat template to convert messages to text
        def convert_to_text(example):
            # standardize_sharegpt creates "conversations" field, not "messages"
            messages = example.get("conversations", example.get("messages", []))
            if not messages:
                return {"text": ""}

            # Apply chat template
            # Only use enable_thinking for models that support it
            try:
                if supports_enable_thinking:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=enable_thinking
                    )
                else:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                return {"text": text}
            except Exception as e:
                # Log error for debugging
                print(f"\n⚠️  Error converting sample: {str(e)[:100]}")
                return {"text": ""}

        thinking_status = "enabled" if enable_thinking else "disabled"
        print(f"   Applying chat template (thinking: {thinking_status})...")
        dataset = dataset.map(convert_to_text, num_proc=4, desc="Formatting")

        print("   Filtering invalid samples...")
        original_size = len(dataset)
        dataset = dataset.filter(lambda x: len(x["text"]) > 0)
        filtered_out = original_size - len(dataset)
        print(f"   ✅ Kept {len(dataset)}/{original_size} valid samples ({filtered_out} filtered)")

        # Check if all samples were filtered out
        if len(dataset) == 0:
            print(f"\n❌ ERROR: All samples were filtered out!")
            print(f"   This usually means the chat template conversion failed.")
            print(f"   Check that the model supports the dataset format.")
            exit(1)

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
# Recommendation Logic - Top-Down Strategy
# ============================================
print(f"\n" + "-"*70)
print("🎯 RECOMMENDED CONFIGURATION (Go Big First)")
print("-"*70)

# Step 1: Detect dataset quality (heuristic-based)
KNOWN_HIGH_QUALITY = ["GAIR/lima", "databricks/databricks-dolly-15k"]
is_high_quality = any(ds in DATASET_NAME for ds in KNOWN_HIGH_QUALITY)

# Analyze average text length as quality proxy (if available)
if dataset_metadata.get('avg_length') != "not_checked":
    avg_token_length = dataset_metadata.get('avg_length', 0)
    if avg_token_length > 800:  # Longer, detailed responses suggest higher quality
        is_high_quality = True

# Step 2: Determine epochs based on quality
if is_high_quality:
    target_epochs = 3  # High quality: safe to train longer
    quality_note = "High quality curated data detected → Safe for 3 epochs"
else:
    target_epochs = 1  # Lower quality/synthetic: one pass only
    quality_note = "Large/synthetic data → 1 epoch to avoid memorizing noise"

# Step 3: Determine LoRA configuration (Go Big First!)
if final_dataset_size < 5000:
    # Small dataset: Need strong signal
    recommended_lora_rank = 64
    recommended_lora_alpha = 128  # 2× rank (aggressive)
    recommended_lr = "3e-4"  # Higher LR for small data
    rank_note = "Small dataset → High rank + aggressive LR for strong learning"
elif final_dataset_size < 20000:
    # Medium dataset
    recommended_lora_rank = 32
    recommended_lora_alpha = 64
    recommended_lr = "2e-4"
    rank_note = "Medium dataset → Balanced configuration"
else:
    # Large dataset
    recommended_lora_rank = 16
    recommended_lora_alpha = 32
    recommended_lr = "2e-4"
    rank_note = "Large dataset → Standard LoRA configuration"

# Step 4: Batch size (start with 8, adjust for VRAM)
recommended_effective_batch = 8  # Good default for most cases
if gpu_available and available_vram < 8:
    # Low VRAM: smaller batch, more accumulation
    recommended_batch_size = 2
    recommended_grad_accum = 4
elif gpu_available and available_vram < 12:
    # Medium VRAM
    recommended_batch_size = 4
    recommended_grad_accum = 2
else:
    # High VRAM
    recommended_batch_size = 4
    recommended_grad_accum = 2

# Calculate training steps
steps_per_epoch = math.ceil(final_dataset_size / recommended_effective_batch)
recommended_max_steps = steps_per_epoch * target_epochs

print(f"\n📊 Dataset Analysis:")
print(f"   Size: {final_dataset_size} samples")
print(f"   Quality: {'HIGH (curated/detailed)' if is_high_quality else 'STANDARD (large-scale)'}")
avg_len_display = dataset_metadata.get('avg_length', 'unknown')
if avg_len_display != "not_checked":
    print(f"   Avg length: {avg_len_display:.0f} tokens")
else:
    print(f"   Avg length: {avg_len_display}")
print(f"   → {quality_note}")

print(f"\n🚀 OPTIMAL CONFIGURATION (Maximum Performance):")
print(f"   Add to training_params.yaml:")
print(f"   training:")
print(f"     lora:")
print(f"       rank: {recommended_lora_rank}")
print(f"       alpha: {recommended_lora_alpha}")
print(f"     batch:")
print(f"       size: {recommended_batch_size}")
print(f"       gradient_accumulation_steps: {recommended_grad_accum}")
print(f"     optimization:")
print(f"       learning_rate: {recommended_lr}")
print(f"     epochs:")
print(f"       num_train_epochs: {target_epochs}")
print(f"       max_steps: {recommended_max_steps}  # Or use epochs")

vram_estimate = 6 + (recommended_lora_rank / 16) * 2  # Rough estimate
print(f"\n⚙️  Estimated VRAM: ~{vram_estimate:.0f}GB")
if gpu_available:
    print(f"   Your GPU: {available_vram:.1f}GB available")

# Provide fallback options if VRAM limited
if gpu_available and available_vram < vram_estimate:
    print(f"\n⚠️  VRAM MAY BE INSUFFICIENT")
    print(f"\n💡 If you encounter OOM errors, try these (in order):")
    print(f"\n   Option 1 - Reduce Batch Size:")
    print(f"      batch:")
    print(f"        size: 2")
    print(f"        gradient_accumulation_steps: 4")
    print(f"\n   Option 2 - Reduce LoRA Rank:")
    print(f"      lora:")
    print(f"        rank: {recommended_lora_rank // 2}")
    print(f"        alpha: {recommended_lora_alpha // 2}")
    print(f"\n   Option 3 - Reduce Alpha First (if overfitting):")
    print(f"      lora:")
    print(f"        alpha: {recommended_lora_rank}  # 1:1 ratio instead of 2:1")
    print(f"\n   Option 4 - Both:")
    print(f"      lora.rank: {recommended_lora_rank // 2}, batch.size: 2")

print(f"\n📖 Strategy:")
print(f"   {rank_note}")
print(f"   • Start BIG: High rank + aggressive alpha (rank×2)")
print(f"   • If overfitting: Reduce alpha first (2:1 → 1:1)")
print(f"   • If still overfitting: Then reduce rank")
print(f"   • If VRAM limited: Reduce batch size first, then rank")

print(f"\n🎓 Training Plan:")
print(f"   {steps_per_epoch} steps/epoch × {target_epochs} epoch(s) = {recommended_max_steps} total steps")
print(f"   Samples seen: {recommended_max_steps * recommended_effective_batch}")
print(f"   Expected time: ~{recommended_max_steps * 2 / 60:.0f}-{recommended_max_steps * 4 / 60:.0f} minutes")

# VRAM-based recommendations
if gpu_available:
    print(f"\n🎮 GPU Optimization:")
    print(f"   Available VRAM: {available_vram:.1f} GB")
    if available_vram < 4:
        print(f"   • Low VRAM: Using batch.size=1 with high accumulation")
        print(f"   • Consider: optimization.use_gradient_checkpointing=true (saves VRAM)")
    elif available_vram < 8:
        print(f"   • Medium VRAM: Balanced settings")
        print(f"   • Optional: optimization.use_gradient_checkpointing=false for 20% speed boost")
    else:
        print(f"   • High VRAM: Can increase batch.size for faster training")
        print(f"   • Recommended: optimization.use_gradient_checkpointing=false for maximum speed")

# Warning if current config will train too many epochs
if CURRENT_MAX_STEPS > 0:
    current_epochs_estimate = (CURRENT_MAX_STEPS * current_effective_batch) / final_dataset_size
    # Warn if training for more than 3 epochs (likely overfitting)
    if current_epochs_estimate > 3:
        print(f"\n⚠️  WARNING: Current epochs.max_steps={CURRENT_MAX_STEPS} will train for ~{current_epochs_estimate:.1f} epochs")
        print(f"   This is likely too much and may cause overfitting!")
        # Calculate recommended max_steps for target_epochs
        recommended_steps_for_target = steps_per_epoch * target_epochs
        print(f"   Recommended: Use epochs.max_steps={recommended_steps_for_target} for {target_epochs} epoch(s) instead")

print("\n" + "="*70)
print("✅ PREPROCESSING COMPLETE")
print("="*70)
print(f"\nNext steps:")
print(f"1. Update training_params.yaml with recommended settings above")
print(f"2. Run: python scripts/train.py")
print(f"   (Or: python scripts/train.py --config quick_test.yaml)")
print(f"3. Monitor training loss - stop if it plateaus early")
print(f"\nPreprocessed data saved to: {PREPROCESSED_DATASET_PATH}")
print("="*70 + "\n")
