"""
Build/Convert LoRA adapters to various formats
Handles: merged_16bit, merged_4bit, GGUF (various quants)

Requires LoRA adapters from train.py first!
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from unsloth import FastLanguageModel

# Helper functions
def get_bool_env(key, default=False):
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')

def get_int_env(key, default):
    return int(os.getenv(key, str(default)))

# Configuration
LORA_BASE_MODEL = os.getenv("LORA_BASE_MODEL", "unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
INFERENCE_BASE_MODEL = os.getenv("INFERENCE_BASE_MODEL", "")  # Optional - for true 16-bit quality
OUTPUT_MODEL_NAME = os.getenv("OUTPUT_MODEL_NAME", "auto")
MAX_SEQ_LENGTH = get_int_env("MAX_SEQ_LENGTH", 2048)
OUTPUT_FORMATS = os.getenv("OUTPUT_FORMATS", "").split(",")
OUTPUT_FORMATS = [fmt.strip() for fmt in OUTPUT_FORMATS if fmt.strip()]
FORCE_REBUILD = get_bool_env("FORCE_REBUILD", False)

PUSH_TO_HUB = get_bool_env("PUSH_TO_HUB", False)
HF_USERNAME = os.getenv("HF_USERNAME", "your_username")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "auto")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Generate output model name
DATASET_NAME = os.getenv("DATASET_NAME", "yahma/alpaca-cleaned")
dataset_short_name = DATASET_NAME.split("/")[-1].lower().replace("_", "-")
if OUTPUT_MODEL_NAME == "auto" or not OUTPUT_MODEL_NAME:
    # Auto-generate from base model + dataset
    model_base = LORA_BASE_MODEL.split("/")[-1].replace("-unsloth-bnb-4bit", "").replace("-unsloth", "")
    output_model_name = f"{model_base}-{dataset_short_name}"
else:
    output_model_name = OUTPUT_MODEL_NAME

# Auto-generate paths
OUTPUT_DIR_BASE = os.getenv("OUTPUT_DIR_BASE", "./outputs")
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, output_model_name)
LORA_DIR = os.path.join(OUTPUT_DIR, "lora")

# Auto-generate HF model name
if HF_MODEL_NAME == "auto":
    HF_MODEL_NAME = output_model_name

print("\n" + "="*60)
print("🔨 UNSLOTH BUILD - Model Merging & Format Conversion")
print("="*60)

# Check build requirements
print("\n📋 Checking build requirements...")

# Check if LoRA adapters exist (required)
if not os.path.exists(LORA_DIR):
    print(f"\n❌ ERROR: LoRA adapters not found at: {LORA_DIR}")
    print("   Run 'python train.py' first to create LoRA adapters")
    exit(1)
print(f"✅ LoRA adapters found: {LORA_DIR}")

# Check for llama.cpp if GGUF format is requested
gguf_formats = [f for f in OUTPUT_FORMATS if f.startswith("gguf_")]
if gguf_formats:
    llama_cpp_paths = [
        "./llama.cpp/build/bin/llama-quantize",  # Local build
        "/usr/bin/llama-quantize",  # System install (AUR)
        shutil.which("llama-quantize"),  # In PATH
    ]
    llama_cpp_found = any(p and os.path.exists(p) if isinstance(p, str) else p for p in llama_cpp_paths)

    if not llama_cpp_found:
        print(f"\n❌ ERROR: GGUF conversion requires llama.cpp")
        print(f"   Requested formats: {gguf_formats}")
        print(f"\n   Install options:")
        print(f"   • System package manager (with CUDA support for GPU)")
        print(f"   • Build from source: https://github.com/ggerganov/llama.cpp")
        print(f"   • Pre-built release: https://github.com/ggerganov/llama.cpp/releases")
        print(f"\n   After install, ensure 'llama-quantize' is in PATH or ./llama.cpp/")
        exit(1)
    else:
        print("✅ llama.cpp found")

        # Create symlink for local build if needed (Unsloth expects binaries in root)
        if os.path.exists("./llama.cpp/build/bin/llama-quantize"):
            symlink_path = "./llama.cpp/llama-quantize"
            if not os.path.exists(symlink_path):
                try:
                    os.symlink("build/bin/llama-quantize", symlink_path)
                    print("   Created symlink: llama.cpp/llama-quantize → build/bin/llama-quantize")
                except Exception as e:
                    print(f"   ⚠️  Could not create symlink (non-critical): {e}")

# Determine which base model to use for merging
if INFERENCE_BASE_MODEL:
    merge_base_model = INFERENCE_BASE_MODEL
    print(f"\n🔧 Using INFERENCE_BASE_MODEL for merging: {merge_base_model}")
    print(f"   (16-bit base for best quality - requires more VRAM)")
else:
    merge_base_model = LORA_BASE_MODEL
    print(f"\n🔧 Using LORA_BASE_MODEL for merging: {merge_base_model}")
    print(f"   (4-bit base - same quality as training)")

# Check HuggingFace cache for base model
hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
model_cache_name = "models--" + merge_base_model.replace("/", "--")
model_cache_path = hf_cache_dir / model_cache_name

if model_cache_path.exists():
    snapshots = list((model_cache_path / "snapshots").glob("*"))
    if snapshots:
        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
        cached_files = list(latest_snapshot.glob("*.safetensors"))
        if cached_files:
            print(f"✅ Base model cached: {len(cached_files)} files found")
        else:
            print(f"⚠️  Base model cache exists but no safetensors files found")
    else:
        print(f"⚠️  Base model cache directory exists but empty")
else:
    print(f"ℹ️  Base model not cached - will download on first use")

MERGED_16BIT_DIR = os.path.join(OUTPUT_DIR, "merged_16bit")
print(f"\n📋 Build plan:")
print(f"   1. Create merged_16bit from LoRA + base model")
if OUTPUT_FORMATS:
    print(f"   2. Convert to additional formats: {', '.join(OUTPUT_FORMATS)}")
else:
    print(f"   2. No additional formats requested")
print()

# ==============================================================================
# STEP 1: Create merged_16bit model (LoRA + base)
# ==============================================================================
print("="*60)
print("📦 STEP 1: Creating Merged 16-bit Model")
print("="*60)

# Check if merged_16bit already exists
if os.path.exists(MERGED_16BIT_DIR) and not FORCE_REBUILD:
    print(f"✅ Merged 16-bit model already exists: {MERGED_16BIT_DIR}")
    print("   Skipping merge... (set FORCE_REBUILD=true to rebuild)")

    # Load tokenizer for later use
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MERGED_16BIT_DIR)
    model = None
else:
    if os.path.exists(MERGED_16BIT_DIR) and FORCE_REBUILD:
        print(f"⚠️  Merged model exists but FORCE_REBUILD=true")
        print(f"   Will rebuild: {MERGED_16BIT_DIR}\n")

    print(f"\n🦥 Loading base model: {merge_base_model}")
    print(f"   (This may take a while if not cached...)")

    # Load base model + LoRA adapters
    # Note: load_in_4bit should match the base model type
    load_in_4bit = "4bit" in merge_base_model.lower() or "bnb" in merge_base_model.lower()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merge_base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    print(f"✅ Base model loaded")

    # Apply LoRA adapters
    print(f"\n🔗 Applying LoRA adapters from: {LORA_DIR}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, LORA_DIR)
    print(f"✅ LoRA adapters applied")

    # Merge LoRA into base model
    print(f"\n🔀 Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print(f"✅ Weights merged")

    # Save merged model
    print(f"\n💾 Saving merged 16-bit model to: {MERGED_16BIT_DIR}")
    os.makedirs(MERGED_16BIT_DIR, exist_ok=True)
    model.save_pretrained(MERGED_16BIT_DIR)
    tokenizer.save_pretrained(MERGED_16BIT_DIR)
    print(f"✅ Merged model saved!")

    # Create Modelfile for Ollama
    modelfile_path = os.path.join(MERGED_16BIT_DIR, "Modelfile")
    print(f"\n📝 Creating Modelfile: {modelfile_path}")
    with open(modelfile_path, "w") as f:
        f.write(f"""# Modelfile for Ollama
# This is a template - adjust parameters as needed

FROM {MERGED_16BIT_DIR}

# Parameters (adjust based on your use case)
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System prompt (customize for your model)
SYSTEM You are a helpful AI assistant.

# Usage:
# 1. Create Ollama model:
#    ollama create my-model -f {modelfile_path}
#
# 2. Run the model:
#    ollama run my-model "Your prompt here"
#
# 3. For GGUF versions, see ../gguf/ folder
""")
    print(f"✅ Modelfile created")

    # Optional: Push to HuggingFace
    if PUSH_TO_HUB and HF_TOKEN:
        print(f"\n📤 Pushing to HuggingFace: {HF_USERNAME}/{HF_MODEL_NAME}")
        model.push_to_hub(f"{HF_USERNAME}/{HF_MODEL_NAME}", token=HF_TOKEN)
        tokenizer.push_to_hub(f"{HF_USERNAME}/{HF_MODEL_NAME}", token=HF_TOKEN)
        print("✅ Pushed to HuggingFace")

print("\n" + "="*60)
print("✅ Merged 16-bit model ready!")
print("="*60)
print(f"Location: {MERGED_16BIT_DIR}\n")

# If no additional formats requested, we're done
if not OUTPUT_FORMATS:
    print("✅ No additional formats requested (OUTPUT_FORMATS is empty)")

    # Calculate actual sizes
    from pathlib import Path
    lora_size = sum(f.stat().st_size for f in Path(LORA_DIR).rglob('*') if f.is_file())
    lora_size_mb = lora_size / (1024 * 1024)

    merged_size = sum(f.stat().st_size for f in Path(MERGED_16BIT_DIR).rglob('*') if f.is_file())
    merged_size_gb = merged_size / (1024 * 1024 * 1024)

    print(f"\n📊 Build complete! Available models:")
    print(f"   • LoRA adapters: {LORA_DIR}")
    print(f"     Size: {lora_size_mb:.1f} MB")
    print(f"   • Merged 16-bit:  {MERGED_16BIT_DIR}")
    print(f"     Size: {merged_size_gb:.2f} GB")
    print(f"   • Modelfile:      {os.path.join(MERGED_16BIT_DIR, 'Modelfile')}")
    exit(0)

# ==============================================================================
# STEP 2: Create additional formats (if requested)
# ==============================================================================
print("\n" + "="*60)
print("📦 STEP 2: Creating Additional Formats")
print("="*60)

# Separate GGUF formats from others
gguf_formats = [f for f in OUTPUT_FORMATS if f.startswith("gguf_")]
other_formats = [f for f in OUTPUT_FORMATS if not f.startswith("gguf_")]

# Process merged_4bit format (requires model reload)
if "merged_4bit" in other_formats:
    print("\n🔨 Building: merged_4bit")
    print("="*60)

    output_path = os.path.join(OUTPUT_DIR, "merged_4bit")

    # Check if already exists
    if os.path.exists(output_path) and not FORCE_REBUILD:
        print(f"✅ Already exists: {output_path}")
        print("   Skipping... (set FORCE_REBUILD=true to rebuild)")
    else:
        if os.path.exists(output_path) and FORCE_REBUILD:
            print(f"⚠️  Exists but FORCE_REBUILD=true")
            print(f"   Will overwrite: {output_path}")

        # Need to reload as LoRA model for 4-bit save
        print(f"\n🦥 Loading LoRA model for 4-bit conversion...")
        model_4bit, tokenizer_4bit = FastLanguageModel.from_pretrained(
            model_name=LORA_DIR,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        print(f"✅ Model loaded")

        print(f"\n💾 Saving merged 4-bit model to: {output_path}")
        model_4bit.save_pretrained_merged(output_path, tokenizer_4bit, save_method="merged_4bit")
        print(f"✅ Saved: {output_path}")

        if PUSH_TO_HUB and HF_TOKEN:
            print(f"\n📤 Pushing to HuggingFace: {HF_USERNAME}/{HF_MODEL_NAME}-4bit")
            model_4bit.push_to_hub_merged(
                f"{HF_USERNAME}/{HF_MODEL_NAME}-4bit",
                tokenizer_4bit,
                save_method="merged_4bit",
                token=HF_TOKEN
            )
            print("✅ Pushed to HuggingFace")

    print()

# Warn about unknown formats
unknown_formats = [f for f in other_formats if f not in ["merged_4bit"]]
if unknown_formats:
    print(f"⚠️  Unknown formats: {', '.join(unknown_formats)}")
    print("   Valid formats: merged_4bit")
    print("   For GGUF, use: gguf_q4_k_m, gguf_q5_k_m, gguf_q8_0, etc.")
    print()

# Process all GGUF formats together in a single gguf/ folder
if gguf_formats:
    print("="*60)
    print(f"🔨 Building GGUF Quantizations")
    print("="*60)

    # All GGUF quants go into a single gguf/ folder
    gguf_dir = os.path.join(OUTPUT_DIR, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    # Extract quantization methods
    quant_methods = [f.replace("gguf_", "").upper() for f in gguf_formats]

    print(f"\nOutput directory: {gguf_dir}")
    print(f"Quantizations to build: {', '.join(quant_methods)}")
    print()

    # Use the merged_16bit model created above
    print(f"Using merged 16-bit model: {MERGED_16BIT_DIR}")
    print()

    # Step 2: Convert to F16 GGUF (only once, reused for all quants)
    import subprocess

    f16_gguf = os.path.join(gguf_dir, "model.F16.gguf")

    # Check if F16 already exists
    if os.path.exists(f16_gguf) and not FORCE_REBUILD:
        print(f"Step 2: F16 GGUF already exists, reusing: {f16_gguf}")
    else:
        print("Step 2: Converting safetensors → F16 GGUF...")
        result = subprocess.run([
            "python", "llama.cpp/convert_hf_to_gguf.py",
            MERGED_16BIT_DIR,
            "--outfile", f16_gguf,
            "--outtype", "f16"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Conversion failed:\n{result.stderr}")
            raise RuntimeError("GGUF conversion failed")

        print(f"✅ F16 GGUF created: {f16_gguf}")
    print()

    # Step 3: Quantize to each requested precision
    print("Step 3: Creating quantized versions...")
    quantized_files = []

    for quant_method in quant_methods:
        quant_gguf = os.path.join(gguf_dir, f"model.{quant_method}.gguf")

        # Skip if already exists
        if os.path.exists(quant_gguf) and not FORCE_REBUILD:
            print(f"  ✓ {quant_method}: Already exists, skipping")
            quantized_files.append((quant_method, quant_gguf))
            continue

        # Don't quantize F16 (it's the source)
        if quant_method == "F16":
            print(f"  ✓ F16: Already created (source file)")
            quantized_files.append((quant_method, f16_gguf))
            continue

        print(f"  → {quant_method}: Quantizing...")
        result = subprocess.run([
            "llama.cpp/build/bin/llama-quantize",
            f16_gguf,
            quant_gguf,
            quant_method
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"    ❌ Failed: {result.stderr}")
            continue

        quantized_files.append((quant_method, quant_gguf))
        print(f"    ✅ Created: {quant_gguf}")

    print()

    # Step 4: Copy tokenizer files (only once)
    print("Step 4: Copying tokenizer files...")
    for ext in ["*.json", "*.txt"]:
        for src_file in Path(MERGED_16BIT_DIR).glob(ext):
            dst_file = os.path.join(gguf_dir, src_file.name)
            if not os.path.exists(dst_file) or FORCE_REBUILD:
                shutil.copy2(src_file, dst_file)
    print("✅ Tokenizer files copied")
    print()

    # Step 5: Summary
    print("="*60)
    print("✅ GGUF QUANTIZATIONS COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {gguf_dir}\n")
    print("Generated files:")
    for quant_method, quant_file in quantized_files:
        if os.path.exists(quant_file):
            size = os.path.getsize(quant_file) / (1024**3)
            print(f"  • {quant_method:10s} {size:6.2f} GB - {os.path.basename(quant_file)}")
    print()

    # Optional: Remove F16 if not explicitly requested
    if "F16" not in quant_methods and os.path.exists(f16_gguf):
        print("💾 Removing intermediate F16 file (not in OUTPUT_FORMATS)...")
        os.remove(f16_gguf)
        print("✅ F16 file removed to save space")
        print()

print("="*60)
print("✅ BUILD COMPLETE")
print("="*60)
print(f"\nOutput directory: {OUTPUT_DIR}\n")

# Calculate actual sizes
from pathlib import Path
lora_size = sum(f.stat().st_size for f in Path(LORA_DIR).rglob('*') if f.is_file())
lora_size_mb = lora_size / (1024 * 1024)

merged_size = sum(f.stat().st_size for f in Path(MERGED_16BIT_DIR).rglob('*') if f.is_file())
merged_size_gb = merged_size / (1024 * 1024 * 1024)

print("Available models:")
print(f"  ✓ lora:          {LORA_DIR}")
print(f"                   Size: {lora_size_mb:.1f} MB")
print(f"  ✓ merged_16bit:  {MERGED_16BIT_DIR}")
print(f"                   Size: {merged_size_gb:.2f} GB")

for output_format in other_formats:
    output_path = os.path.join(OUTPUT_DIR, output_format)
    if os.path.exists(output_path):
        print(f"  ✓ {output_format}: {output_path}")
    else:
        print(f"  ✗ {output_format}: skipped or failed")

if gguf_formats:
    gguf_dir = os.path.join(OUTPUT_DIR, "gguf")
    if os.path.exists(gguf_dir):
        gguf_files = list(Path(gguf_dir).glob("*.gguf"))
        print(f"  ✓ gguf:          {gguf_dir} ({len(gguf_files)} quantizations)")
    else:
        print(f"  ✗ gguf:          failed")

# Generate/update README files for all built formats
print("\n📝 Generating README documentation for built formats...")
try:
    import subprocess
    result = subprocess.run(
        ["python", "generate_readme.py"],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode == 0:
        print(f"✅ README files updated")
    else:
        print(f"⚠️  README generation failed (non-critical)")
except Exception as e:
    print(f"⚠️  Could not generate READMEs (non-critical): {e}")

print("="*60 + "\n")
