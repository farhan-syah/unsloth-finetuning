"""
Build/Convert LoRA adapters to various formats
Handles: merged_16bit, merged_4bit, GGUF (various quants)

Requires LoRA adapters from train.py first!
Configure output formats in training_params.yaml
"""

import os
import json
import shutil
import argparse
from pathlib import Path

from unsloth import FastLanguageModel
from unsloth.save import create_ollama_modelfile
from unsloth.ollama_template_mappers import MODEL_TO_OLLAMA_TEMPLATE_MAPPER

# Load configuration
from config_loader import get_config_for_script

# Parse command line arguments
parser = argparse.ArgumentParser(description="Build/convert LoRA adapters to various formats")
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
    Get Ollama template name for a given model using Unsloth's mapper.
    Returns template name (e.g., "llama-3.1", "qwen2.5") or None if not found.
    """
    # Check direct mapping first
    if model_name in MODEL_TO_OLLAMA_TEMPLATE_MAPPER:
        return MODEL_TO_OLLAMA_TEMPLATE_MAPPER[model_name]
    return None

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

# Model configuration from YAML config
LORA_BASE_MODEL = config.model.base_model
INFERENCE_BASE_MODEL = config.model.inference_model
OUTPUT_MODEL_NAME = config.model.output_name

# Training parameters from YAML config
MAX_SEQ_LENGTH = config.training.data.max_seq_length
OUTPUT_FORMATS = config.output.formats

# Paths from .env config
FORCE_REBUILD = env_config['force_rebuild']
CACHE_DIR = env_config['cache_dir']

# Set HuggingFace cache to project directory for consistency
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")

PUSH_TO_HUB = env_config['push_to_hub']
HF_USERNAME = env_config['hf_username']
HF_MODEL_NAME = env_config['hf_model_name']
HF_TOKEN = env_config['hf_token']

# Generate output model name (dataset from YAML config)
DATASET_NAME = config.dataset.name
dataset_short_name = DATASET_NAME.split("/")[-1].lower().replace("_", "-")
if OUTPUT_MODEL_NAME == "auto" or not OUTPUT_MODEL_NAME:
    # Auto-generate from base model + dataset
    model_base = LORA_BASE_MODEL.split("/")[-1].replace("-unsloth-bnb-4bit", "").replace("-unsloth", "")
    output_model_name = f"{model_base}-{dataset_short_name}"
else:
    output_model_name = OUTPUT_MODEL_NAME

# Auto-generate paths
OUTPUT_DIR_BASE = env_config['output_dir_base']
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
    print("   Run 'python scripts/train.py' first to create LoRA adapters")
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
    print(f"   (16-bit base for best quality)")
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
    # Save with token=False to use local cache only
    model.save_pretrained(MERGED_16BIT_DIR, token=False)
    save_tokenizer_with_template(tokenizer, MERGED_16BIT_DIR, token=False)
    print(f"✅ Merged model saved!")

    # Create Modelfile for Ollama
    modelfile_path = os.path.join(MERGED_16BIT_DIR, "Modelfile")

    print(f"\n📝 Creating Modelfile: {modelfile_path}")

    # Try Unsloth's create_ollama_modelfile function first
    modelfile_content = create_ollama_modelfile(
        tokenizer=tokenizer,
        base_model_name=LORA_BASE_MODEL,
        model_location=MERGED_16BIT_DIR
    )

    # Track template info for README generation
    template_key = None
    template_display_name = None

    if modelfile_content:
        # Get template name for display
        template_key = MODEL_TO_OLLAMA_TEMPLATE_MAPPER.get(LORA_BASE_MODEL, "unknown")
        print(f"   Detected template: {template_key} (from Unsloth)")
    else:
        # Fallback: Use template mapper to get template from model name
        print(f"   No direct mapping found in create_ollama_modelfile, checking mapper...")

        template_key = get_template_for_model(LORA_BASE_MODEL)

        if template_key:
            # Get Ollama template from CHAT_TEMPLATES using the mapped template name
            from unsloth.chat_templates import CHAT_TEMPLATES

            ollama_template = None
            if template_key in CHAT_TEMPLATES:
                template_tuple = CHAT_TEMPLATES[template_key]
                if len(template_tuple) >= 4:
                    ollama_template = template_tuple[3]  # 4th element is Ollama template

            if ollama_template:
                # Use Unsloth's auto-generated template (works for all models)
                modelfile_content = ollama_template.replace("{__FILE_LOCATION__}", MERGED_16BIT_DIR)
                print(f"   ✅ Created Modelfile using '{template_key}' template from mapper")
            else:
                template_key = None  # Reset if no template found

        if not template_key and tokenizer.chat_template:
            # Generic fallback using tokenizer template
            print(f"   Creating generic Modelfile from tokenizer's chat template...")
            template = tokenizer.chat_template

            # Create basic Modelfile with template
            modelfile_content = (
                f"FROM {MERGED_16BIT_DIR}\n"
                f'TEMPLATE """{template}"""\n'
                'PARAMETER stop "<|im_end|>"\n'
                'PARAMETER stop "<|im_start|>"\n'
                'PARAMETER temperature 0.6\n'
                'PARAMETER min_p 0.0\n'
                'PARAMETER top_k 20\n'
                'PARAMETER top_p 0.95\n'
                'PARAMETER repeat_penalty 1\n'
            )
            print(f"   ✅ Created generic Modelfile")
        else:
            print(f"   ⚠️  Warning: No chat template available")
            modelfile_content = None

    if modelfile_content:
        with open(modelfile_path, "w") as f:
            f.write(modelfile_content)

        print(f"✅ Modelfile created: {modelfile_path}")
        print(f"\n💡 IMPORTANT: Review the Modelfile before using with Ollama!")
        print(f"   Verify the chat template matches your training dataset format")

        # Save chat template info for README generation
        if template_key:
            try:
                from unsloth.chat_templates import CHAT_TEMPLATES
                if template_key in CHAT_TEMPLATES:
                    template_tuple = CHAT_TEMPLATES[template_key]
                    # Extract template from Ollama format
                    if len(template_tuple) >= 4 and template_tuple[3]:
                        ollama_template = template_tuple[3]
                        # Extract TEMPLATE content
                        template_start = ollama_template.find('TEMPLATE """')
                        if template_start != -1:
                            template_start += len('TEMPLATE """')
                            template_end = ollama_template.find('"""', template_start)
                            if template_end != -1:
                                template_format = ollama_template[template_start:template_end].strip()

                                # Save to JSON for README generator
                                chat_template_info = {
                                    "template_key": template_key,
                                    "template_name": template_key.replace("-", " ").replace("_", " ").title(),
                                    "template_format": template_format,
                                    "model_name": LORA_BASE_MODEL
                                }

                                chat_template_json_path = os.path.join(MERGED_16BIT_DIR, "chat_template.json")
                                with open(chat_template_json_path, "w") as f:
                                    json.dump(chat_template_info, f, indent=2)
                                print(f"   ✅ Saved chat template info: {chat_template_json_path}")
            except Exception as e:
                print(f"   ⚠️  Could not save chat template info: {e}")
    else:
        print(f"⚠️  Could not create Modelfile automatically")
        print(f"   You may need to create a custom Modelfile manually")

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

# Clean up merged directory immediately after creation
print("🧹 Cleaning up merged model directory...")
from cleanup_utils import cleanup_merged_directory
cleanup_merged_directory(MERGED_16BIT_DIR, verbose=True)

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

    f16_gguf = os.path.join(gguf_dir, f"{output_model_name}-F16.gguf")

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
        quant_gguf = os.path.join(gguf_dir, f"{output_model_name}-{quant_method}.gguf")

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

    # Step 4: GGUF files are self-contained (no tokenizer files needed)
    print("Step 4: GGUF directory ready (tokenizer embedded in GGUF files)")
    print()

    # Step 5: Create Modelfile for Ollama with auto-detected template
    print("Step 5: Creating Modelfile for GGUF...")
    # Find the first quantized file (preferably Q4_K_M)
    preferred_quant = None
    for quant_method, quant_file in quantized_files:
        if quant_method == "Q4_K_M":
            preferred_quant = quant_file
            break
    if not preferred_quant and quantized_files:
        preferred_quant = quantized_files[0][1]

    if preferred_quant:
        modelfile_path = os.path.join(gguf_dir, "Modelfile")
        relative_gguf = os.path.basename(preferred_quant)

        # Try Unsloth's create_ollama_modelfile function first
        modelfile_content = create_ollama_modelfile(
            tokenizer=tokenizer,
            base_model_name=LORA_BASE_MODEL,
            model_location=f"./{relative_gguf}"
        )

        if modelfile_content:
            # Get template name for display
            template_name = MODEL_TO_OLLAMA_TEMPLATE_MAPPER.get(LORA_BASE_MODEL, "unknown")
            print(f"   Detected template: {template_name} (from Unsloth)")
        else:
            # Fallback: Use template mapper to get template from model name
            print(f"   No direct mapping found in create_ollama_modelfile, checking mapper...")

            template_key = get_template_for_model(LORA_BASE_MODEL)

            if template_key:
                # Get Ollama template from CHAT_TEMPLATES using the mapped template name
                from unsloth.chat_templates import CHAT_TEMPLATES

                ollama_template = None
                if template_key in CHAT_TEMPLATES:
                    template_tuple = CHAT_TEMPLATES[template_key]
                    if len(template_tuple) >= 4:
                        ollama_template = template_tuple[3]  # 4th element is Ollama template

                if ollama_template:
                    # Use Unsloth's auto-generated template (works for all models)
                    modelfile_content = ollama_template.replace("{__FILE_LOCATION__}", f"./{relative_gguf}")
                    print(f"   ✅ Created Modelfile using '{template_key}' template from mapper")
                else:
                    template_key = None  # Reset if no template found

            if not template_key and tokenizer.chat_template:
                # Generic fallback
                print(f"   Creating generic Modelfile from tokenizer's chat template...")
                template = tokenizer.chat_template

                modelfile_content = (
                    f"FROM ./{relative_gguf}\n"
                    f'TEMPLATE """{template}"""\n'
                    'PARAMETER stop "<|im_end|>"\n'
                    'PARAMETER stop "<|im_start|>"\n'
                    'PARAMETER temperature 0.6\n'
                    'PARAMETER min_p 0.0\n'
                    'PARAMETER top_k 20\n'
                    'PARAMETER top_p 0.95\n'
                    'PARAMETER repeat_penalty 1\n'
                )
                print(f"   ✅ Created generic Modelfile")
            else:
                print(f"   ⚠️  Warning: No chat template available")
                modelfile_content = None

        if modelfile_content:

            # Add comments about available quantizations
            available_quants = "\n".join([f"#   - {os.path.basename(qf)}" for _, qf in quantized_files])
            modelfile_with_comments = f"""# Modelfile for Ollama (GGUF)
# Auto-generated using Unsloth's template mapper
# This uses the {os.path.basename(preferred_quant)} quantization
#
# Note: You can change the FROM line to use a different quantization
# Available quantizations in this directory:
{available_quants}

{modelfile_content}"""

            with open(modelfile_path, "w") as f:
                f.write(modelfile_with_comments)

            print(f"✅ Modelfile created using Unsloth's template mapper: {modelfile_path}")
            print(f"\n💡 IMPORTANT: Before pushing GGUF to HuggingFace:")
            print(f"   1. Test the model locally: ollama create test -f {modelfile_path}")
            print(f"   2. Verify chat format works: ollama run test \"Hello\"")
            print(f"   3. Edit Modelfile if needed to match your dataset format")
            print(f"   4. Then push: python scripts/push.py")
        else:
            print(f"⚠️  Warning: Could not create GGUF Modelfile - no template mapping found for {LORA_BASE_MODEL}")
            print(f"   You may need to create a custom Modelfile manually")
    print()

    # Step 6: Summary
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

    # Clean up GGUF directory immediately after creation
    print("🧹 Cleaning up GGUF directory...")
    from cleanup_utils import cleanup_gguf_directory
    cleanup_gguf_directory(gguf_dir, verbose=True)
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
# Uses generate_readme_build.py which reads from training_metrics.json
# This ensures READMEs always reflect actual training parameters
print("\n📝 Generating README documentation for built formats...")
print("   (Reading training configuration from training_metrics.json)")
try:
    import subprocess
    result = subprocess.run(
        ["python", "scripts/generate_readme_build.py"],
        capture_output=True,
        text=True,
        timeout=10
    )
    if result.returncode == 0:
        print(f"✅ README files generated")
    else:
        print(f"⚠️  README generation failed (non-critical)")
        if result.stderr:
            print(f"   Error: {result.stderr}")
except Exception as e:
    print(f"⚠️  Could not generate READMEs (non-critical): {e}")

print("="*60 + "\n")
