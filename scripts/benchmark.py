"""
Interactive Benchmarking Script for Fine-tuned Models

Runs comprehensive benchmarks using lm-evaluation-harness on fine-tuned models.

Backends:
- Ollama: GGUF models via llama.cpp (requires ollama server running)
- PyTorch: Direct GPU inference using HuggingFace transformers (no server)
  Note: lm-eval calls this "hf" backend, but it's just local PyTorch inference

Based on research from /research/top-dataset.md:
- IFEval: Instruction-following evaluation
- GSM8K: Math reasoning with chain-of-thought
- MMLU: Knowledge across 57 subjects (catastrophic forgetting detection)
- HellaSwag: Commonsense reasoning
- TruthfulQA: Truthfulness evaluation (optional)

Usage:
    python scripts/benchmark.py [--timeout DURATION]

    --timeout: Watchdog timeout duration (e.g., "10min", "600s", "600")
               Default: 10min (600 seconds)
"""

import os
import sys
import json
import subprocess
import argparse
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Set HuggingFace cache to project directory for consistency
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CACHE_DIR = PROJECT_ROOT / "cache"
# Don't set HF_HOME - it creates a nested cache/ subdirectory
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")

# Suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress transformers warnings
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress Python warnings globally

# Prevent common lm-eval hangs (multiprocessing deadlocks)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism
os.environ["OMP_NUM_THREADS"] = "1"  # Disable OpenMP parallelism
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Disable BLAS parallelism

# Suppress langdetect warnings (for IFEval benchmark)
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("langdetect").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)

# MONKEYPATCH: Completely disable langdetect to prevent 18-20 minute delay
# When base models generate malformed text, langdetect becomes extremely slow
# We replace it with a dummy that always succeeds (returns 'en')
import sys
class FastLangDetect:
    """Dummy langdetect that always returns 'en' to avoid slow text analysis"""
    class LangDetectException(Exception):
        pass

    @staticmethod
    def detect(text):
        return 'en'  # Always return English

    @staticmethod
    def detect_langs(text):
        class Lang:
            lang = 'en'
            prob = 1.0
        return [Lang()]

# Replace langdetect module before lm_eval imports it
sys.modules['langdetect'] = FastLangDetect()

# Set logging level for subprocesses
os.environ["LOGLEVEL"] = "CRITICAL"
os.environ["PYTHONWARNINGS"] = "ignore"

# Load environment variables
load_dotenv(override=True)

# Load configuration from YAML
try:
    from config_loader import load_config, load_env_config
    config = load_config()
    env_config = load_env_config()
except Exception:
    # Fallback if config loading fails (benchmark can run without training config)
    config = None
    env_config = {}

# Check if lm_eval is installed
def check_lm_eval_installed():
    """Check if lm-evaluation-harness is installed"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "lm_eval", "--help"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

# Helper functions
def get_bool_env(key, default=False):
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')

def get_int_env(key, default):
    return int(os.getenv(key, str(default)))

def parse_timeout(timeout_str):
    """
    Parse timeout string into seconds.

    Supports formats:
    - "10min", "10m" -> 600 seconds
    - "30s", "30sec" -> 30 seconds
    - "600" -> 600 seconds (plain number)

    Returns:
        int: Timeout in seconds
    """
    timeout_str = timeout_str.strip().lower()

    # Match number followed by optional unit
    match = re.match(r'^(\d+)\s*(min|m|sec|s)?$', timeout_str)

    if not match:
        raise ValueError(f"Invalid timeout format: {timeout_str}")

    value = int(match.group(1))
    unit = match.group(2) or 's'  # Default to seconds if no unit

    # Convert to seconds
    if unit in ('min', 'm'):
        return value * 60
    elif unit in ('sec', 's'):
        return value
    else:
        raise ValueError(f"Unknown time unit: {unit}")

# Configuration (from YAML config or .env fallback)
OUTPUT_DIR_BASE = env_config.get('output_dir_base', os.getenv("OUTPUT_DIR_BASE", "./outputs"))
OLLAMA_BASE_URL = env_config.get('ollama_base_url', os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
BENCHMARK_BATCH_SIZE = config.benchmark.batch_size if config else get_int_env("BENCHMARK_BATCH_SIZE", 8)
BENCHMARK_MAX_TOKENS = config.benchmark.max_tokens if config else get_int_env("BENCHMARK_MAX_TOKENS", 640)
BENCHMARK_DEFAULT_BACKEND = config.benchmark.default_backend if config else os.getenv("BENCHMARK_DEFAULT_BACKEND", "ollama")
BENCHMARK_DEFAULT_TASKS = ",".join(config.benchmark.default_tasks) if config else os.getenv("BENCHMARK_DEFAULT_TASKS", "ifeval,gsm8k,hellaswag,mmlu")
INFERENCE_BASE_MODEL = env_config.get('inference_base_model', os.getenv("INFERENCE_BASE_MODEL", ""))

# Expected baselines from research (for comparison and forgetting detection)
EXPECTED_BASELINES = {
    "ifeval": {
        "1.7b": 0.45,  # Qwen2-1.5B baseline: ~51%, expect lower for untrained base
        "description": "Instruction-Following (verifiable constraints)"
    },
    "gsm8k": {
        "1.7b": 0.10,  # Small models: 5-15% on math
        "description": "Math Reasoning (8-shot with CoT)"
    },
    "hellaswag": {
        "1.7b": 0.37,  # TinyLlama 1.1B: ~36-38%
        "description": "Commonsense Reasoning"
    },
    "mmlu": {
        "1.7b": 0.48,  # Qwen2-1.5B: ~48-52%
        "description": "Knowledge (57 subjects)"
    },
    "truthfulqa_mc1": {
        "1.7b": 0.35,
        "description": "Truthfulness (single-answer)"
    },
    "truthfulqa_mc2": {
        "1.7b": 0.50,
        "description": "Truthfulness (multi-answer)"
    }
}

# ============================================================================
# HELPER FUNCTIONS FOR BENCHMARK MANAGEMENT
# ============================================================================

def find_base_benchmark(model_path):
    """
    Check if base benchmark exists.

    Returns: Path to base benchmark directory if exists, None otherwise
    """
    base_benchmark_dir = os.path.join(model_path, "benchmarks", "base", "lm-eval")
    if os.path.exists(base_benchmark_dir):
        # Check if there are result files
        result_files = [f for f in os.listdir(base_benchmark_dir) if f.startswith("results_") and f.endswith(".json")]
        if result_files:
            return base_benchmark_dir
    return None

def load_base_results(base_benchmark_dir):
    """
    Load base benchmark results from directory.

    Returns: Dict with scores and metadata, or None if failed
    """
    try:
        # Find the results JSON file
        result_files = [f for f in os.listdir(base_benchmark_dir) if f.startswith("results_") and f.endswith(".json")]
        if not result_files:
            return None

        result_file = os.path.join(base_benchmark_dir, result_files[0])
        with open(result_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load base benchmark: {str(e)}")
        return None

def find_latest_finetuned_benchmark(model_path):
    """
    Find the most recent fine-tuned benchmark directory.

    Returns: Path to latest fine-tuned benchmark directory, or None
    """
    finetuned_dir = os.path.join(model_path, "benchmarks", "fine-tuned")
    if not os.path.exists(finetuned_dir):
        return None

    # Get all timestamp directories
    timestamp_dirs = [d for d in os.listdir(finetuned_dir) if os.path.isdir(os.path.join(finetuned_dir, d))]
    if not timestamp_dirs:
        return None

    # Sort by timestamp (ISO format sorts correctly)
    timestamp_dirs.sort(reverse=True)
    latest_dir = os.path.join(finetuned_dir, timestamp_dirs[0], "lm-eval")

    if os.path.exists(latest_dir):
        return latest_dir
    return None

def load_finetuned_results(finetuned_benchmark_dir):
    """
    Load fine-tuned benchmark results from directory.

    Returns: Dict with scores and metadata, or None if failed
    """
    try:
        # Find the results JSON file
        result_files = [f for f in os.listdir(finetuned_benchmark_dir) if f.startswith("results_") and f.endswith(".json")]
        if not result_files:
            return None

        result_file = os.path.join(finetuned_benchmark_dir, result_files[0])
        with open(result_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load fine-tuned benchmark: {str(e)}")
        return None

def generate_benchmark_summary(model_path, model_name, base_model_name):
    """
    Generate benchmarks/benchmark.json by combining base + latest fine-tuned results.
    This is the single source of truth for benchmark data.
    """
    print(f"\n📊 Generating benchmark summary...")

    # Find base and fine-tuned benchmarks
    base_dir = find_base_benchmark(model_path)
    finetuned_dir = find_latest_finetuned_benchmark(model_path)

    if not base_dir and not finetuned_dir:
        print(f"⚠️  No benchmark results found")
        return

    # Load results
    base_results = load_base_results(base_dir) if base_dir else None
    finetuned_results = load_finetuned_results(finetuned_dir) if finetuned_dir else None

    # Build summary
    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "base_model_name": base_model_name,
        "timestamp": datetime.now().isoformat(),
        "base_model": {},
        "fine_tuned_model": {},
        "comparison": {}
    }

    # Add base results
    if base_results:
        summary["base_model"] = {
            "results": base_results.get("results", {}),
            "benchmark_dir": base_dir,
            "timestamp": os.path.getmtime(base_dir)
        }
        print(f"   ✅ Loaded base benchmark")

    # Add fine-tuned results
    if finetuned_results:
        summary["fine_tuned_model"] = {
            "results": finetuned_results.get("results", {}),
            "benchmark_dir": finetuned_dir,
            "timestamp": os.path.getmtime(finetuned_dir)
        }
        print(f"   ✅ Loaded fine-tuned benchmark")

    # Calculate comparison if both exist
    if base_results and finetuned_results:
        base_scores = base_results.get("results", {})
        ft_scores = finetuned_results.get("results", {})

        comparison = {}
        for task in base_scores.keys():
            if task in ft_scores:
                # lm-eval uses different metric names for different tasks
                # Try to find the primary accuracy metric
                base_metrics = base_scores[task]
                ft_metrics = ft_scores[task]

                # Priority: prompt_level_strict_acc > inst_level_strict_acc > accuracy
                metric_keys = [
                    "prompt_level_strict_acc,none",
                    "inst_level_strict_acc,none",
                    "accuracy,none",
                    "exact_match,none",
                    "acc,none"
                ]

                base_acc = 0
                ft_acc = 0
                metric_used = None

                for key in metric_keys:
                    if key in base_metrics and key in ft_metrics:
                        base_acc = base_metrics[key]
                        ft_acc = ft_metrics[key]
                        metric_used = key.replace(",none", "")
                        break

                delta = ft_acc - base_acc
                delta_pct = (delta / base_acc * 100) if base_acc > 0 else 0

                comparison[task] = {
                    "metric": metric_used,
                    "base_accuracy": base_acc,
                    "finetuned_accuracy": ft_acc,
                    "delta": delta,
                    "delta_percent": delta_pct
                }

        summary["comparison"] = comparison
        print(f"   ✅ Generated comparison")

    # Save summary
    output_file = os.path.join(model_path, "benchmarks", "benchmark.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"   📁 Saved to: {output_file}")

print("\n" + "="*70)
print("🧪 INTERACTIVE BENCHMARK RUNNER")
print("="*70)
print("\nThis script evaluates fine-tuned models using lm-evaluation-harness.")
print("It supports multiple backends and benchmark suites.\n")

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Interactive benchmark runner for fine-tuned models",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--timeout",
    type=str,
    default=None,
    help="Watchdog timeout duration (e.g., '10min', '30m', '600s', '600'). Default: disabled"
)

args = parser.parse_args()

# Parse timeout (None = disabled)
if args.timeout:
    try:
        WATCHDOG_TIMEOUT = parse_timeout(args.timeout)
        print(f"⏱️  Watchdog timeout: {WATCHDOG_TIMEOUT}s ({args.timeout})")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("   Use format like: --timeout 10min, --timeout 30m, --timeout 600s, or --timeout 600")
        sys.exit(1)
else:
    WATCHDOG_TIMEOUT = None
    print(f"⏱️  Watchdog timeout: disabled (use --timeout to enable)")

# Check if lm_eval is installed
print("🔍 Checking prerequisites...")
if not check_lm_eval_installed():
    print("\n" + "❌"*35)
    print("ERROR: lm-evaluation-harness is not installed!")
    print("❌"*35)
    print("\nTo install, run:")
    print("  pip install lm-eval")
    print("\nOr install with additional dependencies:")
    print("  pip install 'lm-eval[api]'")
    print("\nAfter installation, run this script again.")
    print("="*70 + "\n")
    sys.exit(1)
print("✅ lm-evaluation-harness is installed\n")

# Step 1: Detect available models
print("📂 Scanning for trained models...")
available_models = []
if os.path.exists(OUTPUT_DIR_BASE):
    for item in sorted(os.listdir(OUTPUT_DIR_BASE)):
        model_path = os.path.join(OUTPUT_DIR_BASE, item)
        if os.path.isdir(model_path):
            # Check if it has LoRA adapters or merged models
            has_lora = os.path.exists(os.path.join(model_path, "lora", "adapter_config.json"))
            has_merged_16bit = os.path.exists(os.path.join(model_path, "merged_16bit", "config.json"))
            has_merged_4bit = os.path.exists(os.path.join(model_path, "merged_4bit", "config.json"))
            has_gguf = any(Path(model_path).glob("gguf/*.gguf"))

            if has_lora or has_merged_16bit or has_merged_4bit or has_gguf:
                available_models.append({
                    "name": item,
                    "path": model_path,
                    "lora": has_lora,
                    "merged_16bit": has_merged_16bit,
                    "merged_4bit": has_merged_4bit,
                    "gguf": has_gguf
                })

if not available_models:
    print("❌ No trained models found in", OUTPUT_DIR_BASE)
    print("   Run 'python scripts/train.py' first to train a model.")
    sys.exit(1)

print(f"✅ Found {len(available_models)} model(s):\n")
for idx, model in enumerate(available_models, 1):
    formats = []
    if model["lora"]: formats.append("LoRA")
    if model["merged_16bit"]: formats.append("16-bit")
    if model["merged_4bit"]: formats.append("4-bit")
    if model["gguf"]: formats.append("GGUF")
    print(f"   {idx}. {model['name']}")
    print(f"      Available: {', '.join(formats)}")

# Step 2: Model selection
print("\n" + "-"*70)
if len(available_models) == 1:
    selected_model = available_models[0]
    print(f"📦 Auto-selected: {selected_model['name']}")
else:
    while True:
        try:
            choice = input(f"\nSelect model to benchmark (1-{len(available_models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                selected_model = available_models[idx]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(available_models)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Cancelled by user")
            sys.exit(0)

model_name = selected_model["name"]
model_path = selected_model["path"]
print(f"✅ Selected: {model_name}")

# Load base model info from training_metrics.json
base_model_name = None
training_base_model = None  # 4-bit model used for training
lora_dir = os.path.join(model_path, "lora")
metrics_file = os.path.join(lora_dir, "training_metrics.json")
if os.path.exists(metrics_file):
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
            training_base_model = metrics.get("model_name")
            print(f"   Training base model (4-bit): {training_base_model}")
    except Exception as e:
        print(f"⚠️  Could not load training metrics: {e}")

# Determine which base model to use for benchmarking
# Priority: INFERENCE_BASE_MODEL (16-bit) > training base model (4-bit)
if INFERENCE_BASE_MODEL:
    base_model_name = INFERENCE_BASE_MODEL
    print(f"   Benchmark base model (16-bit): {base_model_name}")
elif training_base_model:
    base_model_name = training_base_model
    print(f"   ⚠️  Using 4-bit training base for benchmark (set INFERENCE_BASE_MODEL for 16-bit)")

# Step 2b: Ask which model to benchmark (base or fine-tuned)
print("\n" + "-"*70)
print("🎯 SELECT BENCHMARK MODE")
print("-"*70)
print("\nWhich model would you like to benchmark?")
print("  1. Fine-tuned model (default)")
print("     → Benchmark your trained model")
print("     → Results saved to: benchmarks/fine-tuned/{timestamp}/")
print("  2. Base model")
print("     → Benchmark the original base model")
print("     → Results saved to: benchmarks/base/")
print("     → Note: Base only needs to be benchmarked once")

# Check if base benchmark already exists
base_exists = find_base_benchmark(model_path) is not None
if base_exists:
    print("\n   ⚠️  Base benchmark already exists!")

mode_choice = input("\nSelect mode (1-2) [1]: ").strip() or "1"

is_base_benchmark = False
if mode_choice == "2":
    is_base_benchmark = True
    if base_exists:
        replace = input("\n⚠️  Base benchmark already exists. Replace it? (y/n) [n]: ").strip().lower()
        if replace != 'y':
            print("❌ Cancelled - keeping existing base benchmark")
            sys.exit(0)
    print(f"✅ Will benchmark base model: {base_model_name}")
else:
    print("✅ Will benchmark fine-tuned model")
    if base_exists:
        print("   ℹ️  Base benchmark found - comparison will be generated automatically")

# Step 3: Backend selection
print("\n" + "-"*70)
print("🔧 SELECT BENCHMARK BACKEND")
print("-"*70)
print("\nAvailable backends:")
print("  1. Local PyTorch - Direct GPU inference (recommended)")
print("     → Requires: merged_16bit or merged_4bit format")
print("     → Uses: transformers library, loads model into VRAM")
print("     → Best for: Most users, no server needed")
print("  2. Ollama API - GGUF models via HTTP server")
print("     → Requires: ollama server running + model loaded")
print("     → Uses: llama.cpp inference via Ollama")
print("     → Best for: Testing GGUF quantizations")
print("  3. Both backends")
print("     → Test both PyTorch (merged) and Ollama (GGUF)")
print("     → Useful for comparing quantization impact")

backend_choice = input(f"\nSelect backend (1-3) [default: 1]: ").strip() or "1"

backends_to_run = []
if backend_choice == "1":
    backends_to_run.append("huggingface")
elif backend_choice == "2":
    backends_to_run.append("ollama")
elif backend_choice == "3":
    backends_to_run.extend(["huggingface", "ollama"])
else:
    print("❌ Invalid choice, using Local PyTorch")
    backends_to_run.append("huggingface")

# Step 4: Benchmark task selection
print("\n" + "-"*70)
print("📊 SELECT BENCHMARK TASKS")
print("-"*70)

# Load default tasks from training config
default_tasks = []
if config and hasattr(config, 'benchmark') and hasattr(config.benchmark, 'default_tasks'):
    default_tasks = config.benchmark.default_tasks or []

print("\nTask selection:")
if default_tasks:
    print(f"  1. Use configured tasks (default)")
    print(f"     → Tasks from training_params.yaml: {', '.join(default_tasks)}")
    print(f"  2. Custom tasks (enter manually)")
else:
    print(f"  1. Custom tasks")
    print(f"     → No default tasks configured in training_params.yaml")

task_choice = input(f"\nSelect option (1-2) [default: 1]: ").strip() or "1"

if task_choice == "1" and default_tasks:
    tasks = default_tasks
    print(f"✅ Using configured tasks: {', '.join(tasks)}")
else:
    print("\n💡 Common benchmark tasks:")
    print("  - ifeval: Instruction-following")
    print("  - gsm8k: Math reasoning")
    print("  - hellaswag: Commonsense reasoning")
    print("  - mmlu: General knowledge (57 subjects)")
    print("  - truthfulqa_mc1: Truthfulness (single-answer)")
    print("  - truthfulqa_mc2: Truthfulness (multi-answer)")
    print("  - arc_challenge: Science questions")
    print("\n  Run 'lm-eval --tasks list' to see all 11,000+ available tasks")
    print("  Or check: config/valid_lm_eval_tasks.txt")
    custom_tasks = input("\nEnter benchmark tasks (comma-separated): ").strip()
    tasks = [t.strip() for t in custom_tasks.split(",") if t.strip()]
    if not tasks:
        print("❌ No tasks entered, using ifeval as default")
        tasks = ["ifeval"]
    else:
        print(f"✅ Selected tasks: {', '.join(tasks)}")

# Validate that all selected tasks exist
print("\n🔍 Validating task names...")
try:
    # Load cached task list (much faster than running lm_eval --tasks list)
    task_list_file = PROJECT_ROOT / "config" / "valid_lm_eval_tasks.txt"

    if not task_list_file.exists():
        print(f"⚠️  Task list cache not found: {task_list_file}")
        print(f"   Run 'python scripts/generate_task_list.py' to create it")
        print(f"   Skipping validation...")
        available_tasks = set()
    else:
        with open(task_list_file, 'r') as f:
            available_tasks = set(line.strip() for line in f if line.strip())

    invalid_tasks = []
    for task in tasks:
        if task not in available_tasks:
            invalid_tasks.append(task)

    if invalid_tasks:
        print(f"\n❌ ERROR: Invalid task name(s) detected: {', '.join(invalid_tasks)}")
        print(f"\n💡 Possible corrections:")
        for task in invalid_tasks:
            if 'truthful' in task.lower():
                print(f"   • '{task}' → Use 'truthfulqa_mc1' or 'truthfulqa_mc2' instead")
            else:
                # Try to find similar tasks
                similar = [t for t in available_tasks if task.lower() in t.lower()]
                if similar:
                    print(f"   • '{task}' → Similar tasks: {', '.join(list(similar)[:3])}")
        print(f"\n   Run 'python -m lm_eval --tasks list' to see all available tasks")
        print(f"   Or check: config/valid_lm_eval_tasks.txt")
        sys.exit(1)

    print("✅ All task names are valid")
except subprocess.TimeoutExpired:
    print("⚠️  Task validation timed out, skipping validation...")
except Exception as e:
    print(f"⚠️  Could not validate task names: {e}")
    print("   Proceeding anyway, but benchmark may fail if tasks are invalid...")

# Step 4.5: Test mode vs Full mode
print("\n" + "-"*70)
print("🧪 SELECT EXECUTION MODE")
print("-"*70)
print("\nHow many samples should be evaluated?")
print("  1. Test mode (5 samples per benchmark) - ~30 seconds per task ⚡")
print("     → Quick verification of benchmark flow and README generation")
print("     → Not representative of actual model performance")
print("  2. Full mode (all samples) - varies by benchmark")
print("     → Complete evaluation for accurate performance metrics")
print("     → IFEval: ~15 min, GSM8K: ~15 min, HellaSwag: ~10 min, MMLU: ~30 min")

mode_choice = input(f"\nSelect mode (1-2) [default: 1 for testing]: ").strip() or "1"

test_mode = False
if mode_choice == "1":
    test_mode = True
    print("⚡ Test mode enabled - will run only 5 samples per benchmark")
    print("   ℹ️  This is for testing the flow, not for accurate performance evaluation")
elif mode_choice == "2":
    test_mode = False
    print("📊 Full mode enabled - will evaluate all samples")
else:
    print("❌ Invalid choice, using Test mode")
    test_mode = True

# Step 5: Batch size configuration
print("\n" + "-"*70)
print("⚙️  BATCH SIZE CONFIGURATION")
print("-"*70)
print("\nBatch size controls memory usage and speed:")
print("  • Higher = Faster but needs more VRAM")
print("  • Lower = Slower but uses less VRAM")
print("\nNote: Optimal batch size depends on both your GPU VRAM and model size.")
print("      Smaller models can use higher batch sizes on the same GPU.")
batch_size_input = input(f"\nBatch size (1-16) [default: {BENCHMARK_BATCH_SIZE}]: ").strip()
batch_size = int(batch_size_input) if batch_size_input else BENCHMARK_BATCH_SIZE

print(f"✅ Using batch size: {batch_size}")

# Step 6: Confirm and run
print("\n" + "="*70)
print("📋 BENCHMARK CONFIGURATION SUMMARY")
print("="*70)
print(f"  Model: {model_name}")
print(f"  Backends: {', '.join(backends_to_run)}")
print(f"  Tasks: {', '.join(tasks)}")
print(f"  Batch size: {batch_size}")
print(f"  Output: {model_path}/benchmark.json")
print("="*70)

confirm = input("\nProceed with benchmark? (y/n) [y]: ").strip().lower()
if confirm and confirm != 'y':
    print("❌ Cancelled by user")
    sys.exit(0)

# Step 7: Run benchmarks
print("\n" + "="*70)
print("🚀 STARTING BENCHMARK EXECUTION")
print("="*70)

# Create a timestamp for this benchmark run
benchmark_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def get_chat_template_info(model_path):
    """
    Extract chat template information from model tokenizer.

    Returns dict with:
    - has_template: bool
    - template: str or None
    - template_preview: str (first 200 chars for display)
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        chat_template = getattr(tokenizer, 'chat_template', None)

        if chat_template:
            # Truncate for preview
            preview = chat_template[:200] + "..." if len(chat_template) > 200 else chat_template
            return {
                "has_template": True,
                "template": chat_template,
                "template_preview": preview
            }
        else:
            return {
                "has_template": False,
                "template": None,
                "template_preview": "No chat template set"
            }
    except Exception as e:
        return {
            "has_template": False,
            "template": None,
            "template_preview": f"Error loading template: {str(e)}"
        }

def run_benchmark(backend, model_identifier, is_base_model=False):
    """
    Run lm-eval benchmark for a specific backend.

    Returns: (benchmark_scores, chat_template_info) tuple or (None, None) on failure
    """
    print(f"\n{'─'*70}")
    print(f"Running benchmarks on {backend.upper()} backend...")
    print(f"{'─'*70}\n")

    chat_template_info = None

    # Build lm_eval command (use python -m for better compatibility)
    cmd = [sys.executable, "-m", "lm_eval"]

    if backend == "ollama":
        # Check if Ollama is running
        try:
            import requests
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                print(f"⚠️  Ollama not responding at {OLLAMA_BASE_URL}")
                return None, None
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            return None, None

        # Prompt user to load model in Ollama
        print(f"\n📝 Before running Ollama benchmarks:")
        print(f"   1. Ensure your model is loaded in Ollama")
        print(f"   2. Example: ollama run {model_name.lower()}")
        ollama_model_name = input(f"\nEnter Ollama model name [{model_name.lower()}]: ").strip() or model_name.lower()

        cmd.extend([
            "--model", "ollama",
            "--model_args", f"base_url={OLLAMA_BASE_URL},model={ollama_model_name}"
        ])

    elif backend == "huggingface":
        # Determine model path based on whether this is base or fine-tuned
        if is_base_model:
            # Convert HF identifier to local cache path for faster evaluation
            # This avoids lm-eval making Hub API calls during post-processing
            try:
                from huggingface_hub import snapshot_download
                print(f"📥 Resolving base model: {model_identifier}")
                hf_model_path = snapshot_download(
                    repo_id=model_identifier,
                    cache_dir=str(CACHE_DIR),
                    local_files_only=False  # Download if not cached
                )
                print(f"✅ Using base model from cache: {hf_model_path}")
            except Exception as e:
                # Fallback to HF identifier if snapshot download fails
                print(f"⚠️  Could not resolve to local path: {e}")
                hf_model_path = model_identifier
                print(f"✅ Using base model from HF Hub: {hf_model_path}")
        else:
            # Check available formats for fine-tuned model
            if selected_model["merged_16bit"]:
                hf_model_path = os.path.abspath(os.path.join(model_path, "merged_16bit"))
                print(f"✅ Using merged_16bit format")
            elif selected_model["merged_4bit"]:
                hf_model_path = os.path.abspath(os.path.join(model_path, "merged_4bit"))
                print(f"✅ Using merged_4bit format")
            else:
                print("❌ No merged model found. Build with 'python scripts/build.py' first.")
                return None, None

        # Check and display chat template
        print(f"\n📝 Checking chat template...")
        chat_template_info = get_chat_template_info(hf_model_path)
        if chat_template_info["has_template"]:
            print(f"✅ Chat template found:")
            print(f"   {chat_template_info['template_preview']}")
        else:
            print(f"⚠️  {chat_template_info['template_preview']}")
            print(f"   Benchmark will run without chat template formatting")

        # HuggingFace backend configuration
        cmd.extend([
            "--model", "hf",
            "--model_args", f"pretrained={hf_model_path},trust_remote_code=True"
        ])

        # Add generation limits to prevent infinite loops and repetition
        # Configurable via BENCHMARK_MAX_TOKENS in .env (default: 640)
        cmd.extend([
            "--gen_kwargs", f"max_gen_toks={BENCHMARK_MAX_TOKENS}"
        ])

    # Create benchmark results directory in model output
    # New structure:
    # - Base: benchmarks/base/lm-eval/
    # - Fine-tuned: benchmarks/fine-tuned/{timestamp}/lm-eval/
    if is_base_model:
        benchmark_results_dir = os.path.join(model_path, "benchmarks", "base", "lm-eval")
    else:
        benchmark_results_dir = os.path.join(model_path, "benchmarks", "fine-tuned", benchmark_timestamp, "lm-eval")
    os.makedirs(benchmark_results_dir, exist_ok=True)

    # Add common arguments
    cmd.extend([
        "--tasks", ",".join(tasks),
        "--device", "cuda",
        "--batch_size", str(batch_size),
        "--output_path", benchmark_results_dir,
        "--log_samples",
        "--verbosity", "INFO"  # Reduce verbosity to avoid langdetect spam
    ])

    # Add limit for test mode
    if test_mode:
        cmd.extend(["--limit", "5"])

    # Apply chat template if available (only for HuggingFace backend)
    if backend == "huggingface" and chat_template_info and chat_template_info.get("has_template"):
        cmd.append("--apply_chat_template")
        print(f"✅ Using --apply_chat_template (template detected)")
    elif backend == "huggingface":
        print(f"⚠️  NOT using --apply_chat_template (no template found)")

    print(f"\n🔧 Command: {' '.join(cmd)}\n")

    try:
        # Run lm_eval with real-time output
        print(f"⏳ Running benchmarks... (this may take 15-60 minutes)")
        print(f"   Command: {' '.join(cmd[:4])}...\n")
        print("-" * 70)
        print("📊 Live output from lm-eval:")
        print("-" * 70)

        # Run with streaming output instead of capture
        # Create environment with our fast langdetect replacement to prevent 18-20 min delay
        env = os.environ.copy()
        env["PYTHONWARNINGS"] = "ignore"
        env["LOGLEVEL"] = "CRITICAL"

        # Inject our fast langdetect module by prepending scripts dir to PYTHONPATH
        # This makes Python import our fast_langdetect.py instead of the real langdetect
        scripts_dir = str(PROJECT_ROOT / "scripts")
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{scripts_dir}:{current_pythonpath}" if current_pythonpath else scripts_dir

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env  # Pass environment with fast langdetect injection
        )

        # Stream output in real-time (filter out langdetect errors)
        output_lines = []
        last_output_time = datetime.now()
        heartbeat_shown = False

        # Watchdog: Kill process if no output for too long (WATCHDOG_TIMEOUT set from --timeout arg)
        WATCHDOG_WARNING = 60   # Warn after 60 seconds of no output (catches post-100% processing)

        import threading
        import time

        def watchdog():
            """Monitor process health and show progress during silent phases"""
            # If timeout is disabled, don't do anything
            if WATCHDOG_TIMEOUT is None:
                return

            warned = False
            last_shown = 0
            last_cpu_check = 0

            while process.poll() is None:  # While process is running
                time_since_output = (datetime.now() - last_output_time).total_seconds()

                # Show warning after 60 seconds of no output (likely post-100% processing)
                if time_since_output > WATCHDOG_WARNING and not warned:
                    print(f"\n⏳ Computing metrics and finalizing results...")
                    print(f"   This phase can take 5-40 minutes depending on the task complexity")
                    print(f"   (Monitoring process activity...)")
                    warned = True
                    last_shown = int(time_since_output)
                    last_cpu_check = int(time_since_output)

                # Show activity updates every 2 minutes after warning
                elif warned and int(time_since_output) >= last_cpu_check + 120:
                    # Check if process is actually doing work
                    try:
                        import psutil
                        proc = psutil.Process(process.pid)
                        cpu_percent = proc.cpu_percent(interval=1.0)
                        mem_info = proc.memory_info()
                        mem_mb = mem_info.rss / (1024 * 1024)

                        if cpu_percent > 1:  # Process is active
                            print(f"   Still processing... ({int(time_since_output/60)}min elapsed, CPU: {cpu_percent:.1f}%, RAM: {mem_mb:.0f}MB)")
                        else:  # Process might be stuck
                            print(f"   ⚠️  Low CPU activity ({int(time_since_output/60)}min elapsed, CPU: {cpu_percent:.1f}%)")
                    except:
                        # psutil not available, just show elapsed time
                        print(f"   Still processing... ({int(time_since_output/60)} minutes elapsed)")

                    last_cpu_check = int(time_since_output)

                # Terminate after timeout (graceful shutdown to allow lm-eval to save results)
                if WATCHDOG_TIMEOUT and time_since_output > WATCHDOG_TIMEOUT:
                    print(f"\n\n⚠️  WARNING: No output for {int(time_since_output/60)} minutes - process may be stuck")
                    print("   Sending termination signal (allowing lm-eval to save results)...")
                    process.terminate()  # SIGTERM instead of SIGKILL - allows cleanup
                    print("   Waiting up to 60s for graceful shutdown...")
                    try:
                        process.wait(timeout=60)  # Wait up to 60s for graceful shutdown
                        print("   ✅ Process terminated gracefully")
                    except subprocess.TimeoutExpired:
                        print("   ⚠️  Process didn't terminate gracefully, forcing kill...")
                        process.kill()  # Force kill if it doesn't terminate
                    return

                time.sleep(5)  # Check every 5 seconds for better responsiveness

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()

        for line in process.stdout:
            # Skip langdetect errors (they're harmless and clutter output)
            if "Unable to detect language" in line or "No features in text" in line:
                continue
            print(line, end='', flush=True)  # Print to terminal
            output_lines.append(line)  # Save for error handling
            last_output_time = datetime.now()
            heartbeat_shown = False

        # Wait for completion with timeout
        print("\n⏳ Finalizing results (saving metrics and samples)...", flush=True)
        try:
            # Wait up to 5 minutes for finalization (should take 30-120 seconds normally)
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            print("\n⚠️  Warning: Finalization taking longer than expected (>5 min)")
            print("   The process might be stuck. Waiting another 2 minutes...")
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print("\n❌ Process appears to be stuck. Terminating...")
                process.kill()
                process.wait()
                print("   Process terminated. Checking for partial results...")
                # Don't return None yet - check if results file exists

        result_returncode = process.returncode
        result_output = ''.join(output_lines)  # Combine output for error checking

        print("\n" + "-" * 70)

        # Check if results exist even if process was killed
        if result_returncode != 0 or result_returncode is None:
            print(f"\n❌ Benchmark failed with return code {result_returncode}")
            print(f"\n📋 Error output:")
            print("-" * 70)
            error_output = result_output[-2000:] if len(result_output) > 2000 else result_output
            print(error_output)
            print("-" * 70)

            # Check for common errors and provide specific solutions
            print(f"\n💡 Troubleshooting tips:")

            if "langdetect" in error_output:
                print(f"\n   ⚠️  Missing dependency detected: langdetect")
                print(f"   This is required for IFEval benchmark.")
                print(f"\n   Quick fix:")
                print(f"   pip install langdetect sacrebleu rouge-score scikit-learn")
                print(f"\n   Or reinstall with all dependencies:")
                print(f"   bash install_dependencies.sh")
            elif "No module named" in error_output:
                module_name = error_output.split("No module named '")[1].split("'")[0] if "No module named '" in error_output else "unknown"
                print(f"\n   ⚠️  Missing Python module: {module_name}")
                print(f"   Install it with: pip install {module_name}")
            elif "CUDA out of memory" in error_output or "OutOfMemoryError" in error_output:
                print(f"\n   ⚠️  GPU out of memory!")
                print(f"   Solutions:")
                print(f"   1. Reduce batch_size (current: {batch_size})")
                print(f"   2. Close other GPU applications")
                print(f"   3. Use a smaller model format (4-bit instead of 16-bit)")
            elif "No such file or directory" in error_output or "FileNotFoundError" in error_output:
                print(f"\n   ⚠️  Model file not found")
                print(f"   1. Check if the model path exists: {model_identifier if isinstance(model_identifier, str) else 'N/A'}")
                print(f"   2. Ensure you ran 'python scripts/build.py' first")
            elif "tokenizer.chat_template is not set" in error_output:
                print(f"\n   ⚠️  Chat template not set in tokenizer")
                print(f"   This is a known issue with the current merge process.")
                print(f"\n   Solution: Run benchmark WITHOUT --apply_chat_template flag")
                print(f"   The benchmark will show a warning but will complete successfully.")
                print(f"\n   Note: This script has been updated to not use --apply_chat_template")
                print(f"   Please re-run: python scripts/benchmark.py")
            else:
                print(f"   1. Check if the model path exists")
                print(f"   2. Ensure you have enough VRAM (try reducing batch_size)")
                print(f"   3. Verify lm-eval is properly installed: pip install lm-eval")
                print(f"   4. For Ollama: ensure the model is loaded first")

            # Don't return None yet - check if partial results exist
            print(f"\n   Checking for partial results...")

        # Parse results from output JSON (even if process failed/was killed)
        # lm-eval creates subdirectories with sanitized model names
        results_dir = Path(benchmark_results_dir)

        # Find the most recent results.json file (may be in subdirectory)
        results_files = list(results_dir.rglob("results_*.json"))
        if not results_files:
            # Try older format
            results_file = results_dir / "results.json"
            if results_file.exists():
                results_files = [results_file]

        if results_files:
            # Get most recent file
            results_file = max(results_files, key=lambda p: p.stat().st_mtime)

            # If results are in a subdirectory, flatten the structure
            if results_file.parent != results_dir:
                import shutil
                print(f"\n📁 Flattening directory structure...")
                # Move all files from subdirectory to benchmark_results_dir
                for file in results_file.parent.glob("*"):
                    dest = results_dir / file.name
                    if not dest.exists():
                        shutil.move(str(file), str(dest))
                # Remove empty subdirectory
                try:
                    results_file.parent.rmdir()
                    # Update results_file path
                    results_file = results_dir / results_file.name
                except:
                    pass  # Directory not empty, that's ok

            print(f"\n📄 Loading results from: {results_file.relative_to(model_path)}")

            with open(results_file) as f:
                eval_results = json.load(f)

            # Extract relevant metrics
            benchmark_scores = {}
            for task in tasks:
                if task in eval_results.get("results", {}):
                    task_results = eval_results["results"][task]
                    benchmark_scores[task] = {}

                    # Extract metrics - lm-eval appends ,none to metric names
                    for key, value in task_results.items():
                        if isinstance(value, (int, float)) and not key.endswith("_stderr"):
                            # Remove ,none suffix and store
                            metric_name = key.replace(",none", "")
                            benchmark_scores[task][metric_name] = value

                            # Also store stderr if available
                            stderr_key = f"{key}_stderr"
                            if stderr_key in task_results:
                                stderr_val = task_results[stderr_key]
                                if isinstance(stderr_val, (int, float)):
                                    benchmark_scores[task][f"{metric_name}_stderr"] = stderr_val

                    # Set primary accuracy metric for compatibility
                    if "acc,none" in task_results:
                        benchmark_scores[task]["accuracy"] = task_results["acc,none"]
                    elif "prompt_level_strict_acc,none" in task_results:
                        # For ifeval, use prompt_level_strict_acc as primary
                        benchmark_scores[task]["accuracy"] = task_results["prompt_level_strict_acc,none"]
                    elif "exact_match,none" in task_results:
                        benchmark_scores[task]["accuracy"] = task_results["exact_match,none"]

            print(f"\n✅ Benchmark completed successfully!")
            print(f"   Tasks evaluated: {', '.join(benchmark_scores.keys())}")
            print(f"   Results saved to: {benchmark_results_dir}")
            return benchmark_scores, chat_template_info
        else:
            print(f"\n⚠️  Results file not found in {benchmark_results_dir}")
            print("   The benchmark may have failed silently.")
            return None, None

    except KeyboardInterrupt:
        print("\n\n❌ Benchmark interrupted by user (Ctrl+C)")
        print("   Partial results may not be saved.")
        return None, None
    except Exception as e:
        print(f"\n❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Run benchmark based on selected mode
if is_base_benchmark:
    # Benchmark base model
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARKING BASE MODEL: {base_model_name}")
    print(f"{'='*70}")

    # Only HuggingFace backend supported for base model
    backend = "huggingface"
    if backend not in backends_to_run:
        print(f"\n⚠️  Base model benchmarking requires HuggingFace backend")
        print("   Switching to HuggingFace backend...")
        backends_to_run = ["huggingface"]

    print(f"\n💡 Loading base model: {base_model_name}")
    scores, template_info = run_benchmark(backend, base_model_name, is_base_model=True)

    if scores:
        print(f"\n✅ Base model benchmark completed")
    else:
        print(f"\n❌ Base model benchmark failed")
        sys.exit(1)
else:
    # Benchmark fine-tuned model
    for backend in backends_to_run:
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARKING FINE-TUNED MODEL: {model_name}")
        print(f"{'='*70}")

        scores, template_info = run_benchmark(backend, model_name, is_base_model=False)
        if scores:
            print(f"\n✅ {backend.upper()} fine-tuned model benchmark completed")
        else:
            print(f"\n⚠️  {backend.upper()} fine-tuned model benchmark failed")

# Generate benchmark summary (combines base + latest fine-tuned)
generate_benchmark_summary(model_path, model_name, base_model_name)

print("\n✨ Benchmark complete!")
print("\n💡 Next steps:")
print("   - Check benchmarks/benchmark.json for combined results")
print("   - READMEs will auto-update to include these results")
print("   - Re-run 'python scripts/generate_readme_build.py' to update now")
