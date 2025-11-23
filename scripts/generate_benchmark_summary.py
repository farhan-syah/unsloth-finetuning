"""
Generate Benchmark Summary JSON

This script combines base and latest fine-tuned benchmark results into a single
benchmarks/benchmark.json file. This is the single source of truth for benchmark data.

Usage:
    python scripts/generate_benchmark_summary.py [model_path]

    If model_path is not provided, will scan outputs/ directory and prompt for selection.

Output:
    - Reads from: benchmarks/base/lm-eval/ and benchmarks/fine-tuned/{timestamp}/lm-eval/
    - Writes to: benchmarks/benchmark.json
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

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
    print(f"\n📊 Generating benchmark summary for {model_name}...")

    # Find base and fine-tuned benchmarks
    base_dir = find_base_benchmark(model_path)
    finetuned_dir = find_latest_finetuned_benchmark(model_path)

    if not base_dir and not finetuned_dir:
        print(f"⚠️  No benchmark results found in {model_path}")
        return False

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
        print(f"   ✅ Loaded base benchmark from {base_dir}")
    else:
        print(f"   ⚠️  No base benchmark found")

    # Add fine-tuned results
    if finetuned_results:
        summary["fine_tuned_model"] = {
            "results": finetuned_results.get("results", {}),
            "benchmark_dir": finetuned_dir,
            "timestamp": os.path.getmtime(finetuned_dir)
        }
        print(f"   ✅ Loaded fine-tuned benchmark from {finetuned_dir}")
    else:
        print(f"   ⚠️  No fine-tuned benchmark found")

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

                # Print comparison
                delta_str = f"+{delta_pct:.1f}%" if delta_pct > 0 else f"{delta_pct:.1f}%"
                print(f"   📈 {task}: {base_acc:.4f} → {ft_acc:.4f} ({delta_str})")

        summary["comparison"] = comparison
        print(f"   ✅ Generated comparison")

    # Save summary
    output_file = os.path.join(model_path, "benchmarks", "benchmark.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"   📁 Saved to: {output_file}")
    return True

def scan_models():
    """Scan outputs directory for trained models"""
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return []

    models = []
    for model_dir in outputs_dir.iterdir():
        if model_dir.is_dir():
            # Check if it has a lora directory (trained model)
            lora_dir = model_dir / "lora"
            if lora_dir.exists():
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir)
                })

    return models

def get_base_model_name(model_path):
    """Get base model name from training_metrics.json"""
    metrics_path = os.path.join(model_path, "lora", "training_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
                return metrics.get("base_model_name", "Unknown")
        except:
            pass
    return "Unknown"

if __name__ == "__main__":
    print("="*70)
    print("📊 BENCHMARK SUMMARY GENERATOR")
    print("="*70)
    print("\nCombines base + latest fine-tuned benchmarks into benchmark.json\n")

    # Get model path from command line or scan
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        model_name = os.path.basename(model_path)
    else:
        # Scan for models
        print("📂 Scanning for trained models...")
        models = scan_models()

        if not models:
            print("❌ No trained models found in outputs/")
            sys.exit(1)

        print(f"✅ Found {len(models)} model(s):\n")
        for i, model in enumerate(models, 1):
            print(f"   {i}. {model['name']}")

        if len(models) == 1:
            selected = models[0]
            print(f"\n📦 Auto-selected: {selected['name']}")
        else:
            try:
                choice = input(f"\nSelect model (1-{len(models)}) [1]: ").strip() or "1"
                idx = int(choice) - 1
                if idx < 0 or idx >= len(models):
                    print("❌ Invalid selection")
                    sys.exit(1)
                selected = models[idx]
            except (ValueError, KeyboardInterrupt):
                print("\n❌ Cancelled")
                sys.exit(1)

        model_path = selected["path"]
        model_name = selected["name"]

    # Get base model name
    base_model_name = get_base_model_name(model_path)

    print(f"\n✅ Selected: {model_name}")
    print(f"   Base model: {base_model_name}")
    print(f"   Path: {model_path}")

    # Generate summary
    success = generate_benchmark_summary(model_path, model_name, base_model_name)

    if success:
        print("\n✨ Benchmark summary generated successfully!")
        print("\n💡 Next steps:")
        print("   - Run 'python scripts/generate_readme_build.py' to update READMEs")
        print("   - Check benchmarks/benchmark.json for combined results")
    else:
        print("\n❌ Failed to generate benchmark summary")
        sys.exit(1)
