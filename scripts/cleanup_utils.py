"""
Cleanup utilities for removing unnecessary files from model directories.

This module provides smart cleanup functions that remove redundant files
while keeping only what's essential for each model format.
"""

import os
import shutil
from pathlib import Path


def cleanup_directory(directory, keep_patterns, description=""):
    """
    Remove files not matching keep patterns.

    Args:
        directory: Path to directory to clean
        keep_patterns: List of glob patterns for files to keep
        description: Optional description for logging

    Returns:
        Tuple of (removed_count, removed_size_bytes)
    """
    if not os.path.exists(directory):
        return 0, 0

    removed_count = 0
    removed_size = 0

    for item in Path(directory).iterdir():
        if item.is_file():
            # Check if file matches any keep pattern
            should_keep = any(item.match(pattern) for pattern in keep_patterns)

            if not should_keep:
                file_size = item.stat().st_size
                item.unlink()
                removed_count += 1
                removed_size += file_size
        elif item.is_dir() and item.name in ["unsloth_compiled_cache", "__pycache__"]:
            # Remove cache directories
            shutil.rmtree(item)
            removed_count += 1

    if removed_count > 0 and description:
        size_mb = removed_size / (1024 * 1024)
        print(f"   {description}: Removed {removed_count} item(s) ({size_mb:.1f} MB)")

    return removed_count, removed_size


def cleanup_lora_directory(lora_dir, verbose=True):
    """
    Clean up LoRA directory, keeping only essential adapter files.

    LoRA adapters only need:
    - adapter_model.safetensors (the actual LoRA weights)
    - adapter_config.json (LoRA configuration)
    - training_metrics.json (training statistics)
    - trainer_state.json (for README generation)
    - README.md (documentation)

    Everything else (tokenizer files, model configs, cache) is redundant
    because they're already in the base model.

    Args:
        lora_dir: Path to LoRA directory
        verbose: Print cleanup summary

    Returns:
        Tuple of (removed_count, removed_size_bytes)
    """
    keep_patterns = [
        "adapter_config.json",       # LoRA adapter configuration
        "adapter_model.safetensors", # LoRA weights
        "training_metrics.json",     # Training stats
        "trainer_state.json",        # Trainer state (for README generation)
        "README.md",                 # Documentation
        "*.md"                       # Keep any markdown files
    ]

    description = "LoRA cleanup" if verbose else ""
    return cleanup_directory(lora_dir, keep_patterns, description)


def cleanup_merged_directory(merged_dir, verbose=True):
    """
    Clean up merged model directory, keeping only model and tokenizer files.

    Merged models need:
    - model.safetensors (or model-*.safetensors for sharded models)
    - config.json, generation_config.json
    - tokenizer files (tokenizer.json, tokenizer_config.json, special_tokens_map.json)
    - Modelfile (for Ollama)
    - README.md (documentation)

    Everything else (cache, training artifacts) should be removed.

    Args:
        merged_dir: Path to merged model directory
        verbose: Print cleanup summary

    Returns:
        Tuple of (removed_count, removed_size_bytes)
    """
    keep_patterns = [
        "model.safetensors",         # Single file model
        "model-*.safetensors",       # Sharded model
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "Modelfile",                 # Ollama configuration
        "chat_template.json",        # Chat template info for README generation
        "README.md",
        "*.md"
    ]

    description = "Merged cleanup" if verbose else ""
    return cleanup_directory(merged_dir, keep_patterns, description)


def cleanup_gguf_directory(gguf_dir, verbose=True):
    """
    Clean up GGUF directory, keeping only GGUF files.

    GGUF files are self-contained with embedded tokenizer, so they only need:
    - *.gguf (the quantized model files)
    - Modelfile (for Ollama)
    - README.md (documentation)

    Tokenizer files are NOT needed because they're embedded in the GGUF file.

    Args:
        gguf_dir: Path to GGUF directory
        verbose: Print cleanup summary

    Returns:
        Tuple of (removed_count, removed_size_bytes)
    """
    keep_patterns = [
        "*.gguf",                    # GGUF model files
        "Modelfile",                 # Ollama configuration
        "README.md",
        "*.md"
    ]

    description = "GGUF cleanup" if verbose else ""
    return cleanup_directory(gguf_dir, keep_patterns, description)


def cleanup_all_outputs(output_dir, verbose=True):
    """
    Clean up all output directories (LoRA, merged, GGUF).

    Args:
        output_dir: Base output directory containing lora/, merged_16bit/, gguf/
        verbose: Print cleanup summary

    Returns:
        Dict with cleanup stats for each directory
    """
    stats = {}

    # Clean LoRA directory
    lora_dir = os.path.join(output_dir, "lora")
    if os.path.exists(lora_dir):
        count, size = cleanup_lora_directory(lora_dir, verbose)
        stats["lora"] = {"count": count, "size_mb": size / (1024 * 1024)}

    # Clean merged directory
    merged_dir = os.path.join(output_dir, "merged_16bit")
    if os.path.exists(merged_dir):
        count, size = cleanup_merged_directory(merged_dir, verbose)
        stats["merged"] = {"count": count, "size_mb": size / (1024 * 1024)}

    # Clean GGUF directory
    gguf_dir = os.path.join(output_dir, "gguf")
    if os.path.exists(gguf_dir):
        count, size = cleanup_gguf_directory(gguf_dir, verbose)
        stats["gguf"] = {"count": count, "size_mb": size / (1024 * 1024)}

    return stats
