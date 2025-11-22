#!/usr/bin/env python3
"""
Push trained models to HuggingFace Hub

This script uploads LoRA adapters, merged models, and/or GGUF models to HuggingFace.
It provides an interactive menu to choose what to push.

Usage:
    python scripts/push.py              # Interactive mode
    python scripts/push.py --lora       # Push only LoRA
    python scripts/push.py --merged     # Push only merged model
    python scripts/push.py --gguf       # Push only GGUF
    python scripts/push.py --all        # Push all formats
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
HF_USERNAME = os.getenv("HF_USERNAME", "")
OUTPUT_DIR_BASE = os.getenv("OUTPUT_DIR_BASE", "./outputs")
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "")

def check_hf_login():
    """Check if user is logged in to HuggingFace"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        return True
    except Exception:
        return False

def login_hf():
    """Login to HuggingFace"""
    from huggingface_hub import login

    # Try to get token from env
    hf_token = os.getenv("HF_TOKEN", "")

    if hf_token:
        try:
            login(token=hf_token)
            print("✅ Logged in using HF_TOKEN from .env")
            return True
        except Exception as e:
            print(f"⚠️  Failed to login with HF_TOKEN: {e}")

    # Interactive login
    print("\n🔐 Please login to HuggingFace")
    print("   Get your token from: https://huggingface.co/settings/tokens")
    try:
        login()
        return True
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def get_dir_size(path):
    """Get total size of directory in MB"""
    if not os.path.exists(path):
        return 0
    total_size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    return total_size / (1024 * 1024)  # Convert to MB

def count_files(path, pattern="*"):
    """Count files matching pattern in directory"""
    if not os.path.exists(path):
        return 0
    return len(list(Path(path).glob(pattern)))

def push_to_hub(local_dir, repo_id, folder_name, force=False):
    """Push directory to HuggingFace Hub"""
    from huggingface_hub import HfApi, create_repo

    # Check if directory exists
    if not os.path.exists(local_dir):
        print(f"❌ Directory not found: {local_dir}")
        return False

    # Get directory info
    dir_size = get_dir_size(local_dir)
    file_count = count_files(local_dir)

    print(f"\n📦 {folder_name}")
    print(f"   Local: {local_dir}")
    print(f"   Size: {dir_size:.1f} MB ({file_count} files)")
    print(f"   Repository: {repo_id}")

    if not force:
        response = input(f"\n⚠️  Push to {repo_id}? [y/N]: ").strip().lower()
        if response != 'y':
            print("   ⏭️  Skipped")
            return False

    try:
        api = HfApi()

        # Create repository
        print(f"\n📤 Creating repository...")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        print(f"   ✅ Repository created/verified")

        # Upload folder
        print(f"\n📤 Uploading {folder_name}...")
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {folder_name}"
        )

        print(f"   ✅ Upload complete!")
        print(f"   🔗 https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return False

def find_model_directories():
    """Find all model output directories"""
    models = []

    if not os.path.exists(OUTPUT_DIR_BASE):
        return models

    for model_dir in os.listdir(OUTPUT_DIR_BASE):
        model_path = os.path.join(OUTPUT_DIR_BASE, model_dir)
        if not os.path.isdir(model_path):
            continue

        # Check what formats exist
        has_lora = os.path.exists(os.path.join(model_path, "lora", "adapter_config.json"))
        has_merged = os.path.exists(os.path.join(model_path, "merged_16bit", "model.safetensors"))
        has_gguf = len(list(Path(os.path.join(model_path, "gguf")).glob("*.gguf"))) > 0 if os.path.exists(os.path.join(model_path, "gguf")) else False

        if has_lora or has_merged or has_gguf:
            models.append({
                'name': model_dir,
                'path': model_path,
                'has_lora': has_lora,
                'has_merged': has_merged,
                'has_gguf': has_gguf
            })

    return models

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Push models to HuggingFace Hub')
    parser.add_argument('--lora', action='store_true', help='Push LoRA adapters')
    parser.add_argument('--merged', action='store_true', help='Push merged model')
    parser.add_argument('--gguf', action='store_true', help='Push GGUF models')
    parser.add_argument('--all', action='store_true', help='Push all formats')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    args = parser.parse_args()

    # Validate HF_USERNAME
    if not HF_USERNAME:
        print("\n" + "="*60)
        print("❌ ERROR: HF_USERNAME not configured!")
        print("="*60)
        print("\nPlease set HF_USERNAME in your .env file:")
        print("   HF_USERNAME=your-hf-username")
        print("="*60 + "\n")
        sys.exit(1)

    # Check HuggingFace login
    if not check_hf_login():
        if not login_hf():
            print("\n❌ HuggingFace login required. Exiting.")
            sys.exit(1)

    print("\n" + "="*60)
    print("📤 Push Models to HuggingFace")
    print("="*60)

    # Find available models
    models = find_model_directories()

    if not models:
        print("\n❌ No trained models found in:", OUTPUT_DIR_BASE)
        print("\nRun 'python scripts/train.py' first to train a model.")
        sys.exit(1)

    # Interactive mode if no specific format specified
    if not (args.lora or args.merged or args.gguf or args.all):
        print(f"\n📦 Found {len(models)} trained model(s):\n")
        for i, m in enumerate(models, 1):
            formats = []
            if m['has_lora']: formats.append("LoRA")
            if m['has_merged']: formats.append("Merged")
            if m['has_gguf']: formats.append("GGUF")
            print(f"   {i}. {m['name']}")
            print(f"      Available: {', '.join(formats)}")

        print(f"\n   {len(models) + 1}. Exit")

        try:
            choice = int(input(f"\nSelect model [1-{len(models) + 1}]: ").strip())

            if choice < 1 or choice > len(models) + 1:
                print("Invalid choice.")
                sys.exit(1)

            if choice == len(models) + 1:
                print("Cancelled.")
                sys.exit(0)

            selected_model = models[choice - 1]

        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(1)

        # Ask which formats to push
        print(f"\nWhat to push for '{selected_model['name']}'?")
        options = []
        if selected_model['has_lora']:
            options.append(("LoRA adapters (~50-200MB)", "lora"))
        if selected_model['has_merged']:
            options.append(("Merged model (full size)", "merged"))
        if selected_model['has_gguf']:
            gguf_count = count_files(os.path.join(selected_model['path'], "gguf"), "*.gguf")
            options.append((f"GGUF models ({gguf_count} file(s))", "gguf"))

        for i, (desc, _) in enumerate(options, 1):
            print(f"   {i}. {desc}")
        print(f"   {len(options) + 1}. All formats")
        print(f"   {len(options) + 2}. Cancel")

        try:
            format_choice = int(input(f"\nEnter choice [1-{len(options) + 2}]: ").strip())

            if format_choice < 1 or format_choice > len(options) + 2:
                print("Invalid choice.")
                sys.exit(1)

            if format_choice == len(options) + 2:
                print("Cancelled.")
                sys.exit(0)

            if format_choice == len(options) + 1:
                push_lora = selected_model['has_lora']
                push_merged = selected_model['has_merged']
                push_gguf = selected_model['has_gguf']
            else:
                _, format_type = options[format_choice - 1]
                push_lora = format_type == "lora"
                push_merged = format_type == "merged"
                push_gguf = format_type == "gguf"

        except (ValueError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(1)
    else:
        # Command-line mode
        if len(models) > 1:
            print(f"\n⚠️  Multiple models found. Please select one:\n")
            for i, m in enumerate(models, 1):
                print(f"   {i}. {m['name']}")

            try:
                choice = int(input(f"\nSelect model [1-{len(models)}]: ").strip())
                if choice < 1 or choice > len(models):
                    print("Invalid choice.")
                    sys.exit(1)
                selected_model = models[choice - 1]
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled.")
                sys.exit(1)
        else:
            selected_model = models[0]

        push_lora = args.lora or args.all
        push_merged = args.merged or args.all
        push_gguf = args.gguf or args.all

    # Execute pushes
    model_name = selected_model['name']
    success_count = 0
    total_count = 0

    print("\n" + "="*60)
    print(f"📤 Pushing: {model_name}")
    print("="*60)

    # Push LoRA
    if push_lora and selected_model['has_lora']:
        total_count += 1
        lora_dir = os.path.join(selected_model['path'], "lora")
        lora_repo = f"{HF_USERNAME}/{model_name}-lora"
        if push_to_hub(lora_dir, lora_repo, "LoRA adapters", args.force):
            success_count += 1

    # Push Merged
    if push_merged and selected_model['has_merged']:
        total_count += 1
        merged_dir = os.path.join(selected_model['path'], "merged_16bit")
        merged_repo = f"{HF_USERNAME}/{model_name}"
        if push_to_hub(merged_dir, merged_repo, "Merged model", args.force):
            success_count += 1

    # Push GGUF
    if push_gguf and selected_model['has_gguf']:
        total_count += 1
        gguf_dir = os.path.join(selected_model['path'], "gguf")
        gguf_repo = f"{HF_USERNAME}/{model_name}-GGUF"
        if push_to_hub(gguf_dir, gguf_repo, "GGUF models", args.force):
            success_count += 1

    # Summary
    print("\n" + "="*60)
    print("📊 Push Summary")
    print("="*60)
    print(f"   Successful: {success_count}/{total_count}")

    if success_count > 0:
        print(f"\n✅ Models uploaded successfully!")
        print(f"\n📍 Your models:")
        if push_lora and selected_model['has_lora']:
            print(f"   • LoRA: https://huggingface.co/{HF_USERNAME}/{model_name}-lora")
        if push_merged and selected_model['has_merged']:
            print(f"   • Merged: https://huggingface.co/{HF_USERNAME}/{model_name}")
        if push_gguf and selected_model['has_gguf']:
            print(f"   • GGUF: https://huggingface.co/{HF_USERNAME}/{model_name}-GGUF")

    print("="*60 + "\n")

if __name__ == "__main__":
    main()
