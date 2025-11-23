#!/usr/bin/env python3
"""
Restore backed-up LoRA adapters from lora_bak/ directory.

This script allows you to interactively restore previously trained LoRA adapters
that were automatically backed up before retraining.

Usage:
    python scripts/restore_trained_data.py [model_path]

    If model_path is not provided, it will scan all models in ./outputs/
"""

import os
import sys
import json
import shutil
from datetime import datetime

def format_size(size_bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_directory_size(path):
    """Calculate total size of directory."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"⚠️  Error calculating size: {e}")
    return total_size

def load_backup_info(backup_path):
    """Load training metrics from backup directory."""
    metrics_path = os.path.join(backup_path, "training_metrics.json")
    if not os.path.exists(metrics_path):
        return None

    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not read metrics from {backup_path}: {e}")
        return None

def parse_backup_name(backup_name):
    """Extract info from backup directory name."""
    # Expected format: 20251123_202255_rank32_lr0.0003_loss1.4846
    parts = backup_name.split('_')
    info = {
        'date': 'unknown',
        'time': 'unknown',
        'rank': 'unknown',
        'lr': 'unknown',
        'loss': 'unknown'
    }

    if len(parts) >= 2:
        # Date and time
        try:
            date_str = parts[0]
            time_str = parts[1]
            # Format: 20251123 -> 2025-11-23
            if len(date_str) == 8:
                info['date'] = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
            # Format: 202255 -> 20:22:55
            if len(time_str) == 6:
                info['time'] = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        except:
            pass

    # Extract rank, lr, loss from remaining parts
    for part in parts[2:]:
        if part.startswith('rank'):
            info['rank'] = part.replace('rank', '')
        elif part.startswith('lr'):
            info['lr'] = part.replace('lr', '')
        elif part.startswith('loss'):
            info['loss'] = part.replace('loss', '')

    return info

def find_models_with_backups(base_dir='./outputs'):
    """Find all models that have backups."""
    models_with_backups = []

    if not os.path.exists(base_dir):
        return models_with_backups

    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        backup_path = os.path.join(model_path, 'lora_bak')

        if os.path.isdir(backup_path) and os.listdir(backup_path):
            models_with_backups.append((model_name, model_path))

    return models_with_backups

def list_backups(model_path):
    """List all available backups for a model."""
    backup_base = os.path.join(model_path, 'lora_bak')

    if not os.path.exists(backup_base):
        print(f"\n❌ No backup directory found at: {backup_base}")
        return []

    backups = []
    for backup_name in os.listdir(backup_base):
        backup_path = os.path.join(backup_base, backup_name)

        if not os.path.isdir(backup_path):
            continue

        # Check if it contains LoRA files
        adapter_config = os.path.join(backup_path, 'adapter_config.json')
        if not os.path.exists(adapter_config):
            continue

        # Load metrics
        metrics = load_backup_info(backup_path)

        # Parse backup name
        name_info = parse_backup_name(backup_name)

        # Get directory size
        size = get_directory_size(backup_path)

        backups.append({
            'name': backup_name,
            'path': backup_path,
            'metrics': metrics,
            'name_info': name_info,
            'size': size
        })

    # Sort by name (most recent first)
    backups.sort(key=lambda x: x['name'], reverse=True)

    return backups

def display_backups(backups):
    """Display list of backups in a nice format."""
    print("\n" + "="*80)
    print("📦 AVAILABLE BACKUPS")
    print("="*80)

    for i, backup in enumerate(backups, 1):
        metrics = backup['metrics']
        name_info = backup['name_info']

        print(f"\n{i}. {backup['name']}")
        print(f"   Date: {name_info['date']} {name_info['time']}")

        if metrics:
            print(f"   LoRA Rank: {metrics.get('lora_rank', 'unknown')}")
            print(f"   LoRA Alpha: {metrics.get('lora_alpha', 'unknown')}")
            print(f"   Learning Rate: {metrics.get('learning_rate', 'unknown')}")
            print(f"   Final Loss: {metrics.get('final_loss', 'unknown')}")
            print(f"   Total Steps: {metrics.get('total_steps', 'unknown')}")
            print(f"   Epochs: {metrics.get('num_train_epochs', 'unknown')}")
            print(f"   Dataset: {metrics.get('dataset_name', 'unknown')}")
            print(f"   Samples: {metrics.get('dataset_samples', 'unknown')}")
        else:
            print(f"   ⚠️  No metrics available")

        print(f"   Size: {format_size(backup['size'])}")

    print("\n" + "="*80)

def restore_backup(backup, model_path):
    """Restore a backup to the lora/ directory."""
    lora_dir = os.path.join(model_path, 'lora')

    # Check if current lora exists
    if os.path.exists(lora_dir):
        print(f"\n⚠️  Current LoRA adapters exist at: {lora_dir}")

        # Backup current lora first
        current_backup_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_current"
        current_backup_path = os.path.join(model_path, 'lora_bak', current_backup_name)

        print(f"   Creating backup of current LoRA...")
        print(f"   → {current_backup_path}")

        try:
            shutil.move(lora_dir, current_backup_path)
            print(f"   ✅ Current LoRA backed up")
        except Exception as e:
            print(f"\n❌ Failed to backup current LoRA: {e}")
            return False

    # Restore selected backup
    print(f"\n📦 Restoring backup: {backup['name']}")
    print(f"   From: {backup['path']}")
    print(f"   To: {lora_dir}")

    try:
        shutil.copytree(backup['path'], lora_dir)
        print(f"\n✅ Successfully restored backup!")

        # Display restored config
        if backup['metrics']:
            print(f"\n📊 Restored Configuration:")
            print(f"   LoRA Rank: {backup['metrics'].get('lora_rank', 'unknown')}")
            print(f"   LoRA Alpha: {backup['metrics'].get('lora_alpha', 'unknown')}")
            print(f"   Learning Rate: {backup['metrics'].get('learning_rate', 'unknown')}")
            print(f"   Final Loss: {backup['metrics'].get('final_loss', 'unknown')}")

        return True
    except Exception as e:
        print(f"\n❌ Failed to restore backup: {e}")
        return False

def main():
    """Main function."""
    print("="*80)
    print("🔄 RESTORE TRAINED DATA - LoRA Backup Manager")
    print("="*80)

    # Determine model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

        if not os.path.exists(model_path):
            print(f"\n❌ Model path does not exist: {model_path}")
            sys.exit(1)
    else:
        # Scan for models with backups
        print("\n🔍 Scanning for models with backups...")
        models = find_models_with_backups()

        if not models:
            print("\n❌ No models with backups found in ./outputs/")
            sys.exit(1)

        print(f"\n📁 Found {len(models)} model(s) with backups:")
        for i, (model_name, _) in enumerate(models, 1):
            print(f"   {i}. {model_name}")

        # Select model
        while True:
            choice = input(f"\nSelect model (1-{len(models)}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("\n👋 Exiting...")
                sys.exit(0)

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    model_path = models[idx][1]
                    print(f"\n✅ Selected: {models[idx][0]}")
                    break
                else:
                    print(f"⚠️  Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("⚠️  Please enter a valid number")

    # List backups
    backups = list_backups(model_path)

    if not backups:
        print(f"\n❌ No backups found for this model")
        sys.exit(1)

    # Display backups
    display_backups(backups)

    # Select backup to restore
    while True:
        choice = input(f"\nSelect backup to restore (1-{len(backups)}) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            print("\n👋 Exiting...")
            sys.exit(0)

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(backups):
                break
            else:
                print(f"⚠️  Please enter a number between 1 and {len(backups)}")
        except ValueError:
            print("⚠️  Please enter a valid number")

    selected_backup = backups[idx]

    # Confirm restoration
    print(f"\n⚠️  You are about to restore:")
    print(f"   Backup: {selected_backup['name']}")
    print(f"   Date: {selected_backup['name_info']['date']} {selected_backup['name_info']['time']}")

    confirm = input("\nProceed with restoration? (yes/no) [no]: ").strip().lower()

    if confirm != 'yes':
        print("\n❌ Restoration cancelled")
        sys.exit(0)

    # Restore
    success = restore_backup(selected_backup, model_path)

    if success:
        print("\n" + "="*80)
        print("✅ RESTORATION COMPLETE")
        print("="*80)
        print(f"\nYou can now use the restored LoRA adapters:")
        print(f"   python scripts/build.py")
    else:
        print("\n❌ Restoration failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
