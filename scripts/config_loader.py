"""
Configuration loader for training parameters.

Loads and validates YAML configuration files using Pydantic schemas.
Also loads environment variables for credentials and paths.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import ValidationError

# Import from same directory when running scripts from project root
try:
    from config_schema import Config
except ImportError:
    from scripts.config_schema import Config


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load and validate training configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses training_params.yaml
                     in project root. Supports relative paths from project root.

    Returns:
        Validated Config object

    Raises:
        ConfigurationError: If config file is missing or invalid
    """
    # Determine project root (parent of scripts/)
    project_root = Path(__file__).parent.parent

    # Determine config file path
    if config_path is None:
        config_file = project_root / "training_params.yaml"
    else:
        config_file = Path(config_path)
        # If relative path, resolve from project root
        if not config_file.is_absolute():
            config_file = project_root / config_file

    # Check if file exists
    if not config_file.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_file}\n"
            f"Available configs in project root:\n"
            f"  - training_params.yaml (default production config)\n"
            f"  - quick_test.yaml (fast testing config)\n"
            f"\nCreate one of these files or specify a custom path."
        )

    # Load YAML file
    try:
        with open(config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML file: {config_file}\n"
            f"Error: {e}"
        )

    # Validate using Pydantic
    try:
        config = Config(**yaml_data)
    except ValidationError as e:
        # Format validation errors nicely
        error_messages = []
        for error in e.errors():
            field = " → ".join(str(x) for x in error['loc'])
            message = error['msg']
            error_messages.append(f"  • {field}: {message}")

        raise ConfigurationError(
            f"Invalid configuration in {config_file}:\n" +
            "\n".join(error_messages) +
            f"\n\nPlease fix the errors above and try again."
        )

    return config


def load_env_config():
    """
    Load environment variables for credentials, paths, and operational settings.

    Returns:
        Dictionary with environment configuration
    """
    # Load .env file
    load_dotenv()

    # Helper functions for type conversion
    def get_bool(key: str, default: bool = False) -> bool:
        val = os.getenv(key, str(default)).lower()
        return val in ('true', '1', 'yes', 'on')

    # Environment configuration (credentials, paths, operational flags only)
    # Model/dataset selection moved to YAML config
    env_config = {
        # Directory paths
        'output_dir_base': os.getenv('OUTPUT_DIR_BASE', './outputs'),
        'preprocessed_data_dir': os.getenv('PREPROCESSED_DATA_DIR', './data/preprocessed'),
        'cache_dir': os.getenv('CACHE_DIR', './cache'),

        # HuggingFace Hub
        'push_to_hub': get_bool('PUSH_TO_HUB', False),
        'hf_username': os.getenv('HF_USERNAME', ''),
        'hf_model_name': os.getenv('HF_MODEL_NAME', ''),
        'hf_token': os.getenv('HF_TOKEN', ''),

        # Weights & Biases
        'wandb_enabled': get_bool('WANDB_ENABLED', False),
        'wandb_project': os.getenv('WANDB_PROJECT', ''),
        'wandb_run_name': os.getenv('WANDB_RUN_NAME', ''),

        # Author attribution
        'author_name': os.getenv('AUTHOR_NAME', ''),

        # Operational flags
        'check_seq_length': get_bool('CHECK_SEQ_LENGTH', True),
        'force_preprocess': get_bool('FORCE_PREPROCESS', True),
        'force_retrain': get_bool('FORCE_RETRAIN', True),
        'force_rebuild': get_bool('FORCE_REBUILD', True),

        # Ollama configuration
        'ollama_base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
    }

    return env_config


def print_config_summary(config: Config, env_config: dict):
    """
    Print a human-readable summary of the loaded configuration.

    Args:
        config: Validated Config object
        env_config: Environment configuration dictionary
    """
    print("=" * 80)
    print("Configuration Loaded Successfully")
    print("=" * 80)

    print("\n📦 Model & Dataset:")
    print(f"  Base Model (Training):  {config.model.base_model}")
    print(f"  Base Model (Inference): {config.model.inference_model or 'None (skip GGUF)'}")
    print(f"  Dataset:                {config.dataset.name}")
    if config.dataset.max_samples > 0:
        print(f"  Max Samples:            {config.dataset.max_samples} (testing mode)")

    print("\n🎯 LoRA Configuration:")
    print(f"  Rank:        {config.training.lora.rank}")
    print(f"  Alpha:       {config.training.lora.alpha}")
    print(f"  Dropout:     {config.training.lora.dropout}")
    print(f"  Use RSLoRA:  {config.training.lora.use_rslora}")

    print("\n📊 Training Setup:")
    print(f"  Batch Size:             {config.training.batch.size}")
    print(f"  Gradient Accumulation:  {config.training.batch.gradient_accumulation_steps}")
    print(f"  Effective Batch Size:   {config.training.batch.effective_batch_size}")
    print(f"  Learning Rate:          {config.training.optimization.learning_rate}")
    print(f"  Optimizer:              {config.training.optimization.optimizer}")
    print(f"  Max Sequence Length:    {config.training.data.max_seq_length}")

    print("\n⏱️  Training Duration:")
    if config.training.epochs.max_steps > 0:
        print(f"  Max Steps:       {config.training.epochs.max_steps}")
    else:
        print(f"  Epochs:          {config.training.epochs.num_train_epochs}")
    # Dataset max samples now in dataset config, already printed above

    print("\n💾 Output Formats:")
    for fmt in config.output.formats:
        print(f"  • {fmt}")

    print("\n📁 Directories:")
    print(f"  Output:       {env_config['output_dir_base']}")
    print(f"  Preprocessed: {env_config['preprocessed_data_dir']}")
    print(f"  Cache:        {env_config['cache_dir']}")

    if env_config['push_to_hub']:
        print("\n🤗 HuggingFace Hub:")
        print(f"  Push to Hub:  Enabled")
        print(f"  Repository:   {env_config['hf_username']}/{env_config['hf_model_name']}")

    print("\n" + "=" * 80)


def get_config_for_script(config_path: Optional[str] = None, verbose: bool = True):
    """
    Convenience function to load both training config and env config.

    Args:
        config_path: Optional path to YAML config file
        verbose: Print configuration summary

    Returns:
        Tuple of (config, env_config)
    """
    try:
        config = load_config(config_path)
        env_config = load_env_config()

        if verbose:
            print_config_summary(config, env_config)

        return config, env_config

    except ConfigurationError as e:
        print(f"\n❌ Configuration Error:\n{e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error loading configuration:\n{e}\n", file=sys.stderr)
        sys.exit(1)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test configuration loader")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: training_params.yaml)"
    )
    args = parser.parse_args()

    # Load and print config
    config, env_config = get_config_for_script(args.config, verbose=True)

    print("\n✅ Configuration is valid!")
