"""
Pydantic models for type-safe training configuration.

This module defines the schema for training_params.yaml with validation rules.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# Valid optimizer options
OptimizerType = Literal["adamw_8bit", "adamw_torch", "sgd", "adafactor"]


class ModelConfig(BaseModel):
    """Model selection configuration."""

    base_model: str = Field(
        default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        description="Base model for LoRA training. Use quantized models (bnb-4bit) to save VRAM."
    )
    inference_model: Optional[str] = Field(
        default="unsloth/Llama-3.2-1B-Instruct",
        description="Unquantized model for GGUF conversion. Leave None to skip GGUF."
    )
    output_name: str = Field(
        default="auto",
        description="Output directory name. 'auto' generates from model+dataset names."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str = Field(
        default="GAIR/lima",
        description="HuggingFace dataset name or path."
    )
    max_samples: int = Field(
        default=0,
        ge=0,
        description="Limit dataset to N samples. 0 = use full dataset."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    rank: int = Field(
        default=64,
        gt=0,
        description="LoRA rank - higher values = more parameters to train. Common: 8, 16, 32, 64, 128"
    )
    alpha: int = Field(
        default=128,
        gt=0,
        description="LoRA alpha scaling factor. Often set to 2x rank. Affects learning rate."
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability for LoRA layers. 0.0 = no dropout."
    )
    use_rslora: bool = Field(
        default=False,
        description="Use Rank-Stabilized LoRA. Helps with training stability for high ranks."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class BatchConfig(BaseModel):
    """Batch size and gradient accumulation configuration."""

    size: int = Field(
        default=4,
        gt=0,
        description="Per-device batch size. Reduce if running out of VRAM."
    )
    gradient_accumulation_steps: int = Field(
        default=2,
        gt=0,
        description="Number of steps to accumulate gradients before updating."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.size * self.gradient_accumulation_steps


class OptimizationConfig(BaseModel):
    """Optimizer and learning rate configuration."""

    learning_rate: float = Field(
        default=3e-4,
        gt=0.0,
        description="Learning rate. Typical range: 1e-5 to 5e-4"
    )
    optimizer: OptimizerType = Field(
        default="adamw_8bit",
        description="Optimizer type. Valid options: adamw_8bit, adamw_torch, sgd, adafactor"
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Fraction of total steps used for warmup. 0.0 = no warmup."
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Absolute number of warmup steps. Overrides warmup_ratio if > 0."
    )
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm for clipping. Prevents exploding gradients."
    )
    use_gradient_checkpointing: bool = Field(
        default=True,
        description="Use gradient checkpointing to save VRAM. Slight speed penalty."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class EpochsConfig(BaseModel):
    """Training duration configuration."""

    num_train_epochs: int = Field(
        default=3,
        gt=0,
        description="Number of training epochs. Ignored if max_steps > 0."
    )
    max_steps: int = Field(
        default=0,
        ge=0,
        description="Maximum training steps. If > 0, overrides num_train_epochs."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class DataConfig(BaseModel):
    """Data processing configuration."""

    packing: bool = Field(
        default=False,
        description="Pack multiple sequences into one. Improves efficiency but changes loss calculation."
    )
    seed: int = Field(
        default=3407,
        description="Random seed for reproducibility."
    )
    max_seq_length: int = Field(
        default=2048,
        gt=0,
        description="Maximum sequence length. Must match model's context window."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    epochs: EpochsConfig = Field(default_factory=EpochsConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class LoggingConfig(BaseModel):
    """Logging and checkpoint configuration."""

    logging_steps: int = Field(
        default=5,
        gt=0,
        description="Log metrics every N steps."
    )
    save_steps: int = Field(
        default=25,
        gt=0,
        description="Save checkpoint every N steps."
    )
    save_total_limit: int = Field(
        default=2,
        ge=0,
        description="Maximum number of checkpoints to keep. 0 = keep all."
    )
    save_only_final: bool = Field(
        default=True,
        description="Only save final checkpoint. Overrides save_total_limit."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class OutputConfig(BaseModel):
    """Output format configuration."""

    formats: List[str] = Field(
        default=["gguf_f16", "gguf_q8_0", "gguf_q6_k", "gguf_q4_k_m"],
        description="Output formats to generate. Options: gguf_f16, gguf_q8_0, gguf_q6_k, gguf_q5_k_m, gguf_q4_k_m, gguf_q3_k_m, gguf_q2_k"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields

    @field_validator('formats')
    @classmethod
    def validate_formats(cls, v: List[str]) -> List[str]:
        """Validate output formats."""
        valid_formats = {
            "gguf_f16", "gguf_f32",
            "gguf_q8_0", "gguf_q6_k", "gguf_q5_k_m",
            "gguf_q4_k_m", "gguf_q3_k_m", "gguf_q2_k",
            "merged_16bit", "merged_4bit"
        }

        for fmt in v:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid format '{fmt}'. Valid formats: {', '.join(sorted(valid_formats))}"
                )

        return v


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    max_tokens: int = Field(
        default=512,
        gt=0,
        description="Maximum tokens to generate during benchmarking."
    )
    batch_size: int = Field(
        default=8,
        gt=0,
        description="Batch size for benchmarking."
    )
    default_backend: Literal["ollama", "vllm", "transformers"] = Field(
        default="transformers",
        description="Default backend for benchmarking. transformers = no server needed."
    )
    default_tasks: List[str] = Field(
        default=["ifeval", "gsm8k", "hellaswag"],
        description="Default lm-eval tasks: ifeval, gsm8k, hellaswag, mmlu, truthfulqa_mc, etc."
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields


class Config(BaseModel):
    """Root configuration object."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Raise error on unknown fields
        validate_assignment = True  # Validate when fields are modified


# Convenience function for backwards compatibility with old code
def get_config_value(config: Config, path: str, default=None):
    """
    Get nested config value using dot notation.

    Example:
        get_config_value(config, "training.lora.rank") -> 64
        get_config_value(config, "training.batch.size") -> 4
    """
    parts = path.split(".")
    value = config

    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        else:
            return default

    return value
