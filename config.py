"""
Centralized configuration for the Latent Video Prediction Engine.

All hyperparameters, paths, and dataset settings are defined here.
Override via CLI arguments or environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    """Dataset generation and loading configuration."""
    dataset: Literal["wave", "balls", "fluid"] = "wave"
    resolution: int = 64
    channels: int = 1  # grayscale for wave equation
    num_sequences: int = 10000
    frames_per_seq: int = 20
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    data_dir: str = os.path.join(os.path.dirname(__file__), "data", "wave_64")
    seed: int = 42

    # Wave equation specific
    wave_speed: float = 1.0
    dt: float = 0.1
    boundary: Literal["reflecting", "absorbing"] = "reflecting"
    pulse_width_range: tuple[float, float] = (3.0, 8.0)
    pulse_amplitude_range: tuple[float, float] = (0.5, 1.0)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Latent dimension
    d_model: int = 256

    # CNN Encoder/Decoder
    encoder_channels: tuple[int, ...] = (32, 64, 128, 256)

    # Transformer
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1

    # Sequence lengths
    t_in: int = 10   # context frames
    t_out: int = 10  # predicted frames

    # Decoding mode (parallel vs autoregressive)
    decoding_mode: str = "parallel"  # "parallel" | "autoregressive"



@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

    # Schedule
    warmup_steps: int = 500
    epochs: int = 100

    # Batch
    batch_size: int = 8
    grad_accumulation_steps: int = 4
    num_workers: int = 0 if os.name == 'nt' else 2

    # Mixed precision
    use_amp: bool = True

    # Loss weights
    lambda_recon: float = 1.0
    lambda_latent: float = 1.0
    lambda_perceptual: float = 0.1
    lambda_temporal: float = 0.5

    # VICReg regularization (Phase 2)
    use_vicreg: bool = True
    lambda_var: float = 1.0        # Weight for variance term
    lambda_cov: float = 0.04       # Weight for off-diagonal covariance decorrelation
    lambda_inv: float = 0.5        # Weight for temporal invariance (smoothness)
    lambda_anchor: float = 1e-4    # Weight for L2 magnitude anchor


    # Logging
    log_interval: int = 50       # log every N steps
    eval_interval: int = 1       # evaluate every N epochs
    save_interval: int = 5       # checkpoint every N epochs
    num_eval_samples: int = 8    # GIFs to generate per eval

    # Paths
    checkpoint_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")
    log_dir: str = os.path.join(os.path.dirname(__file__), "runs")

    # Early stopping
    patience: int = 15  # epochs without improvement


@dataclass
class Config:
    """Top-level config combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


# Global default config
DEFAULT_CONFIG = Config()
