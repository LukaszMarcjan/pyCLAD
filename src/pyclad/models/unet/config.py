from typing import Optional

from pydantic import BaseModel


class UNetConfig(BaseModel):
    in_channels: int = 3
    out_channels: int = 3
    input_size: tuple[int, int] = (256, 256)

    backbone_name: Optional[str] = None
    backbone_return_nodes: Optional[tuple[str, ...]] = None
    pretrained_backbone: bool = False
    freeze_backbone: bool = False

    encoder_channels: tuple[int, ...] = (64, 128, 256, 512)
    decoder_channels: Optional[tuple[int, ...]] = None

    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    show_training_progress: bool = True
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    early_stopping_restore_best: bool = True

    reconstruction_loss: str = "mse"
    score_mode: str = "mse"

    output_activation: str = "sigmoid"

    threshold: Optional[float] = None
    threshold_quantile: float = 0.99

    normalize_mean: Optional[tuple[float, ...]] = None
    normalize_std: Optional[tuple[float, ...]] = None

    device: Optional[str] = None


class ReconstructionConfig(UNetConfig): ...
