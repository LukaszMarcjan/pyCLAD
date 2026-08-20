from typing import Literal

from pydantic import Field

from pyclad.vision.models.utilities.config import ImageSize, LightningVisionConfig


class RD4ADConfig(LightningVisionConfig):
    """Configuration for the RD4AD (Reverse Distillation from One-Class Embedding) detector."""

    input_size: ImageSize = (256, 256)
    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=200, ge=0)
    learning_rate: float = Field(default=5e-3, gt=0.0)

    backbone_name: Literal["resnet18", "resnet34", "resnet50", "wide_resnet50_2"] = "wide_resnet50_2"
    pretrained_encoder: bool = True
    freeze_encoder: bool = True

    adam_beta1: float = Field(default=0.5, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)

    # Gaussian blur applied to the fused anomaly map before scoring (0.0 disables it).
    # The reference implementation smooths with sigma=4 prior to both pixel-level evaluation
    # and image-level max-aggregation, so this is part of RD4AD's scoring, not cosmetics.
    # Sigma is expressed in anomaly-map pixels and the map is produced at ``input_size``:
    # 4.0 is calibrated for the default 256x256 input and gets proportionally stronger if
    # ``input_size`` is reduced.
    score_smoothing_sigma: float = Field(default=4.0, ge=0.0)
