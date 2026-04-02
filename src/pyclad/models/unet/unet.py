from __future__ import annotations

import copy
import inspect
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pyclad.models.model import Model
from pyclad.models.unet.config import ReconstructionConfig, UNetConfig
from pyclad.models.unet.loss import build_reconstruction_loss


def default_backbone_return_nodes(backbone_name: str) -> list[str]:
    defaults = {
        "resnet18": ["relu", "layer1", "layer2", "layer3", "layer4"],
        "resnet34": ["relu", "layer1", "layer2", "layer3", "layer4"],
        "resnet50": ["relu", "layer1", "layer2", "layer3", "layer4"],
        "wide_resnet50_2": ["relu", "layer1", "layer2", "layer3", "layer4"],
        "mobilenet_v2": ["features.1", "features.3", "features.6", "features.13", "features.18"],
    }
    if backbone_name not in defaults:
        raise ValueError(f"No default return nodes for '{backbone_name}'. Set backbone_return_nodes explicitly.")
    return defaults[backbone_name]


class _EncoderBase(nn.Module):
    @property
    def out_channels(self) -> tuple[int, ...]:
        raise NotImplementedError

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError


class _ScratchEncoder(_EncoderBase):
    def __init__(self, in_channels: int, channels: tuple[int, ...], block_cls: type[nn.Module]):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("encoder_channels must have at least 2 levels")

        stages = []
        prev_c = in_channels
        for out_c in channels:
            stages.append(block_cls(prev_c, out_c))
            prev_c = out_c

        self._stages = nn.ModuleList(stages)
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._out_channels = channels

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self._out_channels

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for idx, stage in enumerate(self._stages):
            x = stage(x)
            features.append(x)
            if idx < len(self._stages) - 1:
                x = self._pool(x)
        return features


class _TorchvisionEncoder(_EncoderBase):
    def __init__(
        self,
        backbone_name: str,
        in_channels: int,
        return_nodes: Optional[tuple[str, ...]],
        pretrained: bool,
        freeze: bool,
        input_size: tuple[int, int],
    ):
        super().__init__()

        if in_channels != 3:
            raise ValueError(
                "Torchvision backbones currently require in_channels=3. "
                "Use in_channels=3 in config or set backbone_name=None."
            )

        import torchvision.models as tv_models
        from torchvision.models.feature_extraction import create_feature_extractor

        model_fn = getattr(tv_models, backbone_name, None)
        if model_fn is None:
            raise ValueError(f"Unsupported backbone '{backbone_name}'")

        def resolve_weights():
            get_model_weights = getattr(tv_models, "get_model_weights", None)
            if get_model_weights is not None:
                return get_model_weights(model_fn).DEFAULT

            attr_name = f"{backbone_name}_weights".lower()
            for attr in dir(tv_models):
                if attr.lower() == attr_name:
                    enum_cls = getattr(tv_models, attr)
                    if hasattr(enum_cls, "DEFAULT"):
                        return enum_cls.DEFAULT
                    if hasattr(enum_cls, "IMAGENET1K_V1"):
                        return enum_cls.IMAGENET1K_V1
            return None

        params = inspect.signature(model_fn).parameters
        if "weights" in params:
            weights = resolve_weights() if pretrained else None
            backbone = model_fn(weights=weights)
        else:
            backbone = model_fn(pretrained=pretrained)

        used_nodes = list(return_nodes) if return_nodes is not None else default_backbone_return_nodes(backbone_name)
        self._nodes = used_nodes
        self._extractor = create_feature_extractor(
            model=backbone,
            return_nodes=OrderedDict((node, node) for node in used_nodes),
        )

        if freeze:
            for parameter in self._extractor.parameters():
                parameter.requires_grad = False

        self._out_channels = self._infer_out_channels(input_size=input_size)

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self._out_channels

    def _infer_out_channels(self, input_size: tuple[int, int]) -> tuple[int, ...]:
        was_training = self._extractor.training
        self._extractor.eval()
        with torch.no_grad():
            x = torch.zeros((1, 3, input_size[0], input_size[1]), dtype=torch.float32)
            features: Mapping[str, torch.Tensor] = self._extractor(x)
        self._extractor.train(was_training)

        channels = tuple(int(features[name].shape[1]) for name in self._nodes)
        if len(channels) < 2:
            raise ValueError("Backbone encoder must expose at least 2 feature levels")
        return channels

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self._extractor(x)
        return [feats[name] for name in self._nodes]


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.block = _DoubleConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _UNetArchitecture(nn.Module):
    def __init__(
        self,
        encoder: _EncoderBase,
        out_channels: int,
        output_activation: str,
        decoder_channels: Optional[tuple[int, ...]],
    ):
        super().__init__()
        self.encoder = encoder
        encoder_channels = encoder.out_channels
        if len(encoder_channels) < 2:
            raise ValueError("Encoder must expose at least 2 feature levels")

        if decoder_channels is None:
            used_decoder_channels = tuple(reversed(encoder_channels[:-1]))
        else:
            used_decoder_channels = decoder_channels

        if len(used_decoder_channels) != len(encoder_channels) - 1:
            raise ValueError(
                "decoder_channels length must be exactly len(encoder_channels) - 1 "
                f"({len(encoder_channels) - 1}), got {len(used_decoder_channels)}"
            )

        self.bottleneck = _DoubleConvBlock(encoder_channels[-1], encoder_channels[-1])

        decoder_stages = []
        current_channels = encoder_channels[-1]
        for skip_channels, stage_out_channels in zip(reversed(encoder_channels[:-1]), used_decoder_channels):
            decoder_stages.append(
                _DecoderStage(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=stage_out_channels,
                )
            )
            current_channels = stage_out_channels

        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.output_head = nn.Conv2d(current_channels, out_channels, kernel_size=1)
        self.output_activation = output_activation

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "identity":
            return x
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        raise ValueError(f"Unsupported output_activation='{self.output_activation}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder.forward_features(x)

        x = self.bottleneck(features[-1])
        for stage, skip in zip(self.decoder_stages, reversed(features[:-1])):
            x = stage(x, skip)

        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        x = self.output_head(x)
        return self._apply_output_activation(x)


def _build_encoder(config: UNetConfig) -> _EncoderBase:
    if config.backbone_name is None:
        return _ScratchEncoder(
            in_channels=config.in_channels,
            channels=config.encoder_channels,
            block_cls=_DoubleConvBlock,
        )

    return _TorchvisionEncoder(
        backbone_name=config.backbone_name,
        in_channels=config.in_channels,
        return_nodes=config.backbone_return_nodes,
        pretrained=config.pretrained_backbone,
        freeze=config.freeze_backbone,
        input_size=config.input_size,
    )


def build_architecture(config: UNetConfig) -> nn.Module:
    encoder = _build_encoder(config)
    return _UNetArchitecture(
        encoder=encoder,
        out_channels=config.out_channels,
        output_activation=config.output_activation,
        decoder_channels=config.decoder_channels,
    )


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _trainer_device_config(device: torch.device) -> tuple[str, int | list[int]]:
    if device.type == "cpu":
        return "cpu", 1
    if device.type == "cuda":
        if device.index is None:
            return "gpu", 1
        return "gpu", [device.index]
    if device.type == "mps":
        return "mps", 1
    return "auto", 1


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


_MONITOR_METRIC = "train_loss"


class _BestWeightsCallback(pl.Callback):
    def __init__(self, monitor: str, min_delta: float):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.best_state_dict: Optional[dict[str, torch.Tensor]] = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        current_loss = _to_float(trainer.callback_metrics.get(self.monitor))
        if current_loss is None:
            return
        if self.best_loss is None or (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.best_state_dict = copy.deepcopy(pl_module.network.state_dict())


class ImagePreprocessor:
    def __init__(
        self,
        input_size: tuple[int, int],
        in_channels: int,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
    ):
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self._input_size = input_size
        self._in_channels = in_channels
        self._normalize_mean = tuple(normalize_mean) if normalize_mean is not None else None
        self._normalize_std = tuple(normalize_std) if normalize_std is not None else None

        if (self._normalize_mean is None) != (self._normalize_std is None):
            raise ValueError("normalize_mean and normalize_std must be both set or both None")
        if self._normalize_mean is not None:
            if len(self._normalize_mean) != in_channels or len(self._normalize_std) != in_channels:
                raise ValueError(
                    f"normalize_mean/std length must match in_channels={in_channels}, got "
                    f"{len(self._normalize_mean)} and {len(self._normalize_std)}"
                )

    @staticmethod
    def _to_nchw(x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (NCHW or NHWC), got {x.shape}")

        if x.shape[1] in (1, 3):
            return x
        if x.shape[-1] in (1, 3):
            return np.transpose(x, (0, 3, 1, 2))

        raise ValueError(f"Cannot infer channel dimension from input shape {x.shape}")

    def _match_channels(self, x_t: torch.Tensor) -> torch.Tensor:
        channels = x_t.shape[1]
        if channels == self._in_channels:
            return x_t
        if channels == 1 and self._in_channels > 1:
            return x_t.repeat(1, self._in_channels, 1, 1)
        if channels > 1 and self._in_channels == 1:
            return x_t.mean(dim=1, keepdim=True)
        raise ValueError(f"Cannot convert from channels={channels} to in_channels={self._in_channels}")

    def transform(self, data: np.ndarray) -> torch.Tensor:
        x = np.asarray(data, dtype=np.float32)
        x = self._to_nchw(x)

        if x.size > 0 and float(x.max()) > 1.0:
            x = x / 255.0

        x_t = torch.from_numpy(x)
        x_t = self._match_channels(x_t)

        x_t = F.interpolate(x_t, size=self._input_size, mode="bilinear", align_corners=False)

        if self._normalize_mean is not None and self._normalize_std is not None:
            mean = torch.tensor(self._normalize_mean, dtype=x_t.dtype).view(1, -1, 1, 1)
            std = torch.tensor(self._normalize_std, dtype=x_t.dtype).view(1, -1, 1, 1)
            x_t = (x_t - mean) / std

        return x_t


class UNet(Model):
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        self.config = config or ReconstructionConfig()
        self._validate_config(self.config)

        self._device = _resolve_device(self.config.device)
        self._preprocessor = ImagePreprocessor(
            input_size=self.config.input_size,
            in_channels=self.config.in_channels,
            normalize_mean=self.config.normalize_mean,
            normalize_std=self.config.normalize_std,
        )

        network = build_architecture(self.config)
        self.module = UNetModule(
            network=network,
            reconstruction_loss=self.config.reconstruction_loss,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._threshold = self.config.threshold
        self._last_loss: Optional[float] = None

    @staticmethod
    def _validate_config(config: ReconstructionConfig) -> None:
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if config.threshold_quantile <= 0.0 or config.threshold_quantile >= 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if config.early_stopping_patience is not None and config.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0 or None")
        if config.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")

    def _prepare_batches(self, data: np.ndarray, shuffle: bool) -> DataLoader:
        x_t = self._preprocessor.transform(data)
        dataset = TensorDataset(x_t)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    def fit(self, data: np.ndarray):
        if len(data) == 0 or self.config.epochs == 0:
            return

        callbacks: list[pl.Callback] = []
        best_weights_callback: Optional[_BestWeightsCallback] = None

        if self.config.early_stopping_patience is not None:
            early_stopping_kwargs = {
                "monitor": _MONITOR_METRIC,
                "mode": "min",
                "patience": self.config.early_stopping_patience,
                "min_delta": float(self.config.early_stopping_min_delta),
            }
            if "check_on_train_epoch_end" in inspect.signature(EarlyStopping.__init__).parameters:
                early_stopping_kwargs["check_on_train_epoch_end"] = True
            callbacks.append(EarlyStopping(**early_stopping_kwargs))

            if self.config.early_stopping_restore_best:
                best_weights_callback = _BestWeightsCallback(
                    monitor=_MONITOR_METRIC,
                    min_delta=float(self.config.early_stopping_min_delta),
                )
                callbacks.append(best_weights_callback)

        accelerator, devices = _trainer_device_config(self._device)
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=self.config.show_training_progress,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )

        trainer.fit(self.module, train_dataloaders=self._prepare_batches(data, shuffle=True))
        self.module = self.module.to(self._device)
        self._last_loss = _to_float(trainer.callback_metrics.get(_MONITOR_METRIC))

        if (
            self.config.early_stopping_patience is not None
            and self.config.early_stopping_restore_best
            and best_weights_callback is not None
            and best_weights_callback.best_state_dict is not None
        ):
            self.module.network.load_state_dict(best_weights_callback.best_state_dict)

        if self.config.threshold is None:
            scores = self.score_data(data)
            if len(scores) == 0:
                self._threshold = 0.0
            else:
                self._threshold = float(np.quantile(scores, self.config.threshold_quantile))

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        self.module = self.module.to(self._device)
        self.module.eval()

        reconstructions: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_x,) in self._prepare_batches(data, shuffle=False):
                x = batch_x.to(self._device, dtype=torch.float32)
                x_hat = self.module(x)
                reconstructions.append(x_hat.detach().cpu().numpy().astype(np.float32, copy=False))

        if len(reconstructions) == 0:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(reconstructions, axis=0)

    def score_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return np.asarray([], dtype=np.float32)

        self.module = self.module.to(self._device)
        self.module.eval()

        scores: list[np.ndarray] = []
        with torch.no_grad():
            for (batch_x,) in self._prepare_batches(data, shuffle=False):
                x = batch_x.to(self._device, dtype=torch.float32)
                x_hat = self.module(x)
                batch_scores = self._batch_scores(x=x, x_hat=x_hat)
                scores.append(batch_scores.detach().cpu().numpy().astype(np.float32, copy=False))

        if len(scores) == 0:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(scores, axis=0)

    def _batch_scores(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if self.config.score_mode == "mse":
            err = (x_hat - x) ** 2
        elif self.config.score_mode == "mae":
            err = (x_hat - x).abs()
        else:
            raise ValueError("Unsupported score_mode. Use one of: 'mse', 'mae'.")
        return err.mean(dim=(1, 2, 3))

    def _resolve_threshold(self, scores: np.ndarray) -> float:
        if self.config.threshold is not None:
            return float(self.config.threshold)
        if self._threshold is not None:
            return float(self._threshold)
        if len(scores) == 0:
            return 0.0
        return float(np.quantile(scores, self.config.threshold_quantile))

    def predict(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        anomaly_scores = self.score_data(data)
        threshold = self._resolve_threshold(anomaly_scores)
        y_pred = (anomaly_scores > threshold).astype(int)
        return y_pred, anomaly_scores

    def name(self) -> str:
        return "UNet"

    def additional_info(self) -> Dict:
        return {
            "architecture_name": "unet",
            "backbone_name": self.config.backbone_name,
            "pretrained_backbone": self.config.pretrained_backbone,
            "freeze_backbone": self.config.freeze_backbone,
            "input_size": self.config.input_size,
            "in_channels": self.config.in_channels,
            "out_channels": self.config.out_channels,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "show_training_progress": self.config.show_training_progress,
            "early_stopping_patience": self.config.early_stopping_patience,
            "early_stopping_min_delta": self.config.early_stopping_min_delta,
            "early_stopping_restore_best": self.config.early_stopping_restore_best,
            "reconstruction_loss": self.config.reconstruction_loss,
            "score_mode": self.config.score_mode,
            "output_activation": self.config.output_activation,
            "threshold": self._threshold,
            "threshold_quantile": self.config.threshold_quantile,
            "device": str(self._device),
            "last_loss": self._last_loss,
            "uses_lightning": True,
        }


class UNetModule(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        reconstruction_loss: str,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        self.network = network
        self.loss_fn = build_reconstruction_loss(reconstruction_loss)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.save_hyperparameters(ignore=["network"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class ReconstructionModel(UNet):
    def name(self) -> str:
        return "ReconstructionModel"
