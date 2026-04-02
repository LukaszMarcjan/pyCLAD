import torch.nn as nn


def build_reconstruction_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError("Unsupported reconstruction_loss. Use one of: 'mse', 'l1', 'smooth_l1'.")
