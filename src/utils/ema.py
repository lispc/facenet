"""Exponential Moving Average (EMA) for model parameters."""

import torch


class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.

    Works with torch.compile and DistributedDataParallel because it stores
    references to the actual parameter tensors and operates on ``param.data``.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.shadow = [
            p.detach().clone() for p in model.parameters() if p.requires_grad
        ]
        self.backup: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Update EMA shadow parameters with current model parameters."""
        one_minus_decay = 1.0 - self.decay
        for shadow, param in zip(
            self.shadow, [p for p in model.parameters() if p.requires_grad]
        ):
            shadow.lerp_(param.data, one_minus_decay)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module):
        """Copy EMA shadow parameters into the model and keep a backup."""
        params = [p for p in model.parameters() if p.requires_grad]
        self.backup = [p.data.clone() for p in params]
        for shadow, param in zip(self.shadow, params):
            param.data.copy_(shadow)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        """Restore the original model parameters from the backup."""
        params = [p for p in model.parameters() if p.requires_grad]
        for backup, param in zip(self.backup, params):
            param.data.copy_(backup)
        self.backup = []

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
