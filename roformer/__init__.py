"""Vendored BS-RoFormer model. No transformers dependency."""

import torch
from safetensors.torch import load_file

from .config import BSRoformerConfig
from .download import ensure_config, ensure_weights
from .model import BSRoformerForMaskedEstimation

_cached_model = None
_cached_device = None


def load_model(device: str = "cpu") -> BSRoformerForMaskedEstimation:
    """Load BS-RoFormer, downloading weights on first call. Cached after that."""
    global _cached_model, _cached_device
    if _cached_model is not None and _cached_device == device:
        return _cached_model

    config_dict = ensure_config()
    weights_path = ensure_weights()

    config = BSRoformerConfig.from_dict(config_dict)
    model = BSRoformerForMaskedEstimation(config)

    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    _cached_model = model
    _cached_device = device
    print(f"[SloppyAudio] BS-RoFormer loaded on {device} (46.8M params)")
    return model
