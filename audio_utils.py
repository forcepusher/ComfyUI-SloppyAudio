"""Shared helpers for converting between ComfyUI AUDIO dicts and numpy/soundfile."""

import numpy as np
import torch


def audio_to_numpy(audio: dict) -> tuple[np.ndarray, int]:
    """
    Convert ComfyUI AUDIO dict → numpy array suitable for soundfile.write().
    Returns (data, sample_rate) where data is (samples,) mono or (samples, channels) stereo.
    """
    waveform = audio["waveform"]
    sr = audio["sample_rate"]

    if waveform.dim() == 3:
        wav = waveform[0]
    elif waveform.dim() == 1:
        wav = waveform.unsqueeze(0)
    else:
        wav = waveform

    arr = wav.cpu().numpy()

    if arr.shape[0] == 1:
        return arr[0], sr
    return arr.T, sr


def numpy_to_audio(data: np.ndarray, sample_rate: int) -> dict:
    """
    Convert numpy array from soundfile.read() → ComfyUI AUDIO dict.
    Accepts (samples,) mono or (samples, channels) stereo.
    """
    t = torch.tensor(data, dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(0)
    else:
        t = t.T.unsqueeze(0)
    return {"waveform": t, "sample_rate": sample_rate}
