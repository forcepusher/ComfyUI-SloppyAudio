"""
BS-RoFormer stem separation node.
Auto-downloads HiDolen/Mini-BS-RoFormer-V2-46.8M from HuggingFace on first use.
Outputs 4 stems: vocals, drums, bass, other.
"""

import torch
import numpy as np

from .roformer import load_model

MODEL_SR = 44100
STEM_ORDER = ["bass", "drums", "other", "vocals"]


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _resample(np_audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import librosa

    channels = []
    for ch in range(np_audio.shape[0]):
        channels.append(librosa.resample(np_audio[ch], orig_sr=orig_sr, target_sr=target_sr))
    return np.stack(channels)


class SloppyAudioStemSeparate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "batch_size": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "drums", "bass", "other")
    FUNCTION = "separate"
    CATEGORY = "audio/SloppyAudio"

    def separate(self, audio, device="auto", batch_size=2):
        if audio is None:
            empty = {"waveform": torch.zeros(1, 1, 1), "sample_rate": 44100}
            return (empty, empty, empty, empty)

        dev = _resolve_device(device)
        model = load_model(dev)

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        if waveform.dim() == 3:
            wav = waveform[0]
        elif waveform.dim() == 1:
            wav = waveform.unsqueeze(0)
        else:
            wav = waveform

        need_resample = sample_rate != MODEL_SR
        if need_resample:
            wav_np = _resample(wav.cpu().numpy(), sample_rate, MODEL_SR)
            wav = torch.tensor(wav_np, dtype=torch.float32)

        wav = wav.to(dev).float()

        print(f"[SloppyAudio] Separating: {wav.shape} @ {MODEL_SR}Hz on {dev}")
        stems = model.separate(wav, batch_size=batch_size, verbose=True)

        results = {}
        for i, name in enumerate(STEM_ORDER):
            stem = stems[i].cpu().float()
            if need_resample:
                stem_np = _resample(stem.numpy(), MODEL_SR, sample_rate)
                stem = torch.tensor(stem_np, dtype=torch.float32)
            results[name] = {
                "waveform": stem.unsqueeze(0),
                "sample_rate": sample_rate,
            }

        print("[SloppyAudio] Separation complete")
        return (results["vocals"], results["drums"], results["bass"], results["other"])


NODE_CLASS_MAPPINGS = {
    "SloppyAudioStemSeparate": SloppyAudioStemSeparate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SloppyAudioStemSeparate": "SloppyAudio Stem Separate",
}
