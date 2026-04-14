"""Mix up to 4 audio inputs with per-input gain. Pure tensor math, no SoX."""

import torch


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


class SloppyAudioMerge:
    @classmethod
    def INPUT_TYPES(cls):
        gain = lambda default=0.0: ("FLOAT", {"default": default, "min": -60.0, "max": 24.0, "step": 0.5})
        return {
            "required": {
                "audio1": ("AUDIO",),
            },
            "optional": {
                "audio2": ("AUDIO",),
                "audio3": ("AUDIO",),
                "audio4": ("AUDIO",),
                "gain1_dB": gain(),
                "gain2_dB": gain(),
                "gain3_dB": gain(),
                "gain4_dB": gain(),
                "normalize": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "merge"
    CATEGORY = "audio/SloppyAudio"

    def merge(
        self,
        audio1,
        audio2=None,
        audio3=None,
        audio4=None,
        gain1_dB=0.0,
        gain2_dB=0.0,
        gain3_dB=0.0,
        gain4_dB=0.0,
        normalize=True,
    ):
        inputs = [
            (audio1, gain1_dB),
            (audio2, gain2_dB),
            (audio3, gain3_dB),
            (audio4, gain4_dB),
        ]
        active = [(a, g) for a, g in inputs if a is not None]

        if not active:
            return (audio1,)

        sample_rate = active[0][0]["sample_rate"]
        max_len = max(a["waveform"].shape[-1] for a, _ in active)
        max_ch = max(a["waveform"].shape[-2] for a, _ in active)

        # [batch, channels, samples] — batch dim always 1 for merge
        result = torch.zeros(1, max_ch, max_len)

        for audio, gain_db in active:
            w = audio["waveform"]
            if w.dim() == 1:
                w = w.unsqueeze(0).unsqueeze(0)
            elif w.dim() == 2:
                w = w.unsqueeze(0)

            # Broadcast mono → stereo if other inputs are stereo
            if w.shape[-2] == 1 and max_ch > 1:
                w = w.expand(-1, max_ch, -1)

            gain_lin = _db_to_linear(gain_db)
            result[..., : w.shape[-1]] += w[0].cpu() * gain_lin

        if normalize:
            peak = result.abs().max()
            if peak > 1.0:
                result = result / peak

        return ({"waveform": result, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "SloppyAudioMerge": SloppyAudioMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SloppyAudioMerge": "SloppyAudio Merge",
}
