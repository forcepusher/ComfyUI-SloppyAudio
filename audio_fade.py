"""
SoX Audio Fade In/Out Effect for ComfyUI.
Uses embedded SoX binary managed by sox_utils.
"""

import os
import tempfile
import numpy as np
import torch
import subprocess
import soundfile as sf
import shutil

from .sox_utils import ensure_sox


class AudioFadeEffect:
    """
    ComfyUI Node for applying fade-in and fade-out effects to audio using SoX.
    Supports multiple fade curve shapes (linear, quarter-sine, half-sine,
    logarithmic, parabolic).
    """

    FADE_TYPES = ["linear", "quarter-sine", "half-sine", "logarithmic", "parabolic"]
    FADE_TYPE_MAP = {
        "linear": "t",
        "quarter-sine": "q",
        "half-sine": "h",
        "logarithmic": "l",
        "parabolic": "p",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fade_in_duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                        "display": "slider",
                        "label": "Fade In Duration (seconds)",
                    },
                ),
                "fade_out_duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                        "display": "slider",
                        "label": "Fade Out Duration (seconds)",
                    },
                ),
            },
            "optional": {
                "fade_type": (cls.FADE_TYPES, {"default": "linear"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "audio/effects"

    def process_audio(
        self,
        audio,
        fade_in_duration=0.0,
        fade_out_duration=0.0,
        fade_type="linear",
    ):
        if audio is None:
            print("No audio data to process")
            return (None,)

        if fade_in_duration < 0.01 and fade_out_duration < 0.01:
            print("No fade effect to apply, returning original audio")
            return (audio,)

        sox_executable = ensure_sox()
        print(f"Using SoX executable: {sox_executable}")

        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            if waveform.dim() == 3:
                audio_np = waveform[0, 0].cpu().numpy()
            elif waveform.dim() == 2:
                audio_np = waveform[0].cpu().numpy()
            else:
                audio_np = waveform.cpu().numpy()

            fade_code = self.FADE_TYPE_MAP.get(fade_type, "t")

            print(f"Processing audio with SoX fade effect")
            print(f"- Fade type: {fade_type} ({fade_code})")
            print(f"- Fade in: {fade_in_duration}s")
            print(f"- Fade out: {fade_out_duration}s")

            temp_dir = tempfile.mkdtemp(prefix="slopaudio_fade_")

            try:
                input_path = os.path.join(temp_dir, "input.wav")
                output_path = os.path.join(temp_dir, "output.wav")

                sf.write(input_path, audio_np, sample_rate)

                if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                    print("Failed to write input audio file")
                    return (audio,)

                sox_cmd = [
                    sox_executable,
                    input_path,
                    output_path,
                    "fade",
                    fade_code,
                    str(fade_in_duration),
                    "0",
                    str(fade_out_duration),
                ]

                print("Executing:", " ".join(sox_cmd))

                process = subprocess.run(
                    sox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if process.stdout and process.stdout.strip():
                    print("SoX stdout:", process.stdout)
                if process.stderr and process.stderr.strip():
                    print("SoX stderr:", process.stderr)

                if process.returncode != 0:
                    print(f"SoX command failed with return code {process.returncode}")
                    return (audio,)

                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    print("SoX produced no output file")
                    return (audio,)

                processed_audio, new_sample_rate = sf.read(output_path)

                processed_tensor = torch.tensor(processed_audio.astype(np.float32))
                if processed_tensor.dim() == 1:
                    processed_tensor = processed_tensor.unsqueeze(0)
                if processed_tensor.dim() == 2:
                    processed_tensor = processed_tensor.unsqueeze(0)

                print(f"Processed audio shape: {processed_tensor.shape}")

                return ({
                    "waveform": processed_tensor,
                    "sample_rate": new_sample_rate,
                },)

            finally:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Error cleaning up temporary files: {e}")

        except Exception as e:
            print(f"Error in audio fade processing: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


NODE_CLASS_MAPPINGS = {
    "AudioFadeEffect": AudioFadeEffect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioFadeEffect": "SlopAudio Fade",
}
