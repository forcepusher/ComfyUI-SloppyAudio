"""
SoX Audio Quality Effects for ComfyUI
Includes pitch, speed, reverb, echo, and correct gain control.
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


class AudioQualityEffects:
    """
    ComfyUI Node for applying audio effects (pitch, speed, reverb, echo, gain)
    to audio using an embedded SoX binary.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pitch_shift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -12.0,
                        "max": 12.0,
                        "step": 0.5,
                        "display": "slider",
                        "label": "Pitch Shift (semitones)",
                    },
                ),
                "speed_factor": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.05,
                        "display": "slider",
                        "label": "Speed Factor",
                    },
                ),
            },
            "optional": {
                "gain_db": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.5,
                        "display": "slider",
                        "label": "Gain (dB)",
                    },
                ),
                "use_limiter": ("BOOLEAN", {"default": True, "label": "Use Limiter for Gain"}),
                "normalize_audio": ("BOOLEAN", {"default": False, "label": "Normalize Audio"}),
                "add_reverb": ("BOOLEAN", {"default": False, "label": "Add Reverb"}),
                "reverb_amount": (
                    "FLOAT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 100,
                        "step": 5,
                        "display": "slider",
                        "label": "Reverb Amount",
                    },
                ),
                "reverb_room_scale": (
                    "FLOAT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 100,
                        "step": 5,
                        "display": "slider",
                        "label": "Room Size",
                    },
                ),
                "add_echo": ("BOOLEAN", {"default": False, "label": "Add Echo"}),
                "echo_delay": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                        "label": "Echo Delay (seconds)",
                    },
                ),
                "echo_decay": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.1,
                        "display": "slider",
                        "label": "Echo Decay",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "audio/effects"

    def process_audio(
        self,
        audio,
        pitch_shift=0.0,
        speed_factor=1.0,
        gain_db=0.0,
        use_limiter=True,
        normalize_audio=False,
        add_reverb=False,
        reverb_amount=50,
        reverb_room_scale=50,
        add_echo=False,
        echo_delay=0.5,
        echo_decay=0.5,
    ):
        if audio is None:
            print("No audio data to process")
            return (None,)

        if (
            abs(pitch_shift) < 0.01
            and abs(speed_factor - 1.0) < 0.01
            and abs(gain_db) < 0.01
            and not normalize_audio
            and not add_reverb
            and not add_echo
        ):
            print("No effects to apply, returning original audio")
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

            print(f"Processing audio with SoX")
            print(f"- Pitch shift: {pitch_shift} semitones")
            print(f"- Speed factor: {speed_factor}x")
            print(f"- Gain: {gain_db} dB")
            print(f"- Normalize: {normalize_audio}")
            if add_reverb:
                print(f"- Reverb: amount={reverb_amount}, room_scale={reverb_room_scale}")
            if add_echo:
                print(f"- Echo: delay={echo_delay}s, decay={echo_decay}")

            temp_dir = tempfile.mkdtemp(prefix="slopaudio_fx_")

            try:
                input_path = os.path.join(temp_dir, "input.wav")
                output_path = os.path.join(temp_dir, "output.wav")

                sf.write(input_path, audio_np, sample_rate)

                if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                    print("Failed to write input audio file")
                    return (audio,)

                sox_cmd = [sox_executable, input_path, output_path]
                effects = []

                if normalize_audio:
                    effects.extend(["gain", "-n"])

                if abs(gain_db) >= 0.01:
                    if gain_db > 0 and use_limiter:
                        effects.extend(["gain", "-l", str(gain_db)])
                    else:
                        effects.extend(["gain", str(gain_db)])

                if pitch_shift != 0:
                    pitch_cents = int(pitch_shift * 100)
                    effects.extend(["pitch", str(pitch_cents)])

                if speed_factor != 1.0:
                    effects.extend(["tempo", "-s", str(speed_factor)])

                if add_reverb:
                    effects.extend([
                        "reverb",
                        str(int(reverb_amount)),
                        "50",  # HF-damping
                        str(int(reverb_room_scale)),
                        "50",  # stereo depth
                        "20",  # pre-delay
                        "0",   # wet-gain
                    ])

                if add_echo:
                    delay_ms = int(echo_delay * 1000)
                    effects.extend([
                        "echo",
                        "0.8",            # gain-in
                        "0.9",            # gain-out
                        str(delay_ms),
                        str(echo_decay),
                    ])

                if effects:
                    sox_cmd.extend(effects)

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
            print(f"Error in audio processing: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)


NODE_CLASS_MAPPINGS = {
    "AudioQualityEffects": AudioQualityEffects,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioQualityEffects": "SlopAudio Effects",
}
