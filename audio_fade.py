"""
SoX Audio Fade In/Out Effect for ComfyUI
Applies fade-in and fade-out envelopes using the SoX 'fade' effect.
Cross-platform compatible for Windows and Linux/WSL.
"""

import os
import tempfile
import numpy as np
import torch
import subprocess
import soundfile as sf
import shutil
import platform


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
        if platform.system() == "Windows":
            default_sox_path = "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe"
        else:
            default_sox_path = "sox"

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
                "sox_path": ("STRING", {"default": default_sox_path}),
                "fade_type": (cls.FADE_TYPES, {"default": "linear"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "audio/effects"

    def find_sox_executable(self, provided_path):
        """Find the SoX executable on different platforms."""
        if os.path.exists(provided_path) and os.access(provided_path, os.X_OK):
            return provided_path

        if platform.system() != "Windows":
            try:
                sox_path = subprocess.check_output(
                    ["which", "sox"], text=True
                ).strip()
                if sox_path:
                    print(f"Found SoX at: {sox_path}")
                    return sox_path
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            linux_paths = ["/usr/bin/sox", "/usr/local/bin/sox", "/bin/sox"]
            for path in linux_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    print(f"Found SoX at: {path}")
                    return path

            if provided_path == "sox":
                return provided_path
        else:
            windows_paths = [
                "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe",
                "C:\\Program Files\\sox-14-4-2\\sox.exe",
                "C:\\Program Files (x86)\\sox-14.4.2\\sox.exe",
                "C:\\Program Files\\sox-14.4.2\\sox.exe",
                "C:\\Program Files (x86)\\sox\\sox.exe",
                "C:\\Program Files\\sox\\sox.exe",
            ]
            for path in windows_paths:
                if os.path.exists(path):
                    print(f"Found SoX at: {path}")
                    return path

        return None

    def process_audio(
        self,
        audio,
        fade_in_duration=0.0,
        fade_out_duration=0.0,
        sox_path=None,
        fade_type="linear",
    ):
        """Apply fade-in and/or fade-out to audio using SoX."""
        if audio is None:
            print("No audio data to process")
            return (None,)

        if fade_in_duration < 0.01 and fade_out_duration < 0.01:
            print("No fade effect to apply, returning original audio")
            return (audio,)

        if sox_path is None:
            if platform.system() == "Windows":
                sox_path = "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe"
            else:
                sox_path = "sox"

        print(f"Trying to find SoX at: {sox_path}")
        sox_executable = self.find_sox_executable(sox_path)

        if not sox_executable:
            print("SoX executable not found. Please install SoX:")
            if platform.system() == "Windows":
                print(
                    "- Windows: Download and install from https://sourceforge.net/projects/sox/"
                )
                print("- Then provide the correct path to sox.exe")
            else:
                print("- Linux/WSL: Run 'sudo apt-get install sox'")
            return (audio,)

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

            temp_dir = tempfile.mkdtemp(prefix="sox_fade_")

            try:
                input_path = os.path.join(temp_dir, "input.wav")
                output_path = os.path.join(temp_dir, "output.wav")

                print(f"Writing input audio to {input_path}")
                sf.write(input_path, audio_np, sample_rate)

                if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                    print("Failed to write input audio file")
                    return (audio,)

                # SoX fade syntax: fade [type] fade-in-length [stop-position [fade-out-length]]
                # stop-position of 0 means process to end of file
                sox_cmd = [sox_executable, input_path, output_path]
                sox_cmd.extend([
                    "fade",
                    fade_code,
                    str(fade_in_duration),
                    "0",
                    str(fade_out_duration),
                ])

                print("Executing:", " ".join(sox_cmd))

                process = subprocess.run(
                    sox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if process.stdout and len(process.stdout.strip()) > 0:
                    print("SoX stdout:", process.stdout)
                if process.stderr and len(process.stderr.strip()) > 0:
                    print("SoX stderr:", process.stderr)

                if process.returncode != 0:
                    print(
                        f"SoX command failed with return code {process.returncode}"
                    )
                    return (audio,)

                if not os.path.exists(output_path):
                    print("Output file does not exist")
                    return (audio,)

                if os.path.getsize(output_path) == 0:
                    print("Output file is empty")
                    return (audio,)

                print(f"Reading processed audio from {output_path}")

                processed_audio, new_sample_rate = sf.read(output_path)

                processed_tensor = torch.tensor(processed_audio.astype(np.float32))

                if processed_tensor.dim() == 1:
                    processed_tensor = processed_tensor.unsqueeze(0)
                if processed_tensor.dim() == 2:
                    processed_tensor = processed_tensor.unsqueeze(0)

                print(f"Processed audio shape: {processed_tensor.shape}")

                result_audio = {
                    "waveform": processed_tensor,
                    "sample_rate": new_sample_rate,
                }

                return (result_audio,)

            finally:
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
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
    "AudioFadeEffect": "Audio Fade In/Out",
}
