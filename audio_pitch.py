"""SoX pitch-shift node. Shifts pitch without changing tempo."""

import os
import tempfile
import shutil
import subprocess
import soundfile as sf

from .sox_utils import ensure_sox
from .audio_utils import audio_to_numpy, numpy_to_audio


class SloppyAudioPitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "semitones": (
                    "FLOAT",
                    {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = "audio/SloppyAudio"

    def process(self, audio, semitones=0.0):
        if audio is None:
            return (None,)
        if abs(semitones) < 0.01:
            return (audio,)

        sox = ensure_sox()
        data, sr = audio_to_numpy(audio)
        cents = int(semitones * 100)

        tmp = tempfile.mkdtemp(prefix="sloppy_pitch_")
        try:
            inp = os.path.join(tmp, "in.wav")
            out = os.path.join(tmp, "out.wav")
            sf.write(inp, data, sr)

            cmd = [sox, inp, out, "pitch", str(cents)]
            print(f"[SloppyAudio] {' '.join(cmd)}")

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[SloppyAudio] SoX error: {proc.stderr}")
                return (audio,)

            result_data, result_sr = sf.read(out)
            return (numpy_to_audio(result_data, result_sr),)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


NODE_CLASS_MAPPINGS = {
    "SloppyAudioPitch": SloppyAudioPitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SloppyAudioPitch": "SloppyAudio Pitch",
}
