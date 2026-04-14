"""SoX fade-in / fade-out node."""

import os
import tempfile
import shutil
import subprocess
import soundfile as sf

from .sox_utils import ensure_sox
from .audio_utils import audio_to_numpy, numpy_to_audio

FADE_TYPES = ["linear", "quarter-sine", "half-sine", "logarithmic", "parabolic"]
FADE_CODE = {
    "linear": "t",
    "quarter-sine": "q",
    "half-sine": "h",
    "logarithmic": "l",
    "parabolic": "p",
}


class SloppyAudioFade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fade_in": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1},
                ),
                "fade_out": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1},
                ),
            },
            "optional": {
                "fade_type": (FADE_TYPES, {"default": "linear"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = "audio/SloppyAudio"

    def process(self, audio, fade_in=0.0, fade_out=0.0, fade_type="linear"):
        if audio is None:
            return (None,)
        if fade_in < 0.01 and fade_out < 0.01:
            return (audio,)

        sox = ensure_sox()
        data, sr = audio_to_numpy(audio)
        code = FADE_CODE.get(fade_type, "t")

        tmp = tempfile.mkdtemp(prefix="sloppy_fade_")
        try:
            inp = os.path.join(tmp, "in.wav")
            out = os.path.join(tmp, "out.wav")
            sf.write(inp, data, sr)

            cmd = [sox, inp, out, "fade", code, str(fade_in), "0", str(fade_out)]
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
    "SloppyAudioFade": SloppyAudioFade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SloppyAudioFade": "SloppyAudio Fade",
}
