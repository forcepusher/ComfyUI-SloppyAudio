"""
ComfyUI-SloppyAudio
BS-RoFormer stem separation + SoX audio effects for ComfyUI.
"""

from .audio_separate import NODE_CLASS_MAPPINGS as _sep_cls
from .audio_separate import NODE_DISPLAY_NAME_MAPPINGS as _sep_disp
from .audio_fade import NODE_CLASS_MAPPINGS as _fade_cls
from .audio_fade import NODE_DISPLAY_NAME_MAPPINGS as _fade_disp
from .audio_pitch import NODE_CLASS_MAPPINGS as _pitch_cls
from .audio_pitch import NODE_DISPLAY_NAME_MAPPINGS as _pitch_disp
from .audio_merge import NODE_CLASS_MAPPINGS as _merge_cls
from .audio_merge import NODE_DISPLAY_NAME_MAPPINGS as _merge_disp

NODE_CLASS_MAPPINGS = {**_sep_cls, **_fade_cls, **_pitch_cls, **_merge_cls}
NODE_DISPLAY_NAME_MAPPINGS = {**_sep_disp, **_fade_disp, **_pitch_disp, **_merge_disp}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
