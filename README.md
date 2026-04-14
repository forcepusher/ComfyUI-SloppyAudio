# ComfyUI-SloppyAudio

SoX and BS-RoFormer nodes for ComfyUI. SoX for sound editing, BS-RoFormer for audio stem separation.

## Nodes

### SloppyAudio Stem Separate
Splits audio into **vocals**, **drums**, **bass**, and **other** stems using [Mini-BS-RoFormer-V2-46.8M](https://huggingface.co/HiDolen/Mini-BS-RoFormer-V2-46.8M). Model auto-downloads from HuggingFace on first run (~94 MB, stored in `ComfyUI/models/sloppyaudio/`). Connect only the stem outputs you need.

### SloppyAudio Stem Merge
Mix up to 4 audio inputs back together with per-input gain control (dB). Auto-normalizes to prevent clipping. Handles mono/stereo and mismatched lengths.

### SloppyAudio Fade
Fade-in and fade-out using SoX. Supports linear, quarter-sine, half-sine, logarithmic, and parabolic curves.

### SloppyAudio Pitch
Pitch-shift in semitones using SoX. Changes pitch without altering tempo.

## Install

Clone into `ComfyUI/custom_nodes/`:

```
cd ComfyUI/custom_nodes
git clone https://github.com/forcepusher/ComfyUI-SloppyAudio.git
```

Dependencies install automatically via ComfyUI-Manager, or manually:

```
pip install -r requirements.txt
```

SoX binaries are embedded in `bin/` for Windows, macOS, and Linux — no separate SoX install required.
