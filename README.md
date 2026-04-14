# ComfyUI-SloppyAudio

SoX and BS-RoFormer nodes for ComfyUI. SoX for sound editing, BS-RoFormer for audio stem separation.

## Nodes

### SloppyAudio Separation
Splits audio into **vocals**, **drums**, **bass**, and **other** stems using [Mini-BS-RoFormer-V2-46.8M](https://huggingface.co/HiDolen/Mini-BS-RoFormer-V2-46.8M). Model auto-downloads from HuggingFace on first run. Connect only the stem outputs you need.

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

The BS-RoFormer model (~94 MB) downloads automatically from HuggingFace on first use and is cached by `transformers`.

## Requirements

- Python 3.10+
- PyTorch (ships with ComfyUI)
- `transformers >= 4.55.0`, `librosa`, `einops`, `soundfile`, `numpy`

## License

MIT
