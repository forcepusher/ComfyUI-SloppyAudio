"""Download and cache the BS-RoFormer model weights from HuggingFace."""

import json
import os
import sys
import urllib.request

MODEL_REPO = "HiDolen/Mini-BS-RoFormer-V2-46.8M"
_HF_BASE = f"https://huggingface.co/{MODEL_REPO}"
SAFETENSORS_URL = f"{_HF_BASE}/resolve/main/model.safetensors"
CONFIG_URL = f"{_HF_BASE}/raw/main/config.json"
SAFETENSORS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
_MODEL_SUBDIR = os.path.join("sloppyaudio", "bs-roformer-v2-46.8m")


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[SloppyAudio] Downloading BS-RoFormer: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()


def _download(url: str, dest: str) -> None:
    tmp = dest + ".tmp"
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_hook)
        print()
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _get_models_dir() -> str:
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, _MODEL_SUBDIR)
    except ImportError:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", _MODEL_SUBDIR)


def get_model_dir() -> str:
    d = _get_models_dir()
    os.makedirs(d, exist_ok=True)
    return d


def ensure_config() -> dict:
    cache = get_model_dir()
    path = os.path.join(cache, CONFIG_FILE)
    if not os.path.isfile(path):
        print(f"[SloppyAudio] Downloading config from {CONFIG_URL}")
        _download(CONFIG_URL, path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_weights() -> str:
    cache = get_model_dir()
    path = os.path.join(cache, SAFETENSORS_FILE)
    if os.path.isfile(path):
        return path
    print(f"[SloppyAudio] Model not cached. Downloading ~94 MB from HuggingFace...")
    _download(SAFETENSORS_URL, path)
    print(f"[SloppyAudio] Saved to {path}")
    return path
