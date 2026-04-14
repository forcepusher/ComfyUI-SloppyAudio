"""
Embedded SoX binary management.
Static binaries ship in bin/{win32,darwin,linux}/.
No downloads — everything is bundled.
"""

import os
import stat
import sys

_EXT_DIR = os.path.dirname(os.path.abspath(__file__))
_BIN_DIR = os.path.join(_EXT_DIR, "bin")

_cached_sox: str | None = None


def _platform_key() -> str:
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


_LINUX_SHARED_LIBS = [
    "libbz2.so.1.0",
    "libgomp.so.1",
    "libgsm.so.1",
    "libltdl.so.7",
    "liblzma.so.5",
    "libm.so.6",
    "libmagic.so.1",
    "libpng16.so.16",
    "libpthread.so.0",
    "libsox.so.3",
    "libz.so.1",
]


def _setup_linux_libs(plat_dir: str) -> None:
    prev = os.environ.get("LD_PRELOAD", "")
    paths = [prev] if prev else []
    for lib in _LINUX_SHARED_LIBS:
        p = os.path.join(plat_dir, lib)
        if os.path.isfile(p):
            paths.append(p)
    if paths:
        os.environ["LD_PRELOAD"] = os.pathsep.join(paths)


def _find_embedded() -> str | None:
    key = _platform_key()
    plat_dir = os.path.join(_BIN_DIR, key)
    exe = "sox.exe" if key == "win32" else "sox"
    candidate = os.path.join(plat_dir, exe)

    if not os.path.isfile(candidate):
        return None

    if key != "win32" and not os.access(candidate, os.X_OK):
        bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        os.chmod(candidate, os.stat(candidate).st_mode | bits)

    if key == "linux":
        _setup_linux_libs(plat_dir)

    return candidate


def ensure_sox() -> str:
    global _cached_sox
    if _cached_sox and os.path.isfile(_cached_sox):
        return _cached_sox

    sox = _find_embedded()
    if sox:
        _cached_sox = sox
        return sox

    expected = os.path.join(_BIN_DIR, _platform_key())
    raise RuntimeError(
        f"Embedded SoX binary not found at: {expected}\n"
        "The bin/ directory should ship with the extension. "
        "Re-clone or re-install to restore it."
    )
