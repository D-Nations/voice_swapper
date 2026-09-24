"""Make FFmpeg's shared libraries loadable by TorchCodec on Windows.

Python on Windows doesn't search the PATH for a DLL's dependencies, only folders registered with
os.add_dll_directory. TorchCodec's libraries depend on FFmpeg's DLLs, so the folders holding them
must be registered before TorchCodec decodes anything. Other systems find shared libraries on their own.
"""

import os
import sys
from pathlib import Path

FFMPEG_DLL = "avcodec-*.dll"

_registered: list[Path] = []


def ffmpeg_dll_folders(search_path: str | None = None) -> list[Path]:
    """Folders on the PATH (or search_path) that hold FFmpeg's shared libraries, in PATH order."""
    folders = []
    for entry in (search_path if search_path is not None else os.environ.get("PATH", "")).split(os.pathsep):
        folder = Path(entry)
        if entry and folder not in folders and folder.is_dir() and any(folder.glob(FFMPEG_DLL)):
            folders.append(folder)
    return folders


def register_ffmpeg_dlls() -> list[Path]:
    """On Windows, register the folders holding FFmpeg's DLLs so TorchCodec can load them. Returns them.

    Call before the first decode. Later calls return the same folders without registering again.
    """
    if sys.platform != "win32" or _registered:
        return list(_registered)
    folders = ffmpeg_dll_folders()
    if not folders:
        raise FileNotFoundError(
            "No FFmpeg shared libraries (avcodec-*.dll) on the PATH. TorchCodec needs FFmpeg's shared build, "
            "such as winget install Gyan.FFmpeg.Shared."
        )
    for folder in folders:
        os.add_dll_directory(str(folder))
        _registered.append(folder)
    return list(_registered)
