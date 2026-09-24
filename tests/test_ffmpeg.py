import os
import sys
from pathlib import Path

import pytest

from rvc.lib import ffmpeg
from rvc.lib.ffmpeg import ffmpeg_dll_folders, register_ffmpeg_dlls


def folder_with(tmp_path: Path, name: str, *files: str) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    for file in files:
        (folder / file).touch()
    return folder


def test_ffmpeg_dll_folders_finds_folders_with_ffmpeg_libraries_in_path_order(tmp_path: Path) -> None:
    static = folder_with(tmp_path, "static", "ffmpeg.exe")
    shared = folder_with(tmp_path, "shared", "ffmpeg.exe", "avcodec-63.dll", "avutil-61.dll")
    other = folder_with(tmp_path, "other", "avcodec-61.dll")
    search_path = os.pathsep.join([str(static), "", str(shared), str(tmp_path / "missing"), str(other), str(shared)])

    assert ffmpeg_dll_folders(search_path) == [shared, other]


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Pretend to be on Windows, recording the folders passed to os.add_dll_directory."""
    added: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising=False)
    monkeypatch.setattr(ffmpeg, "_registered", [])
    return added


def test_register_ffmpeg_dlls_registers_each_folder_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str]
) -> None:
    shared = folder_with(tmp_path, "shared", "avcodec-63.dll")
    monkeypatch.setenv("PATH", str(shared))

    assert register_ffmpeg_dlls() == [shared]
    assert register_ffmpeg_dlls() == [shared]
    assert windows == [str(shared)]


def test_register_ffmpeg_dlls_explains_a_missing_shared_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str]
) -> None:
    monkeypatch.setenv("PATH", str(folder_with(tmp_path, "static", "ffmpeg.exe")))

    with pytest.raises(FileNotFoundError, match="Gyan.FFmpeg.Shared"):
        register_ffmpeg_dlls()
    assert windows == []


def test_register_ffmpeg_dlls_does_nothing_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ffmpeg, "_registered", [])

    assert register_ffmpeg_dlls() == []
