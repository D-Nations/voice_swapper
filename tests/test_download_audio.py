from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from yt_dlp import YoutubeDL

from youtube.download_audio import (
    download_audio,
    download_options,
    parse_timestamp,
    section_label,
    trim_to_section,
    whole_seconds,
)


def test_parse_timestamp_reads_seconds_minutes_and_hours() -> None:
    assert parse_timestamp("83") == 83.0
    assert parse_timestamp("1:23.5") == 83.5
    assert parse_timestamp("1:02:03") == 3723.0


@pytest.mark.parametrize("text", ["", "1:", "a:10", "1:2:3:4", "-5"])
def test_parse_timestamp_rejects_other_text(text: str) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        parse_timestamp(text)


def test_section_label_tells_sections_apart() -> None:
    assert section_label(None, None) == ""
    assert section_label(83.0, 91.5) == " 83-91.5s"
    assert section_label(None, 10.0) == " 0-10s"
    assert section_label(30.0, None) == " 30-end"


def test_whole_videos_are_downloaded_without_a_range(tmp_path: Path) -> None:
    options = download_options(tmp_path)

    assert "download_ranges" not in options
    assert options.get("outtmpl") == str(tmp_path / "%(title)s [%(id)s].%(ext)s")


def test_sections_are_cut_and_named_by_their_times(tmp_path: Path) -> None:
    options = download_options(tmp_path, start=83.0, end=91.5)

    assert "download_ranges" in options
    assert options.get("force_keyframes_at_cuts") is True
    assert options.get("outtmpl") == str(tmp_path / "%(title)s [%(id)s] 83-91.5s.%(ext)s")


def test_sections_are_downloaded_in_whole_seconds() -> None:
    ydl = YoutubeDL({"quiet": True})

    assert list(whole_seconds(83.4, 91.5)({}, ydl)) == [{"start_time": 83, "end_time": 92}]
    assert list(whole_seconds(30.0, None)({"duration": 120.2}, ydl)) == [{"start_time": 30, "end_time": 121}]
    with pytest.raises(ValueError, match="length is unknown"):
        list(whole_seconds(30.0, None)({}, ydl))


def test_trim_to_section_keeps_exactly_the_section(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    sample_rate = 1000
    # A download of seconds 83 to 92, with each sample holding its time in ms since 83 s.
    sf.write(path, np.arange(9 * sample_rate, dtype=np.int16), sample_rate, subtype="PCM_16")

    trim_to_section(path, 83.4, 91.5)

    audio, _ = sf.read(path, dtype="int16")
    assert len(audio) == 8100
    assert audio[0] == 400
    assert audio[-1] == 8499


def test_download_audio_rejects_an_end_before_the_start(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="after the start"):
        download_audio("https://www.youtube.com/watch?v=x", tmp_path, start=10.0, end=5.0)
