"""Download the audio of YouTube videos as WAV files, optionally just a section of each.

Files are named "<title> [<video id>].wav", or "<title> [<video id>] <start>-<end>s.wav" for a section.
A file that is already in the output folder is not downloaded again.
Needs ffmpeg on the PATH to extract and cut the audio.

Run with: python -m youtube.download_audio URL [URL ...] [--start 1:23] [--end 1:31] [--output-dir DIR]
"""

import argparse
import io
import math
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf
from yt_dlp import YoutubeDL

if TYPE_CHECKING:
    # Types from the yt-dlp stubs, which don't exist at run time.
    from yt_dlp import _DownloadRange, _YoutubeDLOptions

OUTPUT_DIR = Path("data/audio/youtube")

TIMESTAMP = re.compile(r"^(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d+)?)$")

# The second argument is the YoutubeDL instance, unused here. It is typed object because the yt-dlp stubs name
# both a module and a class YoutubeDL, so Pyright can't match a YoutubeDL annotation.
type RangeFunction = Callable[[Mapping[str, object], object], Iterator[_DownloadRange]]


def parse_timestamp(text: str) -> float:
    """Seconds from "SS", "MM:SS" or "HH:MM:SS", each with optional decimals on the seconds."""
    match = TIMESTAMP.match(text.strip())
    if match is None:
        raise ValueError(f"Not a timestamp: {text!r}. Use seconds, MM:SS or HH:MM:SS.")
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)


def section_label(start: float | None, end: float | None) -> str:
    """The part of a file name that tells sections of the same video apart, like " 83-91.5s"."""
    if start is None and end is None:
        return ""
    return f" {start or 0:g}-{'end' if end is None else f'{end:g}s'}"


def whole_seconds(start: float, end: float | None) -> RangeFunction:
    """A yt-dlp range covering [start, end] in whole seconds. end None means the end of the video."""

    def ranges(info: Mapping[str, object], _: object) -> Iterator[_DownloadRange]:
        last = end if end is not None else info.get("duration")
        if last is None:
            raise ValueError("The video's length is unknown, so give an --end.")
        if not isinstance(last, int | float):
            raise TypeError(f"Expected the video's duration in seconds, got {last!r}.")
        yield {"start_time": math.floor(start), "end_time": math.ceil(last)}

    return ranges


def download_options(output_dir: Path, start: float | None = None, end: float | None = None) -> _YoutubeDLOptions:
    """yt-dlp options for the best audio stream, converted to WAV, and cut around [start, end] if given."""
    options: _YoutubeDLOptions = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"%(title)s [%(id)s]{section_label(start, end)}.%(ext)s"),
        "windowsfilenames": True,
        "noplaylist": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "noprogress": True,
    }
    if start is not None or end is not None:
        options["download_ranges"] = whole_seconds(start or 0.0, end)
        # Re-encode at the cut points, so a section starts where asked instead of at the nearest keyframe.
        options["force_keyframes_at_cuts"] = True
    return options


def trim_to_section(path: Path, start: float, end: float | None) -> None:
    """Trim a WAV file that starts at floor(start) seconds to exactly [start, end]."""
    audio, sample_rate = sf.read(path)
    first = round((start - math.floor(start)) * sample_rate)
    last = first + round((end - start) * sample_rate) if end is not None else len(audio)
    sf.write(path, audio[first:last], sample_rate)


def download_audio(url: str, output_dir: Path, start: float | None = None, end: float | None = None) -> Path:
    """Download one video's audio, or the section [start, end] in seconds, and return the WAV file's path."""
    if start is not None and end is not None and end <= start:
        raise ValueError(f"The end ({end} s) must come after the start ({start} s).")
    output_dir.mkdir(parents=True, exist_ok=True)
    with YoutubeDL(download_options(output_dir, start, end)) as ydl:
        info = ydl.extract_info(url, download=False)
        path = Path(ydl.prepare_filename(info)).with_suffix(".wav")
        if path.exists():
            return path
        ydl.process_ie_result(info, download=True)
    if not path.exists():
        raise RuntimeError(f"yt-dlp did not save {path}.")
    if start is not None or end is not None:
        trim_to_section(path, start or 0.0, end)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("urls", nargs="+", help="YouTube video links")
    parser.add_argument("--start", type=parse_timestamp, help="Keep audio from here, as seconds, MM:SS or HH:MM:SS")
    parser.add_argument("--end", type=parse_timestamp, help="Keep audio up to here")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    # Titles can hold characters the Windows console's default encoding can't print, like the full-width
    # colons yt-dlp swaps in for ones that aren't allowed in file names.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for url in args.urls:
        print(f"Saved {download_audio(url, args.output_dir, args.start, args.end)}")


if __name__ == "__main__":
    main()
