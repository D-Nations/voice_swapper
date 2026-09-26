"""Turn the soundboard's staging Markdown into the soundboard site (the soundboard/ submodule).

soundboard_staging/ holds one Markdown file per tab of the site, named NN-<tab id>.md so the numbers set the
tab order. The first file is the home page: its "# " heading is the site's title and its tab is called Home.
In each file:

    # TAM-9000                                   the tab's name
    HAL 9000's lines from *2001*...              the tab's intro, in Markdown

    ## "Open the pod bay doors, please."        a clip, labeled as shown on the page
    - audio: data/rvc/samples/.../podbay.wav     the WAV it comes from, relative to the project
    - voices: TamBot                             who is heard, from voices.py's labels
    - id: podbay                                 optional; otherwise made from the label
    Any other lines under a clip are a note shown with it.

Anything inside <!-- --> is ignored, so a clip can be switched off by commenting it out.

Each clip has the silence at both ends trimmed, is loudness-normalized so clips play at the same volume, and is
encoded as a mono MP3 into soundboard/clips/<tab id>/<clip id>.mp3. MP3s no longer listed are deleted.
soundboard/clips.json is rewritten, then the soundboard's build.py checks it and builds the site into
soundboard/_site/. Pushing the soundboard repo publishes it.

Run with: python -m voice_service.export_soundboard [--staging soundboard_staging] [--no-build]
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from diarization.select_clips import trim_silence
from voice_service.voices import VOICES

SOUNDBOARD = Path("soundboard")
STAGING = Path("soundboard_staging")
VOICE_NAMES = tuple(voice.label for voice in VOICES.values())
HOME_LABEL = "Home"

# Speech loudness for web audio, with headroom so peaks don't clip after encoding.
LOUDNESS_LUFS = -16.0
TRUE_PEAK_DB = -1.5
MP3_BITRATE = "96k"  # Mono speech. A 10 second clip is about 120 KB.
MP3_SAMPLE_RATE = 44100

FIELD = re.compile(r"^[-*]\s*(audio|voices|id)\s*:\s*(.*?)\s*$", re.IGNORECASE)
STAGING_NAME = re.compile(r"^\d+-(.+)\.md$")


@dataclass
class StagedClip:
    label: str
    audio: Path | None = None
    voices: list[str] = field(default_factory=list)
    id: str = ""
    note: str = ""


@dataclass
class StagedTab:
    id: str
    label: str
    intro: str = ""
    clips: list[StagedClip] = field(default_factory=list)
    source: Path | None = None


def slug(text: str, words: int = 6) -> str:
    """A short lowercase id from the first few words of text."""
    parts = re.findall(r"[a-z0-9]+", text.lower())[:words]
    return "-".join(parts) or "clip"


def parse_tab(path: Path) -> StagedTab:
    """Read one staging file. Missing pieces are left empty for check_staging to report."""
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    match = STAGING_NAME.match(path.name)
    tab = StagedTab(id=match.group(1) if match else path.stem, label="", source=path)
    intro: list[str] = []
    note: list[str] = []
    clip: StagedClip | None = None

    def finish_clip() -> None:
        if clip is not None:
            clip.note = "\n".join(note).strip()
            tab.clips.append(clip)

    for line in text.splitlines():
        if line.startswith("## "):
            finish_clip()
            clip, note = StagedClip(label=line[3:].strip()), []
        elif line.startswith("# ") and not tab.label and clip is None:
            tab.label = line[2:].strip()
        elif clip is None:
            intro.append(line)
        elif found := FIELD.match(line.strip()):
            key, value = found.group(1).lower(), found.group(2)
            if key == "audio":
                clip.audio = Path(value)
            elif key == "voices":
                clip.voices = [name.strip() for name in value.split(",") if name.strip()]
            else:
                clip.id = value
        else:
            note.append(line)
    finish_clip()
    tab.intro = "\n".join(intro).strip()
    # Clip ids default to the label's first words, made unique within the tab.
    used: set[str] = set()
    for c in tab.clips:
        base = c.id or slug(c.label)
        c.id, n = base, 2
        while c.id in used:
            c.id, n = f"{base}-{n}", n + 1
        used.add(c.id)
    return tab


def read_staging(folder: Path = STAGING) -> list[StagedTab]:
    """Every NN-<tab>.md file in folder, in order. Other files, like a README, are left alone."""
    return [parse_tab(path) for path in sorted(folder.glob("*.md")) if STAGING_NAME.match(path.name)]


def check_staging(tabs: list[StagedTab], voices: tuple[str, ...] = VOICE_NAMES) -> list[str]:
    """Everything that would stop an export. Empty when the staging files can be exported."""
    problems = []
    if not tabs:
        problems.append("There are no staging files.")
    seen: set[str] = set()
    for tab in tabs:
        where = tab.source.name if tab.source else tab.id
        if not tab.label:
            problems.append(f"{where} has no '# ' heading naming the tab.")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", tab.id) or tab.id == "index":
            problems.append(f"{where}: name the file NN-<id>.md with an id of lowercase letters, digits and dashes.")
        if tab.id in seen:
            problems.append(f"{where} uses the tab id {tab.id!r} again.")
        seen.add(tab.id)
        for clip in tab.clips:
            at = f"{where}, clip {clip.label!r}"
            if not clip.label:
                problems.append(f"{where} has a clip with no label after '## '.")
            if clip.audio is None:
                problems.append(f"{at} has no '- audio:' line.")
            elif not clip.audio.is_file():
                problems.append(f"{at}: {clip.audio} doesn't exist.")
            if not clip.voices:
                problems.append(f"{at} has no '- voices:' line.")
            for voice in clip.voices:
                if voice not in voices:
                    problems.append(f"{at} uses the unknown voice {voice!r}. Use {', '.join(voices)}.")
    return problems


def encode_mp3(source: Path, target: Path) -> None:
    """Trim silence from source, then loudness-normalize and encode it as a mono MP3 with FFmpeg."""
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    audio = trim_silence(audio.mean(axis=1), rate)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        trimmed = Path(folder) / "trimmed.wav"
        sf.write(trimmed, np.asarray(audio, dtype=np.float32), rate)
        subprocess.run(
            [
                *("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(trimmed)),
                *("-af", f"loudnorm=I={LOUDNESS_LUFS}:TP={TRUE_PEAK_DB}:LRA=11"),
                *("-ac", "1", "-ar", str(MP3_SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", MP3_BITRATE),
                # No embedded metadata, so the files carry nothing but the audio.
                *("-map_metadata", "-1", str(target)),
            ],
            check=True,
        )


def export(tabs: list[StagedTab], soundboard: Path = SOUNDBOARD) -> dict:
    """Encode every clip into the soundboard and rewrite its clips.json to list exactly these tabs and clips.

    The first tab is the home page: its heading becomes the site's title and the tab is labeled Home.
    """
    problems = check_staging(tabs)
    if problems:
        raise ValueError("Can't export:\n" + "\n".join(problems))
    manifest: dict = {"title": tabs[0].label, "tabs": []}
    listed: set[Path] = set()
    for n, tab in enumerate(tabs):
        entries = []
        for clip in tab.clips:
            file = f"clips/{tab.id}/{clip.id}.mp3"
            if clip.audio is None:  # check_staging reports this, so it can't happen here.
                raise ValueError(f"{clip.label} has no audio.")
            encode_mp3(clip.audio, soundboard / file)
            listed.add((soundboard / file).resolve())
            entries.append({"id": clip.id, "label": clip.label, "voices": clip.voices, "note": clip.note, "file": file})
        label = HOME_LABEL if n == 0 else tab.label
        manifest["tabs"].append({"id": tab.id, "label": label, "intro": tab.intro, "clips": entries})
    for old in (soundboard / "clips").rglob("*.mp3"):
        if old.resolve() not in listed:
            old.unlink()
    (soundboard / "clips.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", type=Path, default=STAGING, help="Folder of NN-<tab>.md staging files")
    parser.add_argument("--soundboard", type=Path, default=SOUNDBOARD, help="The soundboard repo's folder")
    parser.add_argument("--no-build", action="store_true", help="Only export the clips and clips.json")
    args = parser.parse_args(argv)

    try:
        manifest = export(read_staging(args.staging), args.soundboard)
    except ValueError as error:
        sys.exit(str(error))
    clips = [c for tab in manifest["tabs"] for c in tab["clips"]]
    total = sum((args.soundboard / c["file"]).stat().st_size for c in clips)
    print(
        f"Exported {len(manifest['tabs'])} tabs with {len(clips)} clips ({total / 1e6:.1f} MB) into {args.soundboard}."
    )
    if not args.no_build:
        subprocess.run([sys.executable, "build.py"], cwd=args.soundboard, check=True)


if __name__ == "__main__":
    main()
