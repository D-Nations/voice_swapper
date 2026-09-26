import json
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from tests.conftest import WavWriter
from voice_service.export_soundboard import check_staging, export, parse_tab, read_staging, slug

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg isn't installed")


def write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_tab_reads_the_heading_intro_clips_fields_and_notes(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "02-tam-9000.md",
        "# TAM-9000\n\nHAL's lines, in *TamBot's* voice.\n\n"
        '## "Open the pod bay doors, please."\n- audio: a.wav\n- voices: TamBot\nThe clean take.\n\n'
        "## Duet\n- Voices: DaveBot, TamBot\n- audio: b.wav\n- id: duo\n",
    )

    tab = parse_tab(path)

    assert (tab.id, tab.label, tab.intro) == ("tam-9000", "TAM-9000", "HAL's lines, in *TamBot's* voice.")
    assert [(c.id, c.label, c.voices, str(c.audio), c.note) for c in tab.clips] == [
        ("open-the-pod-bay-doors-please", '"Open the pod bay doors, please."', ["TamBot"], "a.wav", "The clean take."),
        ("duo", "Duet", ["DaveBot", "TamBot"], "b.wav", ""),
    ]


def test_parse_tab_ignores_comments_and_keeps_clip_ids_unique(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "03-casablanca.md",
        "# Casablanca\n\n## A hill of beans\n- audio: a.wav\n- voices: DaveBot\n\n"
        "<!--\n## Switched off\n- audio: x.wav\n- voices: TamBot\n-->\n\n"
        "## A hill of beans\n- audio: b.wav\n- voices: TamBot\n",
    )

    tab = parse_tab(path)

    assert [c.id for c in tab.clips] == ["a-hill-of-beans", "a-hill-of-beans-2"]


def test_slug_uses_the_first_words() -> None:
    assert slug("\"I'm sorry, Dave. I'm afraid I can't do that.\"") == "i-m-sorry-dave-i-m"


def test_check_staging_reports_each_kind_of_problem(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.touch()
    write(tmp_path / "staging", "01-home.md", "No heading here.\n")
    write(
        tmp_path / "staging",
        "02-Bad Name.md",
        f"# Tab\n\n## One\n- audio: {audio}\n- voices: HAL\n\n## Two\n- voices: DaveBot\n\n## Three\n- audio: missing.wav\n",
    )
    write(tmp_path / "staging", "README.md", "# Not a tab\n")

    tabs = read_staging(tmp_path / "staging")
    problems = " ".join(check_staging(tabs))

    assert [t.source.name for t in tabs if t.source] == ["01-home.md", "02-Bad Name.md"]
    for expected in (
        "no '# ' heading",
        "NN-<id>.md",
        "unknown voice 'HAL'",
        "no '- audio:' line",
        "doesn't exist",
        "no '- voices:' line",
    ):
        assert expected in problems


@needs_ffmpeg
def test_export_writes_mp3s_tabs_and_removes_stale_clips(tmp_path: Path, make_wav: WavWriter) -> None:
    source = make_wav("line.wav", sample_rate=40000, seconds=1.0)
    staging = tmp_path / "staging"
    write(staging, "01-home.md", "# Very Bad Robots\n\nWelcome.\n")
    write(staging, "02-dap-9000.md", f"# DAP-9000\n\n## A line\n- audio: {source}\n- voices: DaveBot, TamBot\n")
    soundboard = tmp_path / "soundboard"
    stale = soundboard / "clips" / "old" / "old.mp3"
    stale.parent.mkdir(parents=True)
    stale.touch()

    export(read_staging(staging), soundboard)

    manifest = json.loads((soundboard / "clips.json").read_text(encoding="utf-8"))
    assert manifest["title"] == "Very Bad Robots"
    assert [(t["id"], t["label"], t["intro"]) for t in manifest["tabs"]] == [
        ("home", "Home", "Welcome."),
        ("dap-9000", "DAP-9000", ""),
    ]
    assert manifest["tabs"][1]["clips"] == [
        {
            "id": "a-line",
            "label": "A line",
            "voices": ["DaveBot", "TamBot"],
            "note": "",
            "file": "clips/dap-9000/a-line.mp3",
        }
    ]
    mp3 = soundboard / "clips" / "dap-9000" / "a-line.mp3"
    assert sf.info(mp3).channels == 1
    assert 0.8 < sf.info(mp3).duration < 1.3
    assert not stale.exists()


def test_export_refuses_bad_staging_without_touching_the_soundboard(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    write(staging, "01-home.md", "# Home\n\n## Missing\n- audio: missing.wav\n- voices: DaveBot\n")
    soundboard = tmp_path / "soundboard"
    soundboard.mkdir()

    with pytest.raises(ValueError, match="doesn't exist"):
        export(read_staging(staging), soundboard)

    assert not (soundboard / "clips.json").exists()
