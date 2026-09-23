import csv
from pathlib import Path

import pytest

from podcast.download_episodes import (
    Episode,
    apply_renames,
    clean_title,
    download,
    episode_filename,
    is_same_episode,
    missing_episodes,
    parse_feed,
    planned_renames,
    title_key,
)

FEED = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Episode 100: It's a Celebration</title>
    <pubDate>Tue, 01 Jan 2019 10:00:00 +0000</pubDate>
    <enclosure url="https://example.com/100.mp3" type="audio/mpeg"/>
  </item>
  <item>
    <title>Episode 2: The "Dangerous Truth" about Free Will</title>
    <pubDate>Sat, 01 Sep 2012 03:00:00 +0000</pubDate>
    <enclosure url="https://example.com/2.mp3" type="audio/mpeg"/>
  </item>
  <item><title>Announcement with no audio</title></item>
</channel></rss>"""


def test_parse_feed_reads_audio_items_oldest_first() -> None:
    episodes = parse_feed(FEED)

    assert [e.title for e in episodes] == [
        'Episode 2: The "Dangerous Truth" about Free Will',
        "Episode 100: It's a Celebration",
    ]
    assert episodes[0].published == "2012-09-01"
    assert episodes[0].audio_url == "https://example.com/2.mp3"


def test_episode_filename_matches_the_existing_style_and_is_windows_safe() -> None:
    assert episode_filename("Episode 341: Psychology Ranked (With Paul Bloom)") == (
        "Very Bad Wizards - Episode 341 - Psychology Ranked (With Paul Bloom).mp3"
    )
    assert episode_filename('Episode 2: The "Dangerous Truth" / Free Will?') == (
        "Very Bad Wizards - Episode 2 - The Dangerous Truth Free Will.mp3"
    )
    assert episode_filename("Episode 104: Smelling Salts for Morality: Our Top 3") == (
        "Very Bad Wizards - Episode 104 - Smelling Salts for Morality Our Top 3.mp3"
    )
    assert episode_filename("Bonus Episode: Top 5 Deadwood Characters") == (
        "Very Bad Wizards - Bonus Episode - Top 5 Deadwood Characters.mp3"
    )


def test_title_key_ignores_prefix_episode_number_and_punctuation() -> None:
    assert title_key("Very Bad Wizards - Episode 99 - It's a Celebration.mp3") == "itsacelebration"
    assert title_key("Episode 100: It's a Celebration") == "itsacelebration"
    assert title_key("Bonus Episode: Top 5 Deadwood Characters") == title_key(
        "Very Bad Wizards - Bonus Episode - Top 5 Deadwood Characters"
    )


def test_is_same_episode_tolerates_typos_and_added_words() -> None:
    assert is_same_episode(title_key("Oral Judgements"), title_key("Oral Judgments"))
    assert is_same_episode(
        title_key("Episode 120 - David Lynch's Mulholland Drive"),
        title_key('Episode 121: The Beauty of Illusion - David Lynch\'s "Mulholland Drive"'),
    )
    assert not is_same_episode(title_key("Red, Black, and Blue"), title_key("Mind the Gap"))
    assert not is_same_episode(
        title_key("Episode 156 - Notes From Underground (Pt. 2)"),
        title_key("Episode 156: Notes From Underground (Pt. 1)"),
    )


def test_missing_episodes_skips_files_already_downloaded_under_another_number(tmp_path: Path) -> None:
    (tmp_path / "Very Bad Wizards - Episode 99 - It's a Celebration.mp3").touch()

    missing = missing_episodes(parse_feed(FEED), tmp_path)

    assert [e.title for e in missing] == ['Episode 2: The "Dangerous Truth" about Free Will']


def test_download_writes_the_file_and_leaves_no_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio bytes")
    episode = Episode("Episode 1: Test", source.as_uri(), "2012-08-30")

    download(episode.audio_url, tmp_path / episode.filename)

    assert (tmp_path / episode.filename).read_bytes() == b"audio bytes"
    assert not list(tmp_path.glob("*.part"))


def test_planned_renames_uses_feed_titles_and_numbers(tmp_path: Path) -> None:
    (tmp_path / "Very Bad Wizards - Episode 99 - It's a Celebration.mp3").touch()
    (tmp_path / "Very Bad Wizards - Episode 2 - The Dangerous Truth about Free Will.mp3").touch()

    renames = planned_renames(parse_feed(FEED), tmp_path)

    assert [(a.name, b.name) for a, b in renames] == [
        (
            "Very Bad Wizards - Episode 99 - It's a Celebration.mp3",
            "Very Bad Wizards - Episode 100 - It's a Celebration.mp3",
        )
    ]


def test_planned_renames_refuses_two_files_for_one_episode(tmp_path: Path) -> None:
    (tmp_path / "Very Bad Wizards - Episode 99 - It's a Celebration.mp3").touch()
    (tmp_path / "Very Bad Wizards - Episode 98 - Its a Celebration.mp3").touch()

    with pytest.raises(ValueError):
        planned_renames(parse_feed(FEED), tmp_path)


def test_apply_renames_can_swap_names_and_logs_them(tmp_path: Path) -> None:
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    a.write_bytes(b"A")
    b.write_bytes(b"B")

    apply_renames([(a, b), (b, a)], tmp_path / "log.csv")

    assert a.read_bytes() == b"B"
    assert b.read_bytes() == b"A"
    with open(tmp_path / "log.csv", newline="") as file:
        assert list(csv.reader(file)) == [["old_name", "new_name"], ["a.mp3", "b.mp3"], ["b.mp3", "a.mp3"]]


def test_clean_title_fixes_feed_typos_and_guest_capitalization() -> None:
    assert clean_title("Episode 156: Notes From Underground (Pt. 1)") == "Episode 156: Notes from Underground (Pt. 1)"
    assert clean_title("Episode 234: Like A Dog (Kafka's Pt. 2)") == "Episode 234: Like a Dog (Kafka's Pt. 2)"
    assert (
        clean_title("Episode 209: Basic Instincts (With Paul Bloom)")
        == "Episode 209: Basic Instincts (with Paul Bloom)"
    )
    assert clean_title("Episode 1: Brains, Robots, and Free Will") == "Episode 1: Brains, Robots, and Free Will"
