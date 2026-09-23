"""Download Very Bad Wizards episodes that are not already in the audio folder.

Episodes come from the podcast's RSS feed. verybadwizards.com/episodes is built from this same
feed and links to the same MP3 files, but the feed lists every episode in one structured file.

New files are named like the existing ones, for example
"Very Bad Wizards - Episode 341 - Psychology Ranked (with Paul Bloom).mp3", and are kept as MP3.
Everything downstream reads MP3 directly, and converting to WAV or FLAC would make the files
about ten times larger without restoring any quality.

Existing files are matched to feed episodes by title rather than number, because the episode
numbers in many existing file names differ from the official ones.

Run with: python -m podcast.download_episodes [--dry-run] [--limit N] [--rename-existing]
"""

import argparse
import csv
import difflib
import io
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

from tqdm import tqdm

FEED_URL = "https://feeds.libsyn.com/474285/rss"
OUTPUT_DIR = Path("data/audio/very_bad_wizards")
FILENAME_PREFIX = "Very Bad Wizards - "
USER_AGENT = "voice-swapper-episode-downloader/1.0"
CHUNK_BYTES = 1 << 20
RENAME_LOG = "rename_log.csv"  # Written in the output folder by --rename-existing.

# Titles at least this similar count as the same episode, which absorbs typos in old file names.
FUZZY_MATCH_CUTOFF = 0.9
# An old title contained in a new one also counts as a match if it is at least this many characters,
# which catches episodes that were retitled by adding words.
MIN_CONTAINED_KEY_LENGTH = 12

# Fixes for typos and inconsistent capitalization in the feed's own titles, applied to every
# title before it becomes a file name. Each pair is (text in the feed, replacement).
TITLE_CORRECTIONS = (
    ("Like A Dog", "Like a Dog"),
    ("If We Only Had A Brain", "If We Only Had a Brain"),
    ("Schooled By Our Listeners", "Schooled by Our Listeners"),
    ("Wizards With (Reactive)", "Wizards with (Reactive)"),
    ("a Word For This Title", "a Word for This Title"),
    ("Notes From Underground", "Notes from Underground"),
    ("Life With No Head", "Life with No Head"),
    ("What Is It Like To Be", "What Is It Like to Be"),
    ("What is it Like to be a Bat", "What Is It Like to Be a Bat"),
    ("Laughs With You", "Laughs with You"),
    ("Gods With Anuses", "Gods with Anuses"),
    ("Civilization and its Discontents", "Civilization and Its Discontents"),
    ("The Gray Man who Dreamed", "The Gray Man Who Dreamed"),
)


@dataclass(frozen=True)
class Episode:
    title: str
    audio_url: str
    published: str  # ISO date, YYYY-MM-DD

    @property
    def filename(self) -> str:
        return episode_filename(self.title)


def clean_title(title: str) -> str:
    """Apply TITLE_CORRECTIONS and write guest credits as "(with Guest)", the feed's usual style."""
    for wrong, right in TITLE_CORRECTIONS:
        title = title.replace(wrong, right)
    return re.sub(r"\(With ", "(with ", title)


def episode_filename(title: str) -> str:
    """Turn a feed title into a file name in the existing style, safe on Windows.

    Like the existing files, the first colon becomes " - " and any later colons are dropped, so
    "Episode 104: Smelling Salts for Morality: Our Top 3" becomes
    "Very Bad Wizards - Episode 104 - Smelling Salts for Morality Our Top 3.mp3".
    """
    name = title.strip().replace(": ", " - ", 1).replace(": ", " ")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(".")
    return f"{FILENAME_PREFIX}{name}.mp3"


def title_key(title: str) -> str:
    """Reduce a feed title or file name to lowercase letters and digits, without prefixes or numbers.

    "Very Bad Wizards - Episode 99 - It's a Celebration" and "Episode 100: It's a Celebration"
    both become "itsacelebration".
    """
    key = title.lower().removesuffix(".mp3")
    key = key.removeprefix(FILENAME_PREFIX.lower())
    key = re.sub(r"^(\[bonus\]\s*)?(bonus )?(episode\s*\d*)?\s*[:\-]?\s*", "", key)
    key = re.sub(r"^bonus episode\s*", "", key)
    return re.sub(r"[^a-z0-9]+", "", key)


def is_same_episode(key_a: str, key_b: str) -> bool:
    """Match title keys exactly, by containment, or by similarity, but never if their numbers differ.

    Requiring the same numbers keeps "Notes From Underground (Pt. 1)" and "(Pt. 2)" apart.
    """
    if re.findall(r"\d+", key_a) != re.findall(r"\d+", key_b):
        return False
    if key_a == key_b:
        return True
    shorter, longer = sorted((key_a, key_b), key=len)
    if len(shorter) >= MIN_CONTAINED_KEY_LENGTH and shorter in longer:
        return True
    return difflib.SequenceMatcher(None, key_a, key_b).ratio() >= FUZZY_MATCH_CUTOFF


def parse_feed(xml_bytes: bytes) -> list[Episode]:
    """Read every item with an audio enclosure from an RSS feed, oldest first."""
    root = ET.fromstring(xml_bytes)
    episodes = []
    for item in root.iter("item"):
        enclosure = item.find("enclosure")
        title = item.findtext("title")
        if enclosure is None or not title or not enclosure.get("url"):
            continue
        published = item.findtext("pubDate")
        date = parsedate_to_datetime(published).date().isoformat() if published else ""
        episodes.append(Episode(title=clean_title(title.strip()), audio_url=str(enclosure.get("url")), published=date))
    return sorted(episodes, key=lambda episode: episode.published)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        return response.read()


def missing_episodes(episodes: list[Episode], output_dir: Path) -> list[Episode]:
    """Return the episodes that have no matching audio file in output_dir."""
    existing_keys = [title_key(path.name) for path in output_dir.glob("*.mp3")]
    return [
        episode
        for episode in episodes
        if not any(is_same_episode(title_key(episode.title), key) for key in existing_keys)
    ]


def planned_renames(episodes: list[Episode], output_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each existing file whose name differs from its feed episode's name with the feed name.

    Files that match no feed episode, or more than one, are left alone. Raises ValueError if two
    files match the same episode, since renaming both would make one overwrite the other.
    """
    renames = []
    claimed: dict[str, Path] = {}
    for path in sorted(output_dir.glob("*.mp3")):
        key = title_key(path.name)
        matches = [episode for episode in episodes if is_same_episode(title_key(episode.title), key)]
        if len(matches) != 1:
            continue
        target = output_dir / matches[0].filename
        if target.name in claimed:
            raise ValueError(f"{path.name} and {claimed[target.name].name} both match {target.name}.")
        claimed[target.name] = path
        if path.name != target.name:
            renames.append((path, target))
    return renames


def apply_renames(renames: list[tuple[Path, Path]], log_path: Path) -> None:
    """Rename files in two steps, so swapping names like Episode 99 and 100 never overwrites anything.

    Every rename is appended to log_path as an old,new CSV row so it can be undone.
    """
    temporary = []
    for index, (source, target) in enumerate(renames):
        holding = source.with_name(f".renaming-{index}.tmp")
        source.rename(holding)
        temporary.append((holding, source, target))
    new_log = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_log:
            writer.writerow(["old_name", "new_name"])
        for holding, source, target in temporary:
            if target.exists():
                holding.rename(source)
                raise FileExistsError(f"{target.name} already exists, so {source.name} was not renamed.")
            holding.rename(target)
            writer.writerow([source.name, target.name])


def download(url: str, destination: Path) -> None:
    """Download to a .part file first, so an interrupted download never looks finished."""
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as file:
        while chunk := response.read(CHUNK_BYTES):
            file.write(chunk)
    partial.replace(destination)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Folder of episode MP3s")
    parser.add_argument("--feed", default=FEED_URL, help="Podcast RSS feed URL")
    parser.add_argument("--dry-run", action="store_true", help="List missing episodes without downloading")
    parser.add_argument("--limit", type=int, help="Download at most this many episodes")
    parser.add_argument(
        "--rename-existing",
        action="store_true",
        help="Rename existing files to their feed titles and numbers instead of downloading",
    )
    args = parser.parse_args(argv)
    if isinstance(sys.stdout, io.TextIOWrapper):
        # Episode titles can contain characters the Windows console cannot print.
        sys.stdout.reconfigure(errors="replace")

    args.output.mkdir(parents=True, exist_ok=True)
    episodes = parse_feed(fetch(args.feed))

    if args.rename_existing:
        renames = planned_renames(episodes, args.output)
        for source, target in renames:
            print(f"  {source.name}\n    -> {target.name}")
        if not args.dry_run and renames:
            apply_renames(renames, args.output / RENAME_LOG)
        print(f"{len(renames)} files {'would be' if args.dry_run else 'were'} renamed.")
        return

    todo = missing_episodes(episodes, args.output)[: args.limit]
    print(f"{len(episodes)} episodes in the feed, {len(todo)} to download.")

    for episode in tqdm(todo, desc="Downloading", unit="episode", disable=args.dry_run):
        if args.dry_run:
            print(f"  {episode.published}  {episode.filename}")
        else:
            download(episode.audio_url, args.output / episode.filename)


if __name__ == "__main__":
    main()
