import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    n_mels: int
    min_db: float
    max_db: float


@dataclass(frozen=True)
class DataConfig:
    segment_frames: int


@dataclass(frozen=True)
class Config:
    audio: AudioConfig
    data: DataConfig


def load_config(path: str | Path = CONFIG_PATH) -> Config:
    """Read the shared settings file.

    Raises TypeError if a section has a missing or unknown key, and KeyError if a
    whole section is missing.
    """
    with open(path, "rb") as file:
        raw = tomllib.load(file)
    return Config(audio=AudioConfig(**raw["audio"]), data=DataConfig(**raw["data"]))


CONFIG = load_config()
