from pathlib import Path

import pytest

from cycle_gan.config import CONFIG, load_config


def test_repository_config_loads() -> None:
    assert CONFIG.audio.sample_rate > 0
    assert CONFIG.audio.min_db < CONFIG.audio.max_db
    assert CONFIG.data.segment_frames > 0


def test_config_rejects_a_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[audio]\nsample_rate = 22050\n\n[data]\nsegment_frames = 128\n")

    with pytest.raises(TypeError):
        load_config(path)


def test_config_rejects_an_unknown_key(tmp_path: Path) -> None:
    text = Path("config.toml").read_text().replace("[data]\n", "[data]\ntypo_setting = 1\n")
    path = tmp_path / "config.toml"
    path.write_text(text)

    with pytest.raises(TypeError):
        load_config(path)
