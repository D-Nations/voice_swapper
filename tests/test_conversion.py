import dataclasses

import numpy as np
import pytest

from rvc.configs.config import RVCConfig
from rvc.infer.pipeline import (
    CENTER_SECONDS,
    MAX_SECONDS,
    QUERY_SECONDS,
    SAMPLE_RATE,
    ConversionSettings,
    Pipeline,
    median_pitch,
    semitones_between,
)
from rvc.train.process.extract_model import read_saved_config, saved_config


def test_saved_config_reads_back_as_the_same_model() -> None:
    config = RVCConfig.for_sample_rate(40000)

    saved = read_saved_config(saved_config(config), speakers=1)

    assert saved.model == dataclasses.replace(config.model, spk_embed_dim=1)
    assert saved.spec_channels == config.data.spec_channels
    assert saved.sample_rate == 40000


def test_read_saved_config_rejects_a_malformed_list() -> None:
    saved = saved_config(RVCConfig.for_sample_rate(40000))
    saved[2] = "192"

    with pytest.raises(TypeError, match="int"):
        read_saved_config(saved, speakers=1)


def test_median_pitch_ignores_unvoiced_frames() -> None:
    assert median_pitch(np.array([0.0, 100.0, 0.0, 120.0, 140.0])) == 120.0
    assert median_pitch(np.array([0.0, 0.0, 130.0])) is None


def test_semitones_between_octaves_is_twelve() -> None:
    assert semitones_between(110.0, 220.0) == pytest.approx(12.0)
    assert semitones_between(220.0, 110.0) == pytest.approx(-12.0)


def test_target_pitch_overrides_the_fixed_shift() -> None:
    f0 = np.array([0.0, 100.0, 100.0, 100.0])

    assert Pipeline._shift(f0, ConversionSettings(pitch_shift=3.0)) == 3.0
    assert Pipeline._shift(f0, ConversionSettings(pitch_shift=3.0, target_pitch_hz=200.0)) == pytest.approx(12.0)


def test_target_pitch_leaves_unvoiced_audio_unshifted() -> None:
    assert Pipeline._shift(np.zeros(10), ConversionSettings(target_pitch_hz=200.0)) == 0.0


def test_short_audio_is_converted_in_one_piece() -> None:
    assert Pipeline._cut_points(np.ones(SAMPLE_RATE * (MAX_SECONDS - 1))) == []


def test_long_audio_is_cut_at_the_quietest_point_near_each_center() -> None:
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.5, 0.5, SAMPLE_RATE * 60)
    quiet = SAMPLE_RATE * (CENTER_SECONDS + 2)
    audio[quiet - 800 : quiet + 800] = 0.0

    cuts = Pipeline._cut_points(audio)

    assert len(cuts) == 1
    assert abs(cuts[0] - quiet) < 800
    assert abs(cuts[0] - SAMPLE_RATE * CENTER_SECONDS) <= SAMPLE_RATE * QUERY_SECONDS
