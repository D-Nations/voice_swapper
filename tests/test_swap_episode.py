import numpy as np
import pytest

from voice_service.swap_episode import crossfade_into, parse_voices


def test_parse_voices_keeps_the_default_label_and_index_rate() -> None:
    voices = parse_voices(["pizarro=pizarro_60min:400"])

    assert list(voices) == ["pizarro"]
    assert (voices["pizarro"].key, voices["pizarro"].epoch, voices["pizarro"].label) == (
        "pizarro_60min",
        400,
        "DaveBot",
    )


@pytest.mark.parametrize("value", ["pizarro", "pizarro=pizarro_60min", "pizarro=:400", "hal=hal:400"])
def test_parse_voices_rejects_malformed_values(value: str) -> None:
    with pytest.raises(ValueError, match="should look like"):
        parse_voices([value])


def test_crossfade_into_blends_the_start_of_a_piece() -> None:
    output = np.zeros(10, dtype=np.float32)

    crossfade_into(output, np.ones(6, dtype=np.float32), start=2, fade=2)

    assert output.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
