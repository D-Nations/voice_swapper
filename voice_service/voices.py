"""The voices the service offers: which trained model each uses, and what it's called.

Epochs were chosen by speaker similarity across every saved epoch (python -m voice_service.evaluate):
130 for pizarro and 220 for sommers. Blind A/B tests on held-out sentences found every epoch from 130
to 400 close to indistinguishable, with a slight lean toward these over 300, and 300 over 400 (400
continued training on a 60-minute clip set). Word error rates didn't differ between any of them.

Index rates were chosen by sweeping 0 to 1 on the chosen epochs (--chosen --index-rates). Speaker
similarity barely changed across the range, and 0.75 and above garbled slightly more words, so both
voices use 0.4.
"""

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path("data/rvc/models")


@dataclass(frozen=True)
class VoiceChoice:
    key: str  # The training folder under MODELS_DIR, and the model's file name prefix.
    label: str  # Shown to people.
    epoch: int
    index_rate: float = 0.4  # How far to pull the input's features toward the voice's training audio.

    def model_path(self, models_dir: Path = MODELS_DIR) -> Path:
        """The exported model for the chosen epoch, like pizarro_130e_12350s.pth."""
        matches = sorted((models_dir / self.key).glob(f"{self.key}_{self.epoch}e_*s.pth"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {self.key} model for epoch {self.epoch} in {models_dir / self.key}, found {len(matches)}."
            )
        return matches[0]

    def index_path(self, models_dir: Path = MODELS_DIR) -> Path | None:
        """The voice's retrieval index, or None if it wasn't built."""
        path = models_dir / self.key / f"{self.key}.index"
        return path if path.exists() else None


VOICES = {
    voice.key: voice
    for voice in (
        VoiceChoice("pizarro", "DaveBot", epoch=130),
        VoiceChoice("sommers", "TamBot", epoch=220),
    )
}
