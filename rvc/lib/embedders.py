"""ContentVec, which turns 16 kHz speech into speaker-neutral content features, 50 frames per second."""

from pathlib import Path

import torch
from transformers import HubertConfig, HubertModel
from transformers.utils import logging as transformers_logging

from rvc.runtime import MODELS_DIR

CONTENTVEC_DIR = MODELS_DIR / "embedders" / "contentvec"


class HubertModelWithFinalProj(HubertModel):
    """HuBERT with the final projection layer that ContentVec's checkpoint includes."""

    def __init__(self, config: HubertConfig) -> None:
        super().__init__(config)
        self.final_proj = torch.nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_contentvec(model_dir: Path = CONTENTVEC_DIR) -> HubertModelWithFinalProj:
    if not (model_dir / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"ContentVec is missing from {model_dir}. Download it with: python -m rvc.weights")
    transformers_logging.set_verbosity_error()
    return HubertModelWithFinalProj.from_pretrained(model_dir)
