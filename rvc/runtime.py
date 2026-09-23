"""Settings and paths for the RVC engine, replacing Applio's app config and fixed folder layout.

Applio expected to run from its own repository root, with a settings file at assets/config.json
and every model's training folder under logs/. Here the package finds its own configs, weights,
and data files relative to itself, and everything that varies per deployment comes from
environment variables, so the same code runs from this repo, in a container, or on a server.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PACKAGE_DIR / "configs"
MODELS_DIR = PACKAGE_DIR / "models"  # Downloaded pretrained weights, not tracked by git
MUTE_DIR = PACKAGE_DIR / "assets" / "mute"  # Silent examples mixed into every training set

# Each trained voice gets a folder here, named after the model.
LOGS_DIR = Path(os.environ.get("RVC_LOGS_DIR", "logs")).resolve()

# Training precision: "fp16", "bf16", or "fp32". fp16 suits GPUs without bf16 support, such as
# the RTX 20 series. Unsupported choices fall back to fp32.
PRECISION = os.environ.get("RVC_PRECISION", "fp16")

# Written into exported model files. Optional.
MODEL_AUTHOR = os.environ.get("RVC_MODEL_AUTHOR")
