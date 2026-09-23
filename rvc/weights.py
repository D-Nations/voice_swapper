"""Download the pretrained weights that training and conversion need, and nothing else.

Replaces Applio's prerequisites downloader, which fetched weights for every sample rate, vocoder,
and pitch tracker its app offers. Files come from Applio's Hugging Face repository.

Run with: python -m rvc.weights [--sample-rate 40000]
"""

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

from rvc.runtime import MODELS_DIR

BASE_URL = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"
SAMPLE_RATES = (32000, 40000, 48000)  # Rates with HiFi-GAN base models
DEFAULT_SAMPLE_RATE = 40000
CHUNK_BYTES = 1 << 20

# Needed at every sample rate: the pitch tracker and the speaker-neutral content encoder.
SHARED_FILES = {
    "predictors/rmvpe.pt": "predictors/rmvpe.pt",
    "embedders/contentvec/pytorch_model.bin": "embedders/contentvec/pytorch_model.bin",
    "embedders/contentvec/config.json": "embedders/contentvec/config.json",
}


def base_model_files(sample_rate: int) -> dict[str, str]:
    """The HiFi-GAN generator and discriminator that training fine-tunes, keyed by local path."""
    if sample_rate not in SAMPLE_RATES:
        raise ValueError(f"No base models for {sample_rate} Hz. Choose one of {SAMPLE_RATES}.")
    khz = str(sample_rate)[:2]
    return {
        f"pretraineds/hifi-gan/f0G{khz}k.pth": f"pretrained_v2/f0G{khz}k.pth",
        f"pretraineds/hifi-gan/f0D{khz}k.pth": f"pretrained_v2/f0D{khz}k.pth",
    }


def required_files(sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict[str, str]:
    """Every file needed to train and convert at one sample rate, as {local path: remote path}."""
    return SHARED_FILES | base_model_files(sample_rate)


def missing_files(sample_rate: int = DEFAULT_SAMPLE_RATE, models_dir: Path = MODELS_DIR) -> list[str]:
    return [local for local in required_files(sample_rate) if not (models_dir / local).is_file()]


def download(url: str, destination: Path) -> None:
    """Download to a .part file first, so an interrupted download never looks finished."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(partial, "wb") as file, tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as bar:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                file.write(chunk)
                bar.update(len(chunk))
    partial.replace(destination)


def ensure_weights(sample_rate: int = DEFAULT_SAMPLE_RATE, models_dir: Path = MODELS_DIR) -> None:
    """Download whichever required files are missing."""
    files = required_files(sample_rate)
    for local in missing_files(sample_rate, models_dir):
        download(f"{BASE_URL}/{files[local]}", models_dir / local)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, choices=SAMPLE_RATES)
    args = parser.parse_args(argv)
    ensure_weights(args.sample_rate)
    print(f"All weights for {args.sample_rate} Hz are in {MODELS_DIR}")


if __name__ == "__main__":
    main()
