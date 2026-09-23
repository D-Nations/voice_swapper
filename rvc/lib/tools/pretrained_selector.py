from rvc.runtime import MODELS_DIR
import os


def pretrained_selector(vocoder, sample_rate):
    base_path = str(MODELS_DIR / "pretraineds" / vocoder.lower())

    path_g = os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth")

    if os.path.exists(path_g) and os.path.exists(path_d):
        return path_g, path_d
    else:
        return "", ""
