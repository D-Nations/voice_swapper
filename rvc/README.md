# rvc

The RVC training engine, adapted from [Applio](https://github.com/IAHispano/Applio) v3.6.5
(commit `55fe0b97`) under its MIT license, which is in `LICENSE`.

Only what this project uses was kept: RVC v2 models with pitch guidance, the HiFi-GAN NSF
vocoder, RMVPE pitch tracking, and ContentVec features, trained on one GPU. The code is typed and
checked by ruff and ty like the rest of the repo.

## Layout

- `configs/`: model and training settings per sample rate (`<rate>.json`), read through the
  dataclasses in `config.py`.
- `lib/algorithm/`: the voice model (`synthesizers.py`) and the discriminator.
- `lib/predictors/rmvpe.py`, `lib/embedders.py`, `lib/audio.py`: pitch, content features, and audio loading.
- `train/`: the four training stages, each runnable as a module:
  1. `python -m rvc.train.preprocess.preprocess`: slice clips at silences into 3-second pieces.
  2. `python -m rvc.train.extract.extract`: pitch and content features, then the file list.
  3. `python -m rvc.train.train`: fine-tune the pretrained base models.
  4. `python -m rvc.train.process.extract_index`: the retrieval index.

  `voice_service/train.py` runs all four for one voice.
- `runtime.py`: paths and environment settings. `weights.py`: downloads the pretrained weights
  (`python -m rvc.weights`).

## Changes from upstream

- **Same numbers.** Model code keeps Applio's parameter names and the order of its random draws,
  so pretrained weights, checkpoints, and exported voice models are interchangeable with Applio's,
  and outputs on fixed inputs match exactly.
- **Removed:** the RefineGAN and MRF HiFi-GAN vocoders, models without pitch guidance, the v1 and
  v3 discriminators, the CREPE and FCPE pitch trackers, other embedders, multi-GPU training, noise
  reduction and other preprocessing options, AMD/ZLUDA workarounds, the bf16 AdamW optimizer, and
  Applio's app settings and process tracking.
- **No `torch.jit`**, which Python 3.14 doesn't support.
- **Command-line options** instead of positional arguments, and no `sys.path` changes.
- **Training checkpoints** are `G_latest.pth` and `D_latest.pth`. Resuming picks up the newest
  `G_*.pth` and `D_*.pth`, so runs started with Applio's naming resume too.
- **Weight norm removal** for inference works with torch's parametrized weight norm.
- **Fixed:** a residual block mask check that would fail on a tensor, a crash in feature
  extraction's progress message, and model export failing without `model_info.json`.
- **Absolute paths in training file lists**, so training works from any directory and across
  Windows drives.
- **Only the ContentVec mute set is bundled,** in `assets/mute/`.
