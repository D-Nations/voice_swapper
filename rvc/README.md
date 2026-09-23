# rvc

The RVC training engine, copied from [Applio](https://github.com/IAHispano/Applio) v3.6.5
(commit `55fe0b97`) under its MIT license, which is in `LICENSE`. The same commit is checked out in
`third_party/applio` for reference.

Only the files that training needs were copied. The code under `configs/`, `lib/`, and `train/` is
kept as close to upstream as possible, so it can be compared with the reference copy, and is
excluded from this repo's linting and formatting.

## Changes from upstream

- **No dependence on Applio's folder layout.** Configs, weights, and bundled data are found
  relative to this package, through `runtime.py`, instead of the current directory. Training runs
  go under `RVC_LOGS_DIR`, which defaults to `logs/`.
- **No Applio app settings file.** Training precision and the model author come from the
  `RVC_PRECISION` and `RVC_MODEL_AUTHOR` environment variables instead of `assets/config.json`.
  The RMVPE high-register options use their defaults.
- **Absolute paths in training file lists**, so training works from any directory and across
  Windows drives.
- **`wget` replaced with `requests`**, since `wget` has no prebuilt package.
- **Only the ContentVec mute set is bundled,** in `assets/mute/`. Other embedders raise an error.
- **A focused weights downloader,** `weights.py`, replaces Applio's prerequisites downloader.

## New files

- `runtime.py`: paths and environment settings.
- `weights.py`: downloads the pretrained weights for one sample rate. Run `python -m rvc.weights`.
