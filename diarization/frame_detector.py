"""A frame-level detector of who is talking: each host's probability of speaking every 20 ms.

The ECAPA speaker map scores 1.5 second windows, so it only knows a handoff to within 0.75 seconds
and can't tell when both hosts talk at once. This detector runs a frozen WavLM model fine-tuned for
speaker diarization (microsoft/wavlm-base-plus-sd), mixes its layers with learned weights, and
passes them through a small BiLSTM with one output per host. The outputs are independent, so both
hosts can be active at once (overlap) and neither can be (silence, music, a guest).

It's trained on synthetic conversations from diarization.mixtures, built from the hosts' training
clips, and validated on conversations built from clips of held-out episodes.

Train with:
    python -m diarization.frame_detector train [--steps 4000]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from diarization.mixtures import FRAME_SECONDS, SAMPLE_RATE, Clip, load_clips, load_other, make_mixture

BACKBONE = "microsoft/wavlm-base-plus-sd"
SPEAKERS = ["pizarro", "sommers"]  # Output order, matching the speaker maps.
MODEL_PATH = Path("models/detector/frame_detector.pt")

TRAIN_CLIPS = Path("data/rvc/dataset_60min/train")
TEST_CLIPS = Path("data/rvc/test")
OTHER_DIR = Path("data/detector/other")

HIDDEN = 128
BATCH_SIZE = 8
STEPS = 4000
LEARNING_RATE = 1e-3
EVAL_EVERY = 250
VALIDATION_MIXTURES = 200
SEED = 0

# detect() runs the model on CHUNK_SECONDS of audio at a time, with CONTEXT_SECONDS of extra audio
# on each side so the frames near the edges of each chunk still see what's around them.
CHUNK_SECONDS = 20.0
CONTEXT_SECONDS = 2.0

ACTIVE = 0.5  # A host counts as talking in a frame when their probability is at least this.


class FrameDetector(nn.Module):
    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        from transformers import WavLMModel

        self.backbone = WavLMModel.from_pretrained(BACKBONE)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        layers = self.backbone.config.num_hidden_layers + 1
        width = self.backbone.config.hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(layers))
        self.project = nn.Linear(width, 2 * hidden)
        self.lstm = nn.LSTM(2 * hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * hidden, len(SPEAKERS))

    def train(self, mode: bool = True) -> FrameDetector:
        super().train(mode)
        self.backbone.eval()  # The backbone is frozen, so its dropout stays off.
        return self

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """(batch, samples) of 16 kHz audio to (batch, frames, speakers) logits, one frame per 20 ms."""
        with torch.no_grad():
            hidden = self.backbone(audio, output_hidden_states=True).hidden_states
        stacked = torch.stack(hidden, dim=0).float()
        weights = torch.softmax(self.layer_weights, dim=0)
        features = (weights[:, None, None, None] * stacked).sum(dim=0)
        x, _ = self.lstm(torch.relu(self.project(features)))
        return self.output(x)

    def head_state(self) -> dict[str, torch.Tensor]:
        """Everything but the frozen backbone, which is reloaded from BACKBONE."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}


def load_detector(path: Path = MODEL_PATH, device: str | torch.device | None = None) -> FrameDetector:
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = FrameDetector()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    return model.to(device).eval()


@torch.inference_mode()
def detect(model: FrameDetector, audio: np.ndarray) -> np.ndarray:
    """Each host's probability of talking in every 20 ms frame of 16 kHz audio, as (frames, speakers)."""
    device = next(model.output.parameters()).device
    frames = len(audio) // round(FRAME_SECONDS * SAMPLE_RATE)
    probs = np.zeros((frames, len(SPEAKERS)), dtype=np.float32)
    chunk_frames = round(CHUNK_SECONDS / FRAME_SECONDS)
    context_frames = round(CONTEXT_SECONDS / FRAME_SECONDS)
    per_frame = round(FRAME_SECONDS * SAMPLE_RATE)
    for first in range(0, frames, chunk_frames):
        last = min(first + chunk_frames, frames)
        lo = max(0, first - context_frames)
        hi = min(frames, last + context_frames)
        # One extra frame of audio, since WavLM's frames need 25 ms each.
        piece = audio[lo * per_frame : hi * per_frame + per_frame]
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32))[None].to(device))
        out = torch.sigmoid(logits.float())[0].cpu().numpy()
        wanted = out[first - lo : first - lo + last - first]
        probs[first : first + len(wanted)] = wanted
    return probs


def mixture_batch(
    speakers: list[list[Clip]], other: list[np.ndarray], rng: np.random.Generator, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    audio, labels = zip(*(make_mixture(speakers, other, rng)[:2] for _ in range(size)))
    return torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(labels)).float()


def frame_scores(probs: np.ndarray, active: np.ndarray) -> dict[str, float]:
    """Per-frame scores of detector probabilities against true labels, summed so they can be pooled.

    single_*: frames with exactly one host talking, and whether the detector picked that host alone.
    overlap_*: frames with both talking, and whether the detector said so.
    """
    predicted = probs >= ACTIVE
    single = active.sum(axis=1) == 1
    both = active.all(axis=1)
    return {
        "single_frames": float(single.sum()),
        "single_right": float((single & (predicted == active).all(axis=1)).sum()),
        "overlap_frames": float(both.sum()),
        "overlap_found": float((both & predicted.all(axis=1)).sum()),
        "overlap_claimed": float(predicted.all(axis=1).sum()),
        "silent_frames": float((~active.any(axis=1)).sum()),
        "silent_right": float((~active.any(axis=1) & ~predicted.any(axis=1)).sum()),
    }


def summarize(totals: dict[str, float]) -> dict[str, float]:
    return {
        "single_accuracy": totals["single_right"] / max(totals["single_frames"], 1),
        "overlap_recall": totals["overlap_found"] / max(totals["overlap_frames"], 1),
        "overlap_precision": totals["overlap_found"] / max(totals["overlap_claimed"], 1),
        "neither_accuracy": totals["silent_right"] / max(totals["silent_frames"], 1),
    }


@torch.inference_mode()
def validate(
    model: FrameDetector, batches: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    for audio, labels in batches:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(audio.to(device))
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        frames = min(probs.shape[1], labels.shape[1])
        for p, l in zip(probs[:, :frames], labels[:, :frames].numpy().astype(bool)):
            for key, value in frame_scores(p, l).items():
                totals[key] = totals.get(key, 0.0) + value
    model.train()
    return summarize(totals)


def train(steps: int = STEPS, output: Path = MODEL_PATH, device: str | None = None) -> None:
    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print("Loading clips...")
    train_speakers = [load_clips(TRAIN_CLIPS / name) for name in SPEAKERS]
    test_speakers = [load_clips(TEST_CLIPS / name) for name in SPEAKERS]
    train_other, test_other = load_other(OTHER_DIR / "train"), load_other(OTHER_DIR / "test")
    for name, clips in zip(SPEAKERS, train_speakers):
        print(f"  {name}: {len(clips)} training clips")
    print(f"  other: {len(train_other)} training stretches, {len(test_other)} test stretches")

    validation_rng = np.random.default_rng(SEED + 1)
    validation = [
        mixture_batch(test_speakers, test_other, validation_rng, BATCH_SIZE)
        for _ in range(VALIDATION_MIXTURES // BATCH_SIZE)
    ]

    model = FrameDetector().to(torch_device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=torch_device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)

    best = -1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    progress = tqdm(range(1, steps + 1), desc="Training detector", unit="step")
    running = 0.0
    for step in progress:
        audio, labels = mixture_batch(train_speakers, train_other, rng, BATCH_SIZE)
        with torch.autocast(torch_device.type, enabled=torch_device.type == "cuda"):
            logits = model(audio.to(torch_device))
        frames = min(logits.shape[1], labels.shape[1])
        loss = loss_fn(logits[:, :frames].float(), labels[:, :frames].to(torch_device))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
        progress.set_postfix(loss=f"{running:.4f}")

        if step % EVAL_EVERY == 0 or step == steps:
            scores = validate(model, validation, torch_device)
            line = " ".join(f"{k} {v:.3f}" for k, v in scores.items())
            improved = scores["single_accuracy"] > best
            if improved:
                best = scores["single_accuracy"]
                torch.save(model.head_state(), output)
            tqdm.write(f"step {step}: loss {running:.4f} | {line}{' | saved' if improved else ''}")
    print(
        f"Trained in {(time.time() - started) / 60:.0f} minutes. Best single-speaker accuracy {best:.3f}, saved to {output}."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train the detector on synthetic conversations")
    train_parser.add_argument("--steps", type=int, default=STEPS)
    train_parser.add_argument("--output", type=Path, default=MODEL_PATH)
    train_parser.add_argument("--device", help="Such as cuda:0 or cpu. Defaults to the GPU if there is one.")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.steps, args.output, args.device)


if __name__ == "__main__":
    main()
