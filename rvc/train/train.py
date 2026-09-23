"""Stage 3: fine-tune the pretrained generator and discriminator on one voice.

Reads <experiment dir>/config.json and filelist.txt from the extract stage. Every
--save-every epochs it overwrites G_latest.pth and D_latest.pth, which let an interrupted run
resume, and exports a small voice model, <name>_<epoch>e_<step>s.pth, for conversion.
TensorBoard logs go to <experiment dir>/eval.

Run with: python -m rvc.train.train --experiment-dir DIR --pretrained-g G.pth --pretrained-d D.pth
"""

import argparse
import dataclasses
import datetime
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rvc.configs.config import RVCConfig
from rvc.lib.algorithm import commons
from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.runtime import PRECISION
from rvc.train.data_utils import BucketBatchSampler, TextAudioLoaderMultiNSFsid, TrainingBatch, collate
from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from rvc.train.model_info import read_model_info
from rvc.train.process.extract_model import export_voice_model
from rvc.train.utils import (
    latest_checkpoint_path,
    load_checkpoint,
    load_wav_to_torch,
    plot_spectrogram_to_numpy,
    save_checkpoint,
    summarize,
)

type Precision = Literal["fp32", "fp16", "bf16"]

BUCKET_BOUNDARIES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
LOADER_WORKERS = 4
LOADER_PREFETCH = 8
LOG_EVERY_STEPS = 50
MIN_BATCHES = 3


@dataclass(frozen=True)
class TrainingOptions:
    experiment_dir: Path
    pretrained_g: Path | None
    pretrained_d: Path | None
    total_epochs: int = 300
    save_every_epochs: int = 10
    batch_size: int = 8
    device: str = "cuda:0"
    precision: Precision = "fp16"
    export_every_save: bool = True  # Export a voice model at every save, not only the last.
    checkpointing: bool = False  # Trade speed for memory.

    @property
    def name(self) -> str:
        return self.experiment_dir.name


def empty_rolling_averages() -> dict[str, deque[float | torch.Tensor]]:
    """The last LOG_EVERY_STEPS values of each logged quantity."""
    names = ("grad_d", "grad_g", "disc_loss", "adv_loss", "fm_loss", "kl_loss", "mel_loss", "gen_loss")
    return {name: deque(maxlen=LOG_EVERY_STEPS) for name in names}


@dataclass
class TrainingState:
    """What carries over between epochs."""

    global_step: int = 0
    lowest_loss: float = float("inf")
    lowest_loss_epoch: int = 0
    lowest_loss_step: int = 0
    rolling: dict[str, deque[float | torch.Tensor]] = field(default_factory=empty_rolling_averages)


class StepResult(NamedTuple):
    """A training step's batch, output, and losses, kept for the end-of-epoch logs."""

    batch: TrainingBatch
    ids_slice: torch.Tensor
    y_hat: torch.Tensor
    losses: dict[str, float | torch.Tensor]


@dataclass
class Trainer:
    options: TrainingOptions
    config: RVCConfig
    device: torch.device
    train_dtype: torch.dtype
    net_g: Synthesizer
    net_d: MultiPeriodDiscriminator
    optim_g: torch.optim.Optimizer
    optim_d: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    loader: DataLoader
    sampler: BucketBatchSampler
    writer: SummaryWriter
    reference: TrainingBatch
    state: TrainingState = field(default_factory=TrainingState)

    @property
    def use_amp(self) -> bool:
        return self.device.type == "cuda" and self.train_dtype in (torch.float16, torch.bfloat16)

    def autocast(self) -> torch.autocast:
        return torch.autocast(device_type="cuda", enabled=self.use_amp, dtype=self.train_dtype)

    def mel(self, audio: torch.Tensor) -> torch.Tensor:
        data = self.config.data
        return mel_spectrogram_torch(
            audio.float().squeeze(1),
            data.filter_length,
            data.n_mel_channels,
            data.sample_rate,
            data.hop_length,
            data.win_length,
            data.mel_fmin,
            data.mel_fmax,
        )

    def backward_and_step(
        self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, model: torch.nn.Module, update_scaler: bool
    ) -> float:
        """Backpropagate, step, and return the gradient norm. fp16 losses are scaled to avoid underflow."""
        optimizer.zero_grad()
        if self.train_dtype == torch.float16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            norm = commons.grad_norm(model.parameters())
            self.scaler.step(optimizer)
            if update_scaler:
                self.scaler.update()
        else:
            loss.backward()
            norm = commons.grad_norm(model.parameters())
            optimizer.step()
        return norm

    def train_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)
        self.net_g.train()
        self.net_d.train()
        started = time.time()

        last: StepResult | None = None
        for batch in tqdm(self.loader, leave=False):
            last = self.train_step(batch.to(self.device, non_blocking=True), epoch)
        if last is None:
            raise ValueError("The training data loader produced no batches.")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        losses = last.losses | {"learning_rate": self.optim_g.param_groups[0]["lr"]}
        self.log_epoch(epoch, last.batch, last.ids_slice, last.y_hat, losses)
        self.print_progress(epoch, time.time() - started)
        if epoch % self.options.save_every_epochs == 0 or epoch >= self.options.total_epochs:
            self.save(epoch)

    def train_step(self, batch: TrainingBatch, epoch: int) -> StepResult:
        """One discriminator update and one generator update on a batch."""
        config, state = self.config, self.state
        with self.autocast():
            y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = self.net_g(
                batch.phone,
                batch.phone_lengths,
                batch.pitch,
                batch.pitchf,
                batch.spec,
                batch.spec_lengths,
                batch.sid,
            )
            # The matching slice of the real audio.
            wave = commons.slice_segments(
                batch.wave, ids_slice * config.data.hop_length, config.train.segment_size, dim=3
            )

        with self.autocast():
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(wave, y_hat.detach())
        loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
        grad_norm_d = self.backward_and_step(loss_disc, self.optim_d, self.net_d, update_scaler=False)

        self.net_d.requires_grad_(False)
        with self.autocast():
            _, y_d_hat_g, fmap_r, fmap_g = self.net_d(wave, y_hat)
        loss_mel = torch.nn.functional.l1_loss(self.mel(wave), self.mel(y_hat)) * config.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        if loss_gen_all.item() < state.lowest_loss:
            state.lowest_loss = loss_gen_all.item()
            state.lowest_loss_epoch = epoch
            state.lowest_loss_step = state.global_step
        grad_norm_g = self.backward_and_step(loss_gen_all, self.optim_g, self.net_g, update_scaler=True)
        self.net_d.requires_grad_(True)
        state.global_step += 1

        for name, value in (
            ("grad_d", grad_norm_d),
            ("grad_g", grad_norm_g),
            ("disc_loss", loss_disc.detach()),
            ("adv_loss", loss_gen.detach()),
            ("fm_loss", loss_fm.detach()),
            ("kl_loss", loss_kl.detach()),
            ("mel_loss", loss_mel.detach()),
            ("gen_loss", loss_gen_all.detach()),
        ):
            state.rolling[name].append(value)
        if state.global_step % LOG_EVERY_STEPS == 0:
            self.log_rolling_averages()

        losses: dict[str, float | torch.Tensor] = {
            "loss/g/total": loss_gen_all,
            "loss/d/adv": loss_disc,
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "loss/g/adv": loss_gen,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/kl": loss_kl,
        }
        return StepResult(batch, ids_slice, y_hat, losses)

    def log_rolling_averages(self) -> None:
        rolling = self.state.rolling
        scalars: dict[str, float | torch.Tensor] = {
            f"grad_avg_50/norm_{side}": sum(rolling[f"grad_{side}"]) / len(rolling[f"grad_{side}"]) for side in "dg"
        }
        for tag, name in (
            ("d/adv", "disc_loss"),
            ("g/adv", "adv_loss"),
            ("g/fm", "fm_loss"),
            ("g/kl", "kl_loss"),
            ("g/mel", "mel_loss"),
            ("g/total", "gen_loss"),
        ):
            scalars[f"loss_avg_50/{tag}"] = torch.mean(torch.stack([torch.as_tensor(v) for v in rolling[name]]))
        summarize(self.writer, self.state.global_step, scalars=scalars)

    def log_epoch(
        self,
        epoch: int,
        batch: TrainingBatch,
        ids_slice: torch.Tensor,
        y_hat: torch.Tensor,
        losses: dict[str, float | torch.Tensor],
    ) -> None:
        """Log the last batch's losses and spectrograms, and at save epochs a conversion of the reference batch."""
        data = self.config.data
        mel = spec_to_mel_torch(
            batch.spec, data.filter_length, data.n_mel_channels, data.sample_rate, data.mel_fmin, data.mel_fmax
        )
        y_mel = commons.slice_segments(mel, ids_slice, self.config.segment_frames, dim=3)
        images = {
            "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].detach().cpu().numpy()),
            "slice/mel_gen": plot_spectrogram_to_numpy(self.mel(y_hat)[0].detach().cpu().numpy()),
            "all/mel": plot_spectrogram_to_numpy(mel[0].detach().cpu().numpy()),
        }
        audios = {}
        if epoch % self.options.save_every_epochs == 0:
            reference = self.reference
            with self.autocast(), torch.no_grad():
                audio, _, _ = self.net_g.infer(
                    reference.phone, reference.phone_lengths, reference.pitch, reference.pitchf, reference.sid
                )
            audios[f"gen/audio_{self.state.global_step:07d}"] = audio[0, :, :]
        summarize(
            self.writer,
            self.state.global_step,
            scalars=losses,
            images=images,
            audios=audios,
            audio_sample_rate=data.sample_rate,
        )

    def print_progress(self, epoch: int, seconds: float) -> None:
        state = self.state
        record = (
            f"{self.options.name} | epoch={epoch} | step={state.global_step} | "
            f"time={datetime.datetime.now().astimezone():%H:%M:%S} | training_speed={datetime.timedelta(seconds=int(seconds))}"
        )
        if epoch > 1:
            record += (
                f" | lowest_value={state.lowest_loss:.3f} "
                f"(epoch {state.lowest_loss_epoch} and step {state.lowest_loss_step})"
            )
        print(record, flush=True)

    def save(self, epoch: int) -> None:
        directory = self.options.experiment_dir
        learning_rate = self.config.train.learning_rate
        save_checkpoint(self.net_g, self.optim_g, learning_rate, epoch, directory / "G_latest.pth", self.scaler)
        save_checkpoint(self.net_d, self.optim_d, learning_rate, epoch, directory / "D_latest.pth", self.scaler)
        final = epoch >= self.options.total_epochs
        if self.options.export_every_save or final:
            model_path = directory / f"{self.options.name}_{epoch}e_{self.state.global_step}s.pth"
            if not model_path.exists():
                export_voice_model(
                    self.net_g.state_dict(), self.config, self.options.name, model_path, epoch, self.state.global_step
                )


def choose_dtype(precision: Precision, device: torch.device) -> torch.dtype:
    if precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision == "fp16" and device.type == "cuda":
        return torch.float16
    return torch.float32


def check_sample_rate(experiment_dir: Path, config: RVCConfig) -> None:
    first_wav = next((experiment_dir / "sliced_audios").glob("*.wav"), None)
    if first_wav is None:
        raise FileNotFoundError(f"No slices in {experiment_dir / 'sliced_audios'}. Run the earlier stages first.")
    _, sample_rate = load_wav_to_torch(first_wav)
    if sample_rate != config.data.sample_rate:
        raise ValueError(
            f"The model config is for {config.data.sample_rate} Hz, but the slices are at {sample_rate} Hz."
        )


def speaker_count(experiment_dir: Path, config: RVCConfig) -> int:
    """Speakers in the dataset, or in the checkpoint being resumed, which wins."""
    count = read_model_info(experiment_dir).get("speakers_id", config.model.spk_embed_dim)
    checkpoint_path = latest_checkpoint_path(experiment_dir, "G_*.pth")
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        count = checkpoint["model"]["emb_g.weight"].shape[0]
    return count


def load_pretrained(net_g: Synthesizer, net_d: MultiPeriodDiscriminator, options: TrainingOptions) -> None:
    """Start from the pretrained base models, keeping the new speaker embedding, which differs in size."""
    if options.pretrained_g is not None:
        state = torch.load(options.pretrained_g, map_location="cpu", weights_only=True)["model"]
        state["emb_g.weight"] = net_g.emb_g.weight.detach().clone()
        net_g.load_state_dict(state)
        print(f"Loaded pretrained generator '{options.pretrained_g}'")
    if options.pretrained_d is not None:
        net_d.load_state_dict(torch.load(options.pretrained_d, map_location="cpu", weights_only=True)["model"])
        print(f"Loaded pretrained discriminator '{options.pretrained_d}'")


def train(options: TrainingOptions) -> None:
    experiment_dir = options.experiment_dir
    config_path = experiment_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found. Run the extract stage first.")
    config = RVCConfig.load(config_path)
    check_sample_rate(experiment_dir, config)

    device = torch.device(options.device)
    train_dtype = choose_dtype(options.precision, device)
    if os.name == "nt":
        torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        print("Training on the CPU, which is very slow.")
    torch.manual_seed(config.train.seed)

    dataset = TextAudioLoaderMultiNSFsid(experiment_dir / "filelist.txt", config.data)
    sampler = BucketBatchSampler(dataset.lengths, options.batch_size, BUCKET_BOUNDARIES)
    loader = DataLoader(
        dataset,
        num_workers=LOADER_WORKERS,
        pin_memory=True,
        collate_fn=collate,
        batch_sampler=sampler,
        persistent_workers=True,
        prefetch_factor=LOADER_PREFETCH,
    )
    if len(loader) < MIN_BATCHES:
        raise ValueError(f"Only {len(loader)} batches of training data. Add more clips or lower the batch size.")

    spk_dim = speaker_count(experiment_dir, config)
    print(f"Initializing the generator with {spk_dim} speakers.")
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, spk_embed_dim=spk_dim))
    net_g = Synthesizer.from_config(
        config.model, config.data.spec_channels, config.segment_frames, config.data.sample_rate, options.checkpointing
    ).to(device)
    net_d = MultiPeriodDiscriminator(config.model.use_spectral_norm, checkpointing=options.checkpointing).to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(), config.train.learning_rate, betas=config.train.betas, eps=config.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), config.train.learning_rate, betas=config.train.betas, eps=config.train.eps
    )
    print(f"Training in {train_dtype}.")

    state = TrainingState()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and train_dtype == torch.float16)
    checkpoint_d = latest_checkpoint_path(experiment_dir, "D_*.pth")
    checkpoint_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
    if checkpoint_d is not None and checkpoint_g is not None:
        _, scaler_state = load_checkpoint(checkpoint_d, net_d, optim_d)
        last_epoch, _ = load_checkpoint(checkpoint_g, net_g, optim_g)
        first_epoch = last_epoch + 1
        state.global_step = last_epoch * len(loader)
        if scaler_state:
            scaler.load_state_dict(scaler_state)
    else:
        first_epoch = 1
        load_pretrained(net_g, net_d, options)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=first_epoch - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=first_epoch - 2
    )

    # A fixed batch to convert at each save, so the TensorBoard audio shows progress on the same input.
    reference = next(iter(loader)).to(device)
    trainer = Trainer(
        options=options,
        config=config,
        device=device,
        train_dtype=train_dtype,
        net_g=net_g,
        net_d=net_d,
        optim_g=optim_g,
        optim_d=optim_d,
        scaler=scaler,
        loader=loader,
        sampler=sampler,
        writer=SummaryWriter(log_dir=str(experiment_dir / "eval")),
        reference=reference,
        state=state,
    )
    print("Starting training...")
    for epoch in range(first_epoch, options.total_epochs + 1):
        trainer.train_epoch(epoch)
        scheduler_g.step()
        scheduler_d.step()

    trainer.writer.close()
    print(
        f"Training finished at epoch {options.total_epochs}, step {state.global_step}. Lowest generator loss "
        f"{state.lowest_loss:.3f} at epoch {state.lowest_loss_epoch}, step {state.lowest_loss_step}."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--pretrained-g", type=Path, help="Pretrained generator to start from")
    parser.add_argument("--pretrained-d", type=Path, help="Pretrained discriminator to start from")
    parser.add_argument("--epochs", type=int, default=TrainingOptions.total_epochs)
    parser.add_argument("--save-every", type=int, default=TrainingOptions.save_every_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingOptions.batch_size)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=PRECISION)
    parser.add_argument("--export-final-only", action="store_true", help="Export a voice model only at the end")
    parser.add_argument("--checkpointing", action="store_true", help="Use less GPU memory, more slowly")
    args = parser.parse_args(argv)
    options = TrainingOptions(
        experiment_dir=args.experiment_dir.resolve(),
        pretrained_g=args.pretrained_g,
        pretrained_d=args.pretrained_d,
        total_epochs=args.epochs,
        save_every_epochs=args.save_every,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
        export_every_save=not args.export_final_only,
        checkpointing=args.checkpointing,
    )
    try:
        train(options)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Training failed: {error}")


if __name__ == "__main__":
    main()
