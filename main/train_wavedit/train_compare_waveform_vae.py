import argparse
import copy
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
except Exception:
    plt = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from data_load import LoadWaveform
from validation.metrics import compute_sample_metrics, summarize_metric_rows
from waveform_scaling import (
    apply_conditioning_config,
    apply_waveform_scale_config,
    denormalize_waveform_array,
    resolve_conditioning_config,
    resolve_waveform_scale_config,
)
from waveform_vae import WaveformKLVAE1D, compute_latent_channel_std, kl_loss


@dataclass(frozen=True)
class VAEExperiment:
    name: str
    latent_channels: int
    strides: tuple[int, ...]
    base_channels: int
    use_adv: bool

    @property
    def downsample_factor(self) -> int:
        factor = 1
        for stride in self.strides:
            factor *= stride
        return factor


SEISMIC_BASE_CONFIGS = (
    ("light_r8_c8", 8, (2, 2, 2), 64),
    ("mid_r12_c16", 16, (2, 2, 3), 64),
    ("heavy_r24_c32", 32, (2, 2, 2, 3), 64),
)


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def setup_distributed(args):
    use_ddp = (
        not args.no_ddp
        and torch.cuda.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
    )
    if not use_ddp:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return False, 0, 1, device

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if args.global_batch_size % world_size != 0:
        cleanup_distributed()
        raise ValueError(
            f"--global-batch-size ({args.global_batch_size}) must be divisible by world_size ({world_size})"
        )
    return True, rank, world_size, device


def make_dataset(args, split: str):
    root = args.wf_dir if split == "train" else args.val_dir
    if root is None:
        return None
    return LoadWaveform(
        root,
        args.sta_lat_min,
        args.sta_lat_max,
        args.sta_lon_min,
        args.sta_lon_max,
        args.src_lat_min,
        args.src_lat_max,
        args.src_lon_min,
        args.src_lon_max,
        args.dep_min,
        args.dep_max,
        args.mag_min,
        args.mag_max,
        args.min_get,
        args.max_get,
        args.length,
        args.waveform_mean_value,
        args.waveform_std_value,
    )


def make_loader(dataset, batch_size, args, sampler=None, shuffle=False, drop_last=False):
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def flatten_batch_channels(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2])


def multi_resolution_stft_loss(x: torch.Tensor, y: torch.Tensor, fft_sizes=(64, 128, 256)) -> torch.Tensor:
    x_flat = flatten_batch_channels(x.float())
    y_flat = flatten_batch_channels(y.float())
    losses = []
    for n_fft in fft_sizes:
        hop = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
        x_mag = torch.stft(x_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True).abs()
        y_mag = torch.stft(y_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True).abs()
        losses.append(F.l1_loss(x_mag, y_mag) + F.l1_loss(torch.log1p(x_mag), torch.log1p(y_mag)))
    return torch.stack(losses).mean()


def log_fas_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_fas = torch.fft.rfft(x.float(), dim=-1).abs()
    y_fas = torch.fft.rfft(y.float(), dim=-1).abs()
    return F.l1_loss(torch.log1p(x_fas), torch.log1p(y_fas))


def envelope_loss(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 65) -> torch.Tensor:
    pad = kernel_size // 2
    x_env = F.avg_pool1d(x.float().abs(), kernel_size=kernel_size, stride=1, padding=pad)[..., : x.shape[-1]]
    y_env = F.avg_pool1d(y.float().abs(), kernel_size=kernel_size, stride=1, padding=pad)[..., : y.shape[-1]]
    return F.l1_loss(x_env, y_env)


class WaveformDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        layers = []
        current = in_channels
        for channel in channels:
            layers.append(nn.utils.weight_norm(nn.Conv1d(current, channel, kernel_size=15, stride=4, padding=7)))
            current = channel
        self.layers = nn.ModuleList(layers)
        self.out = nn.utils.weight_norm(nn.Conv1d(current, 1, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor):
        feats = []
        h = x
        for layer in self.layers:
            h = F.leaky_relu(layer(h), negative_slope=0.2)
            feats.append(h)
        logits = self.out(h)
        feats.append(logits)
        return feats


class MultiScaleWaveformDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                WaveformDiscriminator(in_channels=in_channels, base_channels=24),
                WaveformDiscriminator(in_channels=in_channels, base_channels=24),
                WaveformDiscriminator(in_channels=in_channels, base_channels=24),
            ]
        )

    def forward(self, x: torch.Tensor):
        outs = []
        h = x
        for index, discriminator in enumerate(self.discriminators):
            if index > 0:
                h = F.avg_pool1d(h, kernel_size=4, stride=2, padding=1)
            outs.append(discriminator(h))
        return outs


def set_requires_grad(module: nn.Module, value: bool):
    for param in module.parameters():
        param.requires_grad_(value)


def discriminator_hinge_loss(real_outs, fake_outs):
    loss = 0.0
    for real_feats, fake_feats in zip(real_outs, fake_outs):
        real_logits = real_feats[-1]
        fake_logits = fake_feats[-1]
        loss = loss + F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    return loss / max(1, len(real_outs))


def generator_adv_loss(fake_outs):
    loss = 0.0
    for fake_feats in fake_outs:
        loss = loss - fake_feats[-1].mean()
    return loss / max(1, len(fake_outs))


def feature_matching_loss(real_outs, fake_outs):
    loss = 0.0
    count = 0
    for real_feats, fake_feats in zip(real_outs, fake_outs):
        for real_feat, fake_feat in zip(real_feats[:-1], fake_feats[:-1]):
            loss = loss + F.l1_loss(fake_feat.float(), real_feat.detach().float())
            count += 1
    return loss / max(1, count)


def reconstruction_loss(model, x, args, global_step: int):
    out = model(x, sample=True)
    recon = out["recon"]
    l1 = F.l1_loss(recon.float(), x.float())
    stft = multi_resolution_stft_loss(recon, x)
    fas = log_fas_loss(recon, x)
    env = envelope_loss(recon, x)
    kl = kl_loss(out["mu"], out["logvar"])
    beta = args.beta_kl * min(1.0, global_step / max(1, args.kl_warmup_steps))
    total = (
        args.l1_weight * l1
        + args.stft_weight * stft
        + args.log_fas_weight * fas
        + args.envelope_weight * env
        + beta * kl
    )
    metrics = {
        "loss": float(total.detach().item()),
        "l1": float(l1.detach().item()),
        "stft": float(stft.detach().item()),
        "log_fas": float(fas.detach().item()),
        "envelope": float(env.detach().item()),
        "kl": float(kl.detach().item()),
        "beta_kl": float(beta),
    }
    return total, out, metrics


def build_experiments(args):
    if args.preset not in {"seismic_quick", "seismic_smoke"}:
        raise ValueError(f"Unsupported preset: {args.preset}")
    base_configs = SEISMIC_BASE_CONFIGS if args.preset == "seismic_quick" else SEISMIC_BASE_CONFIGS[:1]
    adv_modes = {
        "none": (False,),
        "adv": (True,),
        "both": (False, True),
    }[args.include_adv]
    experiments = []
    for base_name, latent_channels, strides, base_channels in base_configs:
        for use_adv in adv_modes:
            suffix = "adv" if use_adv else "nonadv"
            experiments.append(
                VAEExperiment(
                    name=f"{base_name}_{suffix}",
                    latent_channels=latent_channels,
                    strides=tuple(strides),
                    base_channels=base_channels,
                    use_adv=use_adv,
                )
            )
    return experiments


def make_experiment_args(args, exp: VAEExperiment):
    exp_args = copy.deepcopy(args)
    exp_args.experiment_name = exp.name
    exp_args.latent_channels = exp.latent_channels
    exp_args.strides = list(exp.strides)
    exp_args.base_channels = exp.base_channels
    exp_args.downsample_factor = exp.downsample_factor
    exp_args.latent_length = exp_args.length // exp.downsample_factor
    exp_args.use_adv = exp.use_adv
    return exp_args


def save_checkpoint(path, model, discriminator, optimizer_g, optimizer_d, scaler, args, global_step, epoch, best_score, latent_scale):
    raw_model = model.module if isinstance(model, DDP) else model
    raw_disc = discriminator.module if isinstance(discriminator, DDP) else discriminator
    payload = {
        "model": raw_model.state_dict(),
        "discriminator": raw_disc.state_dict() if raw_disc is not None else None,
        "optimizer": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": copy.deepcopy(args),
        "vae_config": raw_model.config(),
        "latent_shape": [raw_model.latent_channels, raw_model.latent_length],
        "latent_channels": raw_model.latent_channels,
        "latent_length": raw_model.latent_length,
        "downsample_factor": raw_model.downsample_factor,
        "strides": list(raw_model.strides),
        "latent_scale": latent_scale.tolist() if latent_scale is not None else None,
        "waveform_norm_mode": args.waveform_norm_mode,
        "waveform_mean_value": args.waveform_mean_value,
        "waveform_std_value": args.waveform_std_value,
        "waveform_scale_source": getattr(args, "waveform_scale_source", None),
        "waveform_scale_stats": getattr(args, "waveform_scale_stats", None),
        "loss_weights": {
            "l1": args.l1_weight,
            "stft": args.stft_weight,
            "log_fas": args.log_fas_weight,
            "envelope": args.envelope_weight,
            "beta_kl": args.beta_kl,
            "adv": args.adv_weight if args.use_adv else 0.0,
            "feature_matching": args.feature_matching_weight if args.use_adv else 0.0,
        },
        "global_step": global_step,
        "epoch": epoch,
        "best_score": best_score,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def evaluate_reconstruction(model, loader, args, device, max_batches, save_examples_dir=None):
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    loss_totals = {}
    metric_rows = []
    examples_saved = 0
    if save_examples_dir is not None:
        save_examples_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (x, _) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True).float()
        total, out, losses = reconstruction_loss(raw_model, x, args, global_step=args.max_train_steps)
        recon = out["recon"].detach().float().cpu().numpy()
        target = x.detach().float().cpu().numpy()
        batch_size = target.shape[0]
        for key, value in losses.items():
            loss_totals[key] = loss_totals.get(key, 0.0) + float(value) * batch_size
        for i in range(batch_size):
            real = denormalize_waveform_array(
                target[i].transpose(1, 0),
                mean_value=args.waveform_mean_value,
                std_value=args.waveform_std_value,
            )
            pred = denormalize_waveform_array(
                recon[i].transpose(1, 0),
                mean_value=args.waveform_mean_value,
                std_value=args.waveform_std_value,
            )
            row = compute_sample_metrics(real, pred, args.sampling_hz)
            metric_rows.append(row)
            if save_examples_dir is not None and examples_saved < args.num_recon_examples:
                prefix = f"example_{examples_saved:03d}"
                np.save(save_examples_dir / f"{prefix}_real.npy", real)
                np.save(save_examples_dir / f"{prefix}_recon.npy", pred)
                if plt is not None:
                    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
                    t = np.arange(real.shape[0]) / args.sampling_hz
                    for ch, ax in enumerate(axes):
                        ax.plot(t, real[:, ch], color="black", linewidth=0.8, label="real")
                        ax.plot(t, pred[:, ch], color="tab:red", linewidth=0.8, alpha=0.8, label="recon")
                        ax.grid(True, alpha=0.3)
                    axes[0].legend(loc="upper right")
                    axes[-1].set_xlabel("Time (s)")
                    fig.tight_layout()
                    fig.savefig(save_examples_dir / f"{prefix}.png", dpi=140)
                    plt.close(fig)
                examples_saved += 1

    raw_model.train()
    count = max(1, len(metric_rows))
    loss_summary = {key: value / count for key, value in loss_totals.items()}
    metric_summary = summarize_metric_rows(metric_rows) if metric_rows else {}
    return loss_summary, metric_summary, metric_rows


def score_summary(metric_summary):
    def mean_abs(key):
        return abs(float(metric_summary[key]["mean"]))

    psa = [mean_abs("psa_0p2_log_resid"), mean_abs("psa_0p5_log_resid"), mean_abs("psa_1p0_log_resid"), mean_abs("psa_2p0_log_resid")]
    return (
        mean_abs("pga_log_resid")
        + mean_abs("pgv_log_resid")
        + sum(psa) / len(psa)
        + 0.25 * float(metric_summary["log_fas_mse_0p1_20hz"]["mean"])
        - 0.1 * float(metric_summary["pcc"]["mean"])
    )


def flatten_summary_row(exp, args, best_score, loss_summary, metric_summary, ckpt_path):
    row = {
        "experiment": exp.name,
        "adv": bool(exp.use_adv),
        "latent_channels": exp.latent_channels,
        "latent_length": args.latent_length,
        "downsample_factor": exp.downsample_factor,
        "strides": "-".join(str(s) for s in exp.strides),
        "best_score": best_score,
        "checkpoint": str(ckpt_path),
    }
    for key, value in loss_summary.items():
        row[f"loss_{key}"] = value
    for key, stats in metric_summary.items():
        row[f"{key}_mean"] = stats["mean"]
        row[f"{key}_median"] = stats["median"]
        row[f"{key}_p90"] = stats["p90"]
    return row


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(path, rows):
    sorted_rows = sorted(rows, key=lambda row: float(row.get("best_score", float("inf"))))
    lines = [
        "# Waveform VAE Reconstruction Comparison",
        "",
        "| Rank | Experiment | Latent | Adv | Score | PCC | PGA | PGV | PSA0.2 | PSA1.0 | log-FAS |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(sorted_rows, start=1):
        latent = f"{row['latent_channels']}x{row['latent_length']} R={row['downsample_factor']}"
        lines.append(
            "| {rank} | {exp} | {latent} | {adv} | {score:.5f} | {pcc:.5f} | {pga:.5f} | {pgv:.5f} | {psa02:.5f} | {psa10:.5f} | {fas:.5f} |".format(
                rank=idx,
                exp=row["experiment"],
                latent=latent,
                adv=str(row["adv"]),
                score=float(row["best_score"]),
                pcc=float(row.get("pcc_mean", 0.0)),
                pga=float(row.get("pga_log_resid_mean", 0.0)),
                pgv=float(row.get("pgv_log_resid_mean", 0.0)),
                psa02=float(row.get("psa_0p2_log_resid_mean", 0.0)),
                psa10=float(row.get("psa_1p0_log_resid_mean", 0.0)),
                fas=float(row.get("log_fas_mse_0p1_20hz_mean", 0.0)),
            )
        )
    lines.extend(["", "Lower score is better. Score prioritizes PGA/PGV/PSA residuals, then log-FAS, then PCC."])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def train_one_experiment(exp, base_args, train_dataset, val_dataset, rank, world_size, device, use_ddp):
    args = make_experiment_args(base_args, exp)
    exp_dir = Path(args.output_dir) / exp.name
    ckpt_dir = exp_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    examples_dir = exp_dir / "recon_examples"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    per_process_batch = args.global_batch_size // world_size if use_ddp else args.global_batch_size
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    train_loader = make_loader(train_dataset, per_process_batch, args, sampler=train_sampler, shuffle=True, drop_last=True)
    val_loader = make_loader(val_dataset, per_process_batch, args, shuffle=False, drop_last=False) if rank == 0 else None
    stats_loader = make_loader(train_dataset, per_process_batch, args, shuffle=False, drop_last=False) if rank == 0 else None

    model = WaveformKLVAE1D(
        in_channels=args.in_channels,
        latent_channels=exp.latent_channels,
        input_length=args.length,
        base_channels=exp.base_channels,
        strides=exp.strides,
    ).to(device)
    discriminator = MultiScaleWaveformDiscriminator(in_channels=args.in_channels).to(device) if exp.use_adv else None
    if use_ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index)
        if discriminator is not None:
            discriminator = DDP(discriminator, device_ids=[device.index], output_device=device.index)

    optimizer_g = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.discriminator_lr, weight_decay=args.weight_decay) if discriminator is not None else None
    total_steps = max(1, args.max_train_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        min_ratio = args.min_lr / args.lr
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda) if optimizer_d is not None else None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "amp" and device.type == "cuda"))
    writer = SummaryWriter(str(exp_dir / "tensorboard")) if rank == 0 and SummaryWriter is not None else None

    best_score = float("inf")
    best_loss_summary = {}
    best_metric_summary = {}
    global_step = 0
    epoch = 0
    start = time.time()
    if rank == 0:
        print(
            f"[{exp.name}] start latent=({exp.latent_channels},{args.latent_length}) "
            f"strides={list(exp.strides)} adv={exp.use_adv} batch={args.global_batch_size}",
            flush=True,
        )

    model.train()
    if discriminator is not None:
        discriminator.train()

    while global_step < args.max_train_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for x, _ in train_loader:
            if global_step >= args.max_train_steps:
                break
            x = x.to(device, non_blocking=True).float()
            adv_active = discriminator is not None and global_step >= args.adv_warmup_steps

            optimizer_g.zero_grad(set_to_none=True)
            if optimizer_d is not None:
                optimizer_d.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(args.precision == "amp" and device.type == "cuda")):
                rec_loss, out, metrics = reconstruction_loss(model, x, args, global_step)
                recon = out["recon"]

            d_loss_value = 0.0
            if adv_active:
                set_requires_grad(discriminator, True)
                with torch.amp.autocast("cuda", enabled=(args.precision == "amp" and device.type == "cuda")):
                    real_outs = discriminator(x)
                    fake_outs = discriminator(recon.detach())
                    d_loss = discriminator_hinge_loss(real_outs, fake_outs)
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_d)
                if scheduler_d is not None:
                    scheduler_d.step()
                d_loss_value = float(d_loss.detach().item())

            g_total = rec_loss
            adv_value = 0.0
            fm_value = 0.0
            if adv_active:
                set_requires_grad(discriminator, False)
                with torch.amp.autocast("cuda", enabled=(args.precision == "amp" and device.type == "cuda")):
                    real_outs = discriminator(x)
                    fake_outs = discriminator(recon)
                    adv_loss = generator_adv_loss(fake_outs)
                    fm_loss = feature_matching_loss(real_outs, fake_outs)
                    g_total = g_total + args.adv_weight * adv_loss + args.feature_matching_weight * fm_loss
                adv_value = float(adv_loss.detach().item())
                fm_value = float(fm_loss.detach().item())
                set_requires_grad(discriminator, True)

            scaler.scale(g_total).backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer_g)
            scaler.update()
            scheduler_g.step()

            if rank == 0 and global_step % args.log_every == 0:
                elapsed = max(time.time() - start, 1e-6)
                metrics.update({"d_loss": d_loss_value, "g_adv": adv_value, "feature_matching": fm_value})
                msg = " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
                print(f"[{exp.name}] step={global_step:07d} epoch={epoch} lr={scheduler_g.get_last_lr()[0]:.6g} steps/s={(global_step + 1) / elapsed:.3f} {msg}", flush=True)
                if writer is not None:
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", value, global_step)
                    writer.add_scalar("train/lr", scheduler_g.get_last_lr()[0], global_step)

            should_validate = (
                val_dataset is not None
                and global_step > 0
                and (global_step % args.val_every == 0 or global_step + 1 == args.max_train_steps)
            )
            if should_validate:
                if rank == 0:
                    loss_summary, metric_summary, metric_rows = evaluate_reconstruction(
                        model,
                        val_loader,
                        args,
                        device,
                        max_batches=args.val_max_batches,
                        save_examples_dir=examples_dir,
                    )
                    score = score_summary(metric_summary)
                    write_csv(metrics_dir / "val_reconstruction_metrics.csv", metric_rows)
                    if writer is not None:
                        writer.add_scalar("val/score", score, global_step)
                        for key, value in loss_summary.items():
                            writer.add_scalar(f"val_loss/{key}", value, global_step)
                        for key, stats in metric_summary.items():
                            writer.add_scalar(f"val_metric/{key}_mean", stats["mean"], global_step)
                    print(f"[{exp.name}] validation step={global_step:07d} score={score:.6f}", flush=True)
                    if score < best_score:
                        best_score = score
                        best_loss_summary = loss_summary
                        best_metric_summary = metric_summary
                        latent_scale = compute_latent_channel_std(model.module if use_ddp else model, stats_loader, device, args.latent_stats_max_batches)
                        save_checkpoint(ckpt_dir / "best_val.pt", model, discriminator, optimizer_g, optimizer_d, scaler, args, global_step, epoch, best_score, latent_scale)
                if use_ddp:
                    dist.barrier()

            global_step += 1

        epoch += 1

    if rank == 0:
        if not best_metric_summary and val_loader is not None:
            best_loss_summary, best_metric_summary, metric_rows = evaluate_reconstruction(
                model,
                val_loader,
                args,
                device,
                max_batches=args.val_max_batches,
                save_examples_dir=examples_dir,
            )
            best_score = score_summary(best_metric_summary)
            write_csv(metrics_dir / "val_reconstruction_metrics.csv", metric_rows)
        latent_scale = compute_latent_channel_std(model.module if use_ddp else model, stats_loader, device, args.latent_stats_max_batches)
        save_checkpoint(ckpt_dir / "last.pt", model, discriminator, optimizer_g, optimizer_d, scaler, args, global_step, epoch, best_score, latent_scale)
        if not (ckpt_dir / "best_val.pt").exists():
            save_checkpoint(ckpt_dir / "best_val.pt", model, discriminator, optimizer_g, optimizer_d, scaler, args, global_step, epoch, best_score, latent_scale)
        if writer is not None:
            writer.close()

    if use_ddp:
        dist.barrier()
    return flatten_summary_row(exp, args, best_score, best_loss_summary, best_metric_summary, ckpt_dir / "best_val.pt") if rank == 0 else None


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare multiple seismic waveform VAE configurations.")
    parser.add_argument("--wf_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/vae_compare_quick")
    parser.add_argument("--preset", type=str, default="seismic_quick", choices=["seismic_quick", "seismic_smoke"])
    parser.add_argument("--include-adv", type=str, default="both", choices=["none", "adv", "both"])
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--discriminator-lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--stft-weight", type=float, default=0.1)
    parser.add_argument("--log-fas-weight", type=float, default=0.2)
    parser.add_argument("--envelope-weight", type=float, default=0.1)
    parser.add_argument("--beta-kl", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-steps", type=int, default=10000)
    parser.add_argument("--adv-weight", type=float, default=0.05)
    parser.add_argument("--feature-matching-weight", type=float, default=1.0)
    parser.add_argument("--adv-warmup-steps", type=int, default=2000)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-max-batches", type=int, default=32)
    parser.add_argument("--latent-stats-max-batches", type=int, default=64)
    parser.add_argument("--num-recon-examples", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-ddp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sta-lat-min", dest="sta_lat_min", default=18.0, type=float)
    parser.add_argument("--sta-lat-max", dest="sta_lat_max", default=55.0, type=float)
    parser.add_argument("--sta-lon-min", dest="sta_lon_min", default=122.0, type=float)
    parser.add_argument("--sta-lon-max", dest="sta_lon_max", default=156.0, type=float)
    parser.add_argument("--src-lat-min", dest="src_lat_min", default=18.0, type=float)
    parser.add_argument("--src-lat-max", dest="src_lat_max", default=55.0, type=float)
    parser.add_argument("--src-lon-min", dest="src_lon_min", default=122.0, type=float)
    parser.add_argument("--src-lon-max", dest="src_lon_max", default=156.0, type=float)
    parser.add_argument("--dep-min", dest="dep_min", default=-100.0, type=float)
    parser.add_argument("--dep-max", dest="dep_max", default=800.0, type=float)
    parser.add_argument("--mag-min", dest="mag_min", default=-1.0, type=float)
    parser.add_argument("--mag-max", dest="mag_max", default=10.0, type=float)
    parser.add_argument("--min-get", dest="min_get", default=0, type=float)
    parser.add_argument("--max-get", dest="max_get", default=1000, type=float)
    parser.add_argument("--waveform-scale-stats", type=str, default=None)
    parser.add_argument("--waveform-mean-value", type=float, default=None)
    parser.add_argument("--waveform-std-value", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    use_ddp, rank, world_size, device = setup_distributed(args)
    torch.manual_seed(args.seed + rank)
    conditioning_config = resolve_conditioning_config(args, require_ranges=True)
    apply_conditioning_config(args, conditioning_config)
    scale_config = resolve_waveform_scale_config(args, require_scale=True)
    apply_waveform_scale_config(args, scale_config)

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sweep_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_dataset = make_dataset(args, "train")
    val_dataset = make_dataset(args, "val")
    if train_dataset is None or val_dataset is None:
        raise ValueError("--wf_dir and --val_dir are required")

    experiments = build_experiments(args)
    rows = []
    try:
        for exp in experiments:
            row = train_one_experiment(exp, args, train_dataset, val_dataset, rank, world_size, device, use_ddp)
            if rank == 0 and row is not None:
                rows.append(row)
                output_dir = Path(args.output_dir)
                write_csv(output_dir / "vae_comparison.csv", rows)
                (output_dir / "vae_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
                write_markdown_report(output_dir / "vae_comparison.md", rows)
        if rank == 0:
            print(f"comparison report saved to {Path(args.output_dir) / 'vae_comparison.md'}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
