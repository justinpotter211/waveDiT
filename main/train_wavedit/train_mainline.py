import argparse
import json
import logging
import math
import os
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from glob import glob
from pathlib import Path
from time import time

import torch
import torch.distributed as dist
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_load import LoadWaveform
from diffusion import create_diffusion
from long_tail_weighting import JointCellDataset, LongTailWeightedDataset, TOTAL_JOINT_CELLS
from models_mainline import build_waveform_model
from waveform_vae import latent_scale_from_checkpoint, latent_scale_view, load_waveform_vae_checkpoint
from waveform_scaling import (
    apply_conditioning_config,
    apply_waveform_scale_config,
    conditioning_dimension,
    resolve_conditioning_config,
    resolve_waveform_scale_config,
    utc_now_iso,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy("file_system")

MAINLINE_VARIANT = "mainline"
MAINLINE_DISPLAY_NAME = "mainline light conv patch stem + transformer backbone + scalar conditioning"
REPRESENTATION_WAVEFORM = "waveform"
REPRESENTATION_KLVAE_LATENT = "klvae_latent"
REPO_ROOT = Path(__file__).resolve().parents[1]


def epsilon_min_snr_weights(snr_by_t, t, gamma):
    if gamma <= 0:
        return torch.ones(t.shape[0], device=t.device, dtype=torch.float32)
    snr = snr_by_t.to(device=t.device)[t].float().clamp_min(1e-8)
    return torch.minimum(snr, torch.full_like(snr, float(gamma))) / snr


def conditioning_schema(args):
    if getattr(args, "abs_geo_encoder_type", "none") == "relative_geometry":
        return "coords_relative_geometry_v2"
    return f"legacy_{getattr(args, 'abs_geo_encoder_type', 'none')}"


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir, rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def resolve_amp_config(precision):
    precision = str(precision).lower()
    if precision not in {"auto", "fp16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "fp32":
        return False, torch.float32, "fp32", "fp32 (full precision)"
    if precision == "fp16":
        return True, torch.float16, "fp16", "fp16 (CUDA AMP mixed precision)"
    return True, torch.float16, "fp16", "fp16 (resolved from auto; CUDA AMP mixed precision)"


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def build_failure_info(
    rank,
    stage,
    epoch,
    batch_index,
    step,
    variant,
    x=None,
    y=None,
    loss=None,
    grad_norm=None,
    message=None,
):
    info = {
        "rank": int(rank),
        "stage": str(stage),
        "epoch": int(epoch),
        "batch": int(batch_index),
        "step": int(step),
        "variant": str(variant),
        "message": str(message) if message else None,
    }
    if x is not None:
        info["x_range"] = [float(x.min().item()), float(x.max().item())]
    if y is not None:
        info["y_range"] = [float(y.min().item()), float(y.max().item())]
    if loss is not None:
        info["loss"] = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
    if grad_norm is not None:
        info["grad_norm"] = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
    return info


def sync_failure_info(local_failure, use_ddp, world_size):
    if not use_ddp:
        return local_failure
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_failure)
    failures = [item for item in gathered if item is not None]
    if not failures:
        return None
    failures.sort(key=lambda item: (item["rank"], item["stage"], item["step"], item["batch"]))
    return failures[0]


def format_failure_message(failure):
    x_range = failure.get("x_range")
    y_range = failure.get("y_range")
    x_text = (
        f"x_range=({x_range[0]:.6f}, {x_range[1]:.6f})"
        if x_range is not None
        else "x_range=n/a"
    )
    y_text = (
        f"y_range=({y_range[0]:.6f}, {y_range[1]:.6f})"
        if y_range is not None
        else "y_range=n/a"
    )
    grad_text = (
        f", grad_norm={failure['grad_norm']:.6f}"
        if "grad_norm" in failure and failure["grad_norm"] is not None
        else ""
    )
    loss_text = (
        f", loss={failure['loss']:.6f}"
        if "loss" in failure and failure["loss"] is not None
        else ""
    )
    extra = f", detail={failure['message']}" if failure.get("message") else ""
    return (
        f"Non-finite training state detected on rank={failure['rank']} during {failure['stage']} "
        f"at epoch={failure['epoch']}, batch={failure['batch']}, step={failure['step']}, "
        f"variant={failure['variant']}, {x_text}, {y_text}{loss_text}{grad_text}{extra}."
    )


def find_nonfinite_gradient(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return name
    return None


def configure_representation(args):
    representation = getattr(args, "representation", REPRESENTATION_WAVEFORM)
    if representation == REPRESENTATION_WAVEFORM:
        args.model_in_channels = 3
        args.model_length = args.length
        args.model_patch_size = args.patch_size
        return
    if representation != REPRESENTATION_KLVAE_LATENT:
        raise ValueError(f"Unsupported representation: {representation}")
    if not getattr(args, "autoencoder_ckpt", None):
        raise ValueError("--autoencoder-ckpt is required when --representation klvae_latent")
    args.model_in_channels = args.latent_channels
    args.model_length = args.latent_length
    args.model_patch_size = args.latent_patch_size


def build_latent_codec(args, device):
    if getattr(args, "representation", REPRESENTATION_WAVEFORM) != REPRESENTATION_KLVAE_LATENT:
        return None
    torch_device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    vae, payload = load_waveform_vae_checkpoint(args.autoencoder_ckpt, device=torch_device, dtype=torch.float32)
    if vae.latent_channels != args.latent_channels or vae.latent_length != args.latent_length:
        raise ValueError(
            "VAE latent shape does not match training args: "
            f"vae=({vae.latent_channels},{vae.latent_length}) "
            f"args=({args.latent_channels},{args.latent_length})"
        )
    if vae.input_length != args.length:
        raise ValueError(f"VAE input_length={vae.input_length} does not match waveform length={args.length}")
    latent_scale = latent_scale_from_checkpoint(payload, args.latent_channels)
    args.latent_scale = latent_scale.tolist()
    return {
        "model": vae,
        "scale": latent_scale_view(latent_scale, torch_device, torch.float32),
        "checkpoint": args.autoencoder_ckpt,
    }


@torch.no_grad()
def prepare_diffusion_input(x, latent_codec=None):
    if latent_codec is None:
        return x
    vae = latent_codec["model"]
    scale = latent_codec["scale"]
    with torch.amp.autocast(device_type="cuda", enabled=False):
        z = vae.encode_to_latent(x.float(), sample=False)
    return z / scale.to(device=z.device, dtype=z.dtype)


@torch.no_grad()
def evaluate_on_loader(
    model,
    diffusion,
    loader,
    device,
    rank,
    use_ddp,
    world_size,
    variant,
    amp_enabled=False,
    amp_dtype=torch.float16,
    max_batches=0,
    latent_codec=None,
):
    model.eval()
    loss_sum = 0.0
    loss_count = 0
    cell_loss_sum = torch.zeros(TOTAL_JOINT_CELLS, device=device, dtype=torch.float64)
    cell_loss_count = torch.zeros(TOTAL_JOINT_CELLS, device=device, dtype=torch.float64)
    for i, batch in enumerate(loader):
        x, y, cell_id, sample_index = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cell_id = cell_id.to(device, non_blocking=True)
        sample_index = sample_index.to(device, non_blocking=True)

        local_failure = None
        if not torch.isfinite(x).all():
            local_failure = build_failure_info(
                rank, "validation_input", -1, i, -1, variant, x=x, y=y, message="non-finite waveform"
            )
        elif not torch.isfinite(y).all():
            local_failure = build_failure_info(
                rank, "validation_input", -1, i, -1, variant, x=x, y=y, message="non-finite conditioning label"
            )
        failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
        if failure is not None:
            return 0.0, 0, cell_loss_sum, cell_loss_count, failure

        x = prepare_diffusion_input(x, latent_codec)
        local_failure = None
        if not torch.isfinite(x).all():
            local_failure = build_failure_info(
                rank, "validation_representation", -1, i, -1, variant, x=x, y=y, message="non-finite diffusion input"
            )
        failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
        if failure is not None:
            return 0.0, 0, cell_loss_sum, cell_loss_count, failure

        t = ((sample_index * 9973 + 20260415) % diffusion.num_timesteps).long()
        model_kwargs = dict(y=y)
        amp_context = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        try:
            with amp_context:
                loss_vec = diffusion.training_losses(model, x, t, model_kwargs)["mse"]
        except FloatingPointError as exc:
            local_failure = build_failure_info(
                rank,
                "validation_forward",
                -1,
                i,
                -1,
                variant,
                x=x,
                y=y,
                message=str(exc),
            )
            failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
            return 0.0, 0, cell_loss_sum, cell_loss_count, failure

        local_failure = None
        if not torch.isfinite(loss_vec).all():
            local_failure = build_failure_info(
                rank,
                "validation_loss",
                -1,
                i,
                -1,
                variant,
                x=x,
                y=y,
                loss=loss_vec.mean(),
                message="non-finite validation loss",
            )
        failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
        if failure is not None:
            return 0.0, 0, cell_loss_sum, cell_loss_count, failure

        detached_loss = loss_vec.detach().double()
        loss_sum += float(detached_loss.sum().item())
        loss_count += int(detached_loss.numel())
        cell_loss_sum.scatter_add_(0, cell_id, detached_loss)
        cell_loss_count.scatter_add_(0, cell_id, torch.ones_like(detached_loss))
        if max_batches > 0 and (i + 1) >= max_batches:
            break
    return loss_sum, loss_count, cell_loss_sum, cell_loss_count, None


def _load_checkpoint(model, checkpoint_path, allow_partial_load):
    state_dict = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False
    )["ema"]
    if allow_partial_load:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        return missing, unexpected
    model.load_state_dict(state_dict, strict=True)
    return [], []


def create_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps, base_lr, min_lr):
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            warmup_progress = current_step / max(1, warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * warmup_progress
        cosine_steps = max(1, total_steps - warmup_steps)
        cosine_progress = min((current_step - warmup_steps) / cosine_steps, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def sync_scheduler_state(scheduler, use_ddp, rank):
    if not use_ddp:
        return
    state = [scheduler.state_dict() if rank == 0 else None]
    dist.broadcast_object_list(state, src=0)
    if rank != 0:
        scheduler.load_state_dict(state[0])


def model_without_ddp(model, use_ddp):
    return model.module if use_ddp else model


def infer_resume_steps(checkpoint, checkpoint_path):
    if "train_steps" in checkpoint:
        return int(checkpoint["train_steps"])
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem.isdigit():
        return int(stem)
    scheduler_state = checkpoint.get("scheduler")
    if isinstance(scheduler_state, dict) and "last_epoch" in scheduler_state:
        return max(0, int(scheduler_state["last_epoch"]))
    return 0


def load_training_checkpoint(checkpoint_path, model, ema, opt, scheduler, scaler, use_ddp, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    model_without_ddp(model, use_ddp).load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    opt.load_state_dict(checkpoint["opt"])
    target_device = torch.device("cuda", device)
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    train_steps = infer_resume_steps(checkpoint, checkpoint_path)
    return {
        "train_steps": train_steps,
        "best_val_loss": float(checkpoint.get("best_val_macro_loss", checkpoint.get("best_val_loss", float("inf")))),
        "best_val_micro_loss": float(checkpoint.get("best_val_micro_loss", float("inf"))),
        "early_stop_counter": int(checkpoint.get("early_stop_counter", 0)),
    }


def write_scale_metadata(experiment_dir, args):
    payload = {
        "waveform_norm_mode": args.waveform_norm_mode,
        "waveform_mean_value": args.waveform_mean_value,
        "waveform_std_value": args.waveform_std_value,
        "waveform_scale_source": args.waveform_scale_source,
        "waveform_scale_stats": args.waveform_scale_stats,
        "conditioning_source": getattr(args, "conditioning_source", None),
        "conditioning_stats_path": getattr(args, "conditioning_stats_path", None),
        "variant": MAINLINE_VARIANT,
        "representation": getattr(args, "representation", REPRESENTATION_WAVEFORM),
        "autoencoder_ckpt": getattr(args, "autoencoder_ckpt", None),
        "latent_channels": getattr(args, "latent_channels", None),
        "latent_length": getattr(args, "latent_length", None),
        "latent_patch_size": getattr(args, "latent_patch_size", None),
        "latent_scale": getattr(args, "latent_scale", None),
        "model_in_channels": getattr(args, "model_in_channels", None),
        "model_length": getattr(args, "model_length", None),
        "model_patch_size": getattr(args, "model_patch_size", None),
        "abs_geo_encoder_type": getattr(args, "abs_geo_encoder_type", None),
        "conditioning_dimension": conditioning_dimension(getattr(args, "abs_geo_encoder_type", "none")),
        "conditioning_schema": conditioning_schema(args),
        "azimuth_encoding": "raw_sin_cos_pair" if getattr(args, "abs_geo_encoder_type", "none") == "relative_geometry" else None,
        "azimuth_embedder": "pair_mlp" if getattr(args, "abs_geo_encoder_type", "none") == "relative_geometry" else None,
        "prediction_type": "epsilon",
        "class_dropout_prob": getattr(args, "class_dropout_prob", None),
        "min_snr_gamma": getattr(args, "min_snr_gamma", None),
        "long_tail_weight_manifest": getattr(args, "long_tail_weight_manifest", None),
        "long_tail_alpha": getattr(args, "long_tail_alpha", None),
        "long_tail_weight_min": getattr(args, "long_tail_weight_min", None),
        "long_tail_weight_max": getattr(args, "long_tail_weight_max", None),
        "position_embedding": getattr(args, "position_embedding", None),
        "rope_base": getattr(args, "rope_base", None),
        "min_get": getattr(args, "min_get", None),
        "max_get": getattr(args, "max_get", None),
        "cond_embedding_scale": getattr(args, "cond_embedding_scale", None),
        "conditioning_defaults": {
            "sta_lat_min": args.sta_lat_min,
            "sta_lat_max": args.sta_lat_max,
            "sta_lon_min": args.sta_lon_min,
            "sta_lon_max": args.sta_lon_max,
            "src_lat_min": args.src_lat_min,
            "src_lat_max": args.src_lat_max,
            "src_lon_min": args.src_lon_min,
            "src_lon_max": args.src_lon_max,
            "dep_min": args.dep_min,
            "dep_max": args.dep_max,
            "mag_min": args.mag_min,
            "mag_max": args.mag_max,
            "epi_dist_min": getattr(args, "epi_dist_min", None),
            "epi_dist_max": getattr(args, "epi_dist_max", None),
        },
        "created_at_utc": utc_now_iso(),
    }
    metadata_path = os.path.join(experiment_dir, "waveform_scale_config.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return metadata_path


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    if args.resume and args.pretrained_model:
        raise ValueError("Use either --resume for continuing training or --pretrained_model for warm start, not both.")
    if not 0.0 <= args.class_dropout_prob < 1.0:
        raise ValueError(f"--class-dropout-prob must be in [0, 1), got {args.class_dropout_prob}")
    if args.min_snr_gamma < 0.0:
        raise ValueError(f"--min-snr-gamma must be non-negative, got {args.min_snr_gamma}")
    if not 0.0 < args.long_tail_alpha <= 1.0:
        raise ValueError(f"--long-tail-alpha must be in (0, 1], got {args.long_tail_alpha}")
    if not 0.0 < args.long_tail_weight_min <= args.long_tail_weight_max:
        raise ValueError(
            "Invalid long-tail weight bounds: "
            f"[{args.long_tail_weight_min}, {args.long_tail_weight_max}]"
        )
    use_ddp = not args.no_ddp
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        timeout_minutes = int(os.environ.get("DDP_INIT_TIMEOUT_MINUTES", "10"))
        print(
            "Before DDP init: "
            f"pid={os.getpid()} "
            f"rank_env={os.environ.get('RANK')} "
            f"local_rank={local_rank} "
            f"world_size_env={os.environ.get('WORLD_SIZE')} "
            f"master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')} "
            f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}",
            flush=True,
        )
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_minutes))
        assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = local_rank
        print(
            f"After DDP init: rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}.",
            flush=True,
        )
    else:
        rank = 0
        world_size = 1
        device = 0
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    amp_enabled, amp_dtype, _precision_name, precision_message = resolve_amp_config(args.precision)

    if rank == 0:
        if args.resume:
            checkpoint_dir = os.path.dirname(os.path.abspath(args.resume))
            experiment_dir = os.path.dirname(checkpoint_dir)
        else:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            model_base_dir = args.model_dir if args.model_dir else args.results_dir
            checkpoint_dir = f"{model_base_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        if args.resume:
            logger.info(f"Resuming experiment at {experiment_dir}")
            logger.info(f"Continuing checkpoints in {checkpoint_dir}")
        else:
            logger.info(f"Experiment directory created at {experiment_dir}")
            logger.info(f"Checkpoint directory created at {checkpoint_dir}")
    else:
        logger = create_logger(None, rank)
        experiment_dir = None
        checkpoint_dir = None

    conditioning_config = resolve_conditioning_config(args, require_ranges=True)
    apply_conditioning_config(args, conditioning_config)
    scale_config = resolve_waveform_scale_config(args, require_scale=True)
    apply_waveform_scale_config(args, scale_config)
    configure_representation(args)
    latent_codec = build_latent_codec(args, device)
    if rank == 0:
        metadata_path = write_scale_metadata(experiment_dir, args)
        logger.info(
            "Conditioning ranges resolved from %s (stats=%s)",
            args.conditioning_source,
            args.conditioning_stats_path,
        )
        logger.info(
            "Waveform global Z-score enabled: mean=%.9f std=%.9f source=%s",
            args.waveform_mean_value,
            args.waveform_std_value,
            args.waveform_scale_source,
        )
        logger.info(
            "Resolved training config: depth=%d hidden_size=%d num_heads=%d length=%d position_embedding=%s rope_base=%.1f condition_dim=%d min_get=%.3f max_get=%.3f waveform_std=%.6f",
            args.depth,
            args.hidden_size,
            args.num_heads,
            args.length,
            args.position_embedding,
            args.rope_base,
            conditioning_dimension(args.abs_geo_encoder_type),
            float(args.min_get),
            float(args.max_get),
            args.waveform_std_value,
        )
        logger.info(
            "Representation=%s model_input=(channels=%d,length=%d,patch_size=%d) autoencoder=%s",
            args.representation,
            args.model_in_channels,
            args.model_length,
            args.model_patch_size,
            getattr(args, "autoencoder_ckpt", None),
        )
        logger.info(f"Waveform scaling metadata saved to {metadata_path}")

    display_name = MAINLINE_DISPLAY_NAME
    model = build_waveform_model(args)
    if args.pretrained_model:
        missing, unexpected = _load_checkpoint(
            model, args.pretrained_model, args.allow_partial_load
        )
        if rank == 0:
            if args.allow_partial_load:
                logger.info(
                    f"Partial checkpoint load: missing={len(missing)}, unexpected={len(unexpected)}"
                )
            else:
                logger.info("Checkpoint loaded with strict=True.")

    ema = deepcopy(model).to(device)
    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))], gradient_as_bucket_view=True)
    diffusion = create_diffusion(timestep_respacing="")
    diffusion_alpha_cumprod = torch.as_tensor(diffusion.alphas_cumprod, dtype=torch.float32)
    diffusion_snr_by_t = diffusion_alpha_cumprod / (1.0 - diffusion_alpha_cumprod).clamp_min(1e-8)
    logger.info(f"variant={MAINLINE_VARIANT}, display_name={display_name}")
    logger.info(f"{MAINLINE_VARIANT} Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training precision: {precision_message}")
    logger.info(
        "module_config abs_geo_encoder_type=%s position_embedding=%s rope_base=%.1f",
        getattr(args, "abs_geo_encoder_type", "n/a"),
        getattr(args, "position_embedding", "sincos"),
        float(getattr(args, "rope_base", 10000.0)),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    dataset = LoadWaveform(
        args.wf_dir,
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
        args.abs_geo_encoder_type,
        getattr(args, "epi_dist_min", -100.0),
        getattr(args, "epi_dist_max", 2500.0),
    )
    long_tail_audit = None
    if args.long_tail_weight_manifest:
        dataset = LongTailWeightedDataset(
            dataset,
            args.long_tail_weight_manifest,
            alpha=args.long_tail_alpha,
            weight_min=args.long_tail_weight_min,
            weight_max=args.long_tail_weight_max,
        )
        long_tail_audit = dataset.audit
        if rank == 0:
            logger.info(
                "Long-tail weighting enabled: cells=%d alpha=%.3f bounds=[%.3f, %.3f] manifest=%s",
                int(long_tail_audit["nonempty_cell_count"]),
                args.long_tail_alpha,
                args.long_tail_weight_min,
                args.long_tail_weight_max,
                args.long_tail_weight_manifest,
            )
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed,
        )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    val_sampler = None
    val_dir = args.val_dir if args.val_dir else os.path.join(os.path.dirname(args.wf_dir), "val")
    if os.path.isdir(val_dir):
        val_dataset = LoadWaveform(
            val_dir,
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
            args.abs_geo_encoder_type,
            getattr(args, "epi_dist_min", -100.0),
            getattr(args, "epi_dist_max", 2500.0),
        )
        val_dataset = JointCellDataset(val_dataset)
        if use_ddp:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=args.global_seed,
            )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.global_batch_size // world_size),
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    logger.info(f"Dataset contains {len(dataset):,} waveform ({args.wf_dir})")
    if val_loader is not None:
        logger.info(f"Validation contains {len(val_dataset):,} waveform ({val_dir})")
    else:
        logger.info(f"Validation disabled, val_dir not found: {val_dir}")
    early_stop_enabled = val_loader is not None and args.early_stop_patience > 0
    if early_stop_enabled:
        logger.info(
            "Early stopping enabled: patience=%d validations, min_delta=%.6g",
            args.early_stop_patience,
            args.early_stop_min_delta,
        )
    elif val_loader is None:
        logger.info("Early stopping disabled because validation is unavailable.")
    else:
        logger.info("Early stopping disabled because early_stop_patience <= 0.")

    log_every = max(1, args.log_every)
    total_train_steps = args.max_train_steps if args.max_train_steps > 0 else len(loader) * args.epochs
    total_train_steps = max(1, total_train_steps)
    warmup_steps = max(1, math.ceil(total_train_steps * args.warmup_ratio))
    lr_scheduler = create_warmup_cosine_scheduler(
        opt,
        total_steps=total_train_steps,
        warmup_steps=warmup_steps,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )
    sync_scheduler_state(lr_scheduler, use_ddp, rank)
    logger.info(
        "LR schedule: base_lr=%.6g min_lr=%.6g total_steps=%d warmup_steps=%d warmup_ratio=%.4f",
        args.lr,
        args.min_lr,
        total_train_steps,
        warmup_steps,
        args.warmup_ratio,
    )
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    best_val_loss = float("inf")
    best_val_micro_loss = float("inf")
    early_stop_counter = 0
    early_stop_should_stop = False
    training_error = None
    if args.resume:
        resume_state = load_training_checkpoint(args.resume, model, ema, opt, lr_scheduler, scaler, use_ddp, device)
        train_steps = resume_state["train_steps"]
        best_val_loss = resume_state["best_val_loss"]
        best_val_micro_loss = resume_state["best_val_micro_loss"]
        early_stop_counter = resume_state["early_stop_counter"]
        logger.info(
            "Loaded resume checkpoint %s at step=%07d best_val_loss=%.6g early_stop_counter=%d",
            args.resume,
            train_steps,
            best_val_loss,
            early_stop_counter,
        )
    else:
        update_ema(ema, model_without_ddp(model, use_ddp), decay=0)
    model.train()
    ema.eval()

    logger.info(f"Training variant={MAINLINE_VARIANT}, display_name={display_name} for {args.epochs} epochs...")
    start_epoch = train_steps // max(1, len(loader))
    skip_batches = train_steps % max(1, len(loader))
    if args.resume and skip_batches > 0:
        logger.info(
            "Resume starts inside epoch %d; skipping %d already-trained batches in that epoch.",
            start_epoch,
            skip_batches,
        )
    for epoch in range(start_epoch, args.epochs):
        early_stop_should_stop = False
        if sampler is not None:
            sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch_index, batch in enumerate(tqdm(loader, disable=rank != 0)):
            if args.resume and epoch == start_epoch and batch_index < skip_batches:
                continue

            if len(batch) == 4:
                x, y, sample_weight, cell_id = batch
            else:
                x, y = batch
                sample_weight = torch.ones(x.shape[0], dtype=torch.float32)
                cell_id = torch.full((x.shape[0],), -1, dtype=torch.long)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            cell_id = cell_id.to(device, non_blocking=True)

            local_failure = None
            if not torch.isfinite(x).all():
                local_failure = build_failure_info(
                    rank, "input", epoch, batch_index, train_steps, MAINLINE_VARIANT, x=x, y=y, message="non-finite waveform"
                )
            elif not torch.isfinite(y).all():
                local_failure = build_failure_info(
                    rank, "input", epoch, batch_index, train_steps, MAINLINE_VARIANT, x=x, y=y, message="non-finite conditioning label"
                )
            failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
            if failure is not None:
                training_error = failure
                break

            x = prepare_diffusion_input(x, latent_codec)
            local_failure = None
            if not torch.isfinite(x).all():
                local_failure = build_failure_info(
                    rank,
                    "representation",
                    epoch,
                    batch_index,
                    train_steps,
                    MAINLINE_VARIANT,
                    x=x,
                    y=y,
                    message="non-finite diffusion input",
                )
            failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
            if failure is not None:
                training_error = failure
                break

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            try:
                with amp_context:
                    loss_terms = diffusion.training_losses(model, x, t, model_kwargs)
                    timestep_weight = epsilon_min_snr_weights(
                        diffusion_snr_by_t,
                        t,
                        args.min_snr_gamma,
                    ).to(dtype=loss_terms["mse"].dtype)
                    loss_vec = timestep_weight * loss_terms["mse"]
                    if "vb" in loss_terms:
                        loss_vec = loss_vec + loss_terms["vb"]
                    sample_weight = sample_weight.to(dtype=loss_vec.dtype)
                    loss = (sample_weight * loss_vec).sum() / sample_weight.sum().clamp_min(1e-8)
            except FloatingPointError as exc:
                local_failure = build_failure_info(
                    rank,
                    "forward",
                    epoch,
                    batch_index,
                    train_steps,
                    MAINLINE_VARIANT,
                    x=x,
                    y=y,
                    message=str(exc),
                )
                failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
                if failure is not None:
                    training_error = failure
                    break

            local_failure = None
            if not torch.isfinite(loss):
                local_failure = build_failure_info(
                    rank,
                    "loss",
                    epoch,
                    batch_index,
                    train_steps,
                    MAINLINE_VARIANT,
                    x=x,
                    y=y,
                    loss=loss,
                    message="non-finite training loss",
                )
            failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
            if failure is not None:
                training_error = failure
                break

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            local_failure = None
            nonfinite_grad_name = find_nonfinite_gradient(model)
            if nonfinite_grad_name is not None:
                local_failure = build_failure_info(
                    rank,
                    "gradient",
                    epoch,
                    batch_index,
                    train_steps,
                    MAINLINE_VARIANT,
                    x=x,
                    y=y,
                    loss=loss,
                    grad_norm=grad_norm,
                    message=f"non-finite gradient in {nonfinite_grad_name}",
                )
            elif not torch.isfinite(grad_norm):
                local_failure = build_failure_info(
                    rank,
                    "gradient",
                    epoch,
                    batch_index,
                    train_steps,
                    MAINLINE_VARIANT,
                    x=x,
                    y=y,
                    loss=loss,
                    grad_norm=grad_norm,
                    message="non-finite gradient norm",
                )
            failure = sync_failure_info(local_failure, use_ddp=use_ddp, world_size=world_size)
            if failure is not None:
                opt.zero_grad(set_to_none=True)
                training_error = failure
                break

            scaler.step(opt)
            scaler.update()
            lr_scheduler.step()
            if use_ddp and train_steps + 1 == warmup_steps:
                dist.barrier()
                sync_scheduler_state(lr_scheduler, use_ddp, rank)
            update_ema(ema, model_without_ddp(model, use_ddp))

            running_loss += float(loss.item())
            log_steps += 1
            train_steps += 1

            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / max(end_time - start_time, 1e-8)
                avg_loss = torch.tensor(running_loss / max(log_steps, 1), device=device)
                if use_ddp:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                else:
                    avg_loss = avg_loss.item()
                current_lr = opt.param_groups[0]["lr"]
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.6g}"
                )
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model_without_ddp(model, use_ddp).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_macro_loss": best_val_loss,
                        "best_val_micro_loss": best_val_micro_loss,
                        "early_stop_counter": early_stop_counter,
                        "early_stop_patience": args.early_stop_patience,
                        "early_stop_min_delta": args.early_stop_min_delta,
                        "epoch": epoch,
                        "train_steps": train_steps,
                        "args": args,
                        "representation": args.representation,
                        "autoencoder_ckpt": getattr(args, "autoencoder_ckpt", None),
                        "latent_channels": getattr(args, "latent_channels", None),
                        "latent_length": getattr(args, "latent_length", None),
                        "latent_patch_size": getattr(args, "latent_patch_size", None),
                        "latent_scale": getattr(args, "latent_scale", None),
                        "model_in_channels": getattr(args, "model_in_channels", None),
                        "model_length": getattr(args, "model_length", None),
                        "model_patch_size": getattr(args, "model_patch_size", None),
                        "abs_geo_encoder_type": getattr(args, "abs_geo_encoder_type", "none"),
                        "epi_dist_min": getattr(args, "epi_dist_min", None),
                        "epi_dist_max": getattr(args, "epi_dist_max", None),
                        "position_embedding": getattr(args, "position_embedding", None),
                        "rope_base": getattr(args, "rope_base", None),
                        "conditioning_schema": conditioning_schema(args),
                        "azimuth_encoding": "raw_sin_cos_pair" if args.abs_geo_encoder_type == "relative_geometry" else None,
                        "azimuth_embedder": "pair_mlp" if args.abs_geo_encoder_type == "relative_geometry" else None,
                        "prediction_type": "epsilon",
                        "class_dropout_prob": args.class_dropout_prob,
                        "min_snr_gamma": args.min_snr_gamma,
                        "long_tail_weight_manifest": args.long_tail_weight_manifest,
                        "long_tail_alpha": args.long_tail_alpha,
                        "long_tail_weight_min": args.long_tail_weight_min,
                        "long_tail_weight_max": args.long_tail_weight_max,
                        "long_tail_manifest_sha256": (
                            long_tail_audit.get("ordered_filename_sha256") if long_tail_audit else None
                        ),
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if use_ddp:
                    dist.barrier()

            if args.max_train_steps > 0 and train_steps >= args.max_train_steps:
                break

        if training_error is not None:
            break

        if val_loader is not None and ((epoch + 1) % max(1, args.val_every) == 0):
            val_loss_sum, val_loss_count, val_cell_sum, val_cell_count, val_failure = evaluate_on_loader(
                ema,
                diffusion,
                val_loader,
                device,
                rank=rank,
                use_ddp=use_ddp,
                world_size=world_size,
                variant=MAINLINE_VARIANT,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                max_batches=args.val_max_batches,
                latent_codec=latent_codec,
            )
            if val_failure is not None:
                training_error = val_failure
                break
            val_metric = torch.tensor([val_loss_sum, float(val_loss_count)], device=device)
            if use_ddp:
                dist.all_reduce(val_metric, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_cell_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_cell_count, op=dist.ReduceOp.SUM)
            total_sum = float(val_metric[0].item())
            total_count = int(val_metric[1].item())
            avg_val_micro = total_sum / max(1, total_count)
            nonempty_cells = val_cell_count > 0
            avg_val_macro = float(
                (val_cell_sum[nonempty_cells] / val_cell_count[nonempty_cells]).mean().item()
            )
            best_val_micro_loss = min(best_val_micro_loss, avg_val_micro)
            micro_guard_passed = avg_val_micro <= 1.05 * best_val_micro_loss
            if rank == 0:
                logger.info(
                    "(epoch=%04d) Val epsilon MSE: micro=%.4f macro=%.4f cells=%d micro_guard=%s",
                    epoch,
                    avg_val_micro,
                    avg_val_macro,
                    int(nonempty_cells.sum().item()),
                    micro_guard_passed,
                )
                if micro_guard_passed and avg_val_macro < best_val_loss - args.early_stop_min_delta:
                    best_val_loss = avg_val_macro
                    early_stop_counter = 0
                    best_path = f"{checkpoint_dir}/best_val.pt"
                    checkpoint = {
                        "model": model_without_ddp(model, use_ddp).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "args": args,
                        "best_val_loss": best_val_loss,
                        "best_val_macro_loss": best_val_loss,
                        "best_val_micro_loss": best_val_micro_loss,
                        "early_stop_counter": early_stop_counter,
                        "early_stop_patience": args.early_stop_patience,
                        "early_stop_min_delta": args.early_stop_min_delta,
                        "epoch": epoch,
                        "train_steps": train_steps,
                        "representation": args.representation,
                        "autoencoder_ckpt": getattr(args, "autoencoder_ckpt", None),
                        "latent_channels": getattr(args, "latent_channels", None),
                        "latent_length": getattr(args, "latent_length", None),
                        "latent_patch_size": getattr(args, "latent_patch_size", None),
                        "latent_scale": getattr(args, "latent_scale", None),
                        "model_in_channels": getattr(args, "model_in_channels", None),
                        "model_length": getattr(args, "model_length", None),
                        "model_patch_size": getattr(args, "model_patch_size", None),
                        "abs_geo_encoder_type": getattr(args, "abs_geo_encoder_type", "none"),
                        "epi_dist_min": getattr(args, "epi_dist_min", None),
                        "epi_dist_max": getattr(args, "epi_dist_max", None),
                        "position_embedding": getattr(args, "position_embedding", None),
                        "rope_base": getattr(args, "rope_base", None),
                        "conditioning_schema": conditioning_schema(args),
                        "azimuth_encoding": "raw_sin_cos_pair" if args.abs_geo_encoder_type == "relative_geometry" else None,
                        "azimuth_embedder": "pair_mlp" if args.abs_geo_encoder_type == "relative_geometry" else None,
                        "prediction_type": "epsilon",
                        "class_dropout_prob": args.class_dropout_prob,
                        "min_snr_gamma": args.min_snr_gamma,
                        "long_tail_weight_manifest": args.long_tail_weight_manifest,
                        "long_tail_alpha": args.long_tail_alpha,
                        "long_tail_weight_min": args.long_tail_weight_min,
                        "long_tail_weight_max": args.long_tail_weight_max,
                        "long_tail_manifest_sha256": (
                            long_tail_audit.get("ordered_filename_sha256") if long_tail_audit else None
                        ),
                    }
                    torch.save(checkpoint, best_path)
                    logger.info(f"Saved best validation checkpoint to {best_path}")
                elif early_stop_enabled:
                    early_stop_counter += 1
                    logger.info(
                        "Early stopping counter: %d/%d (best_val_macro_mse=%.4f)",
                        early_stop_counter,
                        args.early_stop_patience,
                        best_val_loss,
                    )
                    early_stop_should_stop = early_stop_counter >= args.early_stop_patience
                    if early_stop_should_stop:
                        logger.info(
                            "Early stopping triggered at epoch=%04d, step=%07d.",
                            epoch,
                            train_steps,
                        )
            if use_ddp:
                stop_tensor = torch.tensor(
                    [1 if early_stop_should_stop else 0],
                    dtype=torch.int,
                    device=device,
                )
                dist.broadcast(stop_tensor, src=0)
                early_stop_should_stop = bool(stop_tensor.item())
            model.train()

        if early_stop_should_stop:
            break

        if args.max_train_steps > 0 and train_steps >= args.max_train_steps:
            break

    model.eval()
    if training_error is not None:
        error_message = format_failure_message(training_error)
        if rank == 0:
            logger.error(error_message)
        cleanup()
        raise FloatingPointError(error_message)

    logger.info("Done!")
    cleanup()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--global-batch-size", default=256, type=int)
    parser.add_argument("--global_seed", default=0, type=int)
    parser.add_argument("--model", type=str, default="waveDiT")
    parser.add_argument("--pretrained_model", type=str)
    parser.add_argument("--resume", type=str, default=None, help="Path to a training checkpoint to continue from.")
    parser.add_argument("--allow-partial-load", action="store_true")
    parser.add_argument("--is_transfer", default=True, type=bool)
    parser.add_argument("--results_dir", default=str(REPO_ROOT / "result"), type=str)
    parser.add_argument("--model_dir", default=None, type=str)
    parser.add_argument("--wf_dir", default=str(REPO_ROOT / "data" / "train"), type=str)
    parser.add_argument("--val_dir", default=str(REPO_ROOT / "data" / "val"), type=str)
    parser.add_argument("--sta_lat_min", default=18.0, type=float)
    parser.add_argument("--sta_lat_max", default=55.0, type=float)
    parser.add_argument("--sta_lon_min", default=122.0, type=float)
    parser.add_argument("--sta_lon_max", default=156.0, type=float)
    parser.add_argument("--src_lat_min", default=18.0, type=float)
    parser.add_argument("--src_lat_max", default=55.0, type=float)
    parser.add_argument("--src_lon_min", default=122.0, type=float)
    parser.add_argument("--src_lon_max", default=156.0, type=float)
    parser.add_argument("--dep_min", default=-100.0, type=float)
    parser.add_argument("--dep_max", default=800.0, type=float)
    parser.add_argument("--mag_min", default=-1.0, type=float)
    parser.add_argument("--mag_max", default=10.0, type=float)
    parser.add_argument("--epi-dist-min", dest="epi_dist_min", default=-100.0, type=float)
    parser.add_argument("--epi-dist-max", dest="epi_dist_max", default=2500.0, type=float)
    parser.add_argument("--min_get", default=0, type=int)
    parser.add_argument("--max_get", default=1000, type=int)
    parser.add_argument("--cond-embedding-scale", dest="cond_embedding_scale", default=1.0, type=float)
    parser.add_argument(
        "--position-embedding",
        dest="position_embedding",
        default="sincos",
        choices=["sincos", "rope"],
        help="Patch-token position encoding: fixed absolute sincos or standard 1D RoPE in attention.",
    )
    parser.add_argument("--rope-base", dest="rope_base", default=10000.0, type=float)
    parser.add_argument("--length", default=3000, type=int)
    parser.add_argument("--waveform-scale-stats", type=str, default=None)
    parser.add_argument("--waveform-mean-value", type=float, default=None)
    parser.add_argument("--waveform-std-value", type=float, default=None)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--depth", default=24, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--hidden_feature", default=168, type=int)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--class-dropout-prob", type=float, default=0.10)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--long-tail-weight-manifest", type=str, default=None)
    parser.add_argument("--long-tail-alpha", type=float, default=0.75)
    parser.add_argument("--long-tail-weight-min", type=float, default=0.25)
    parser.add_argument("--long-tail-weight-max", type=float, default=4.0)
    parser.add_argument("--no-ddp", action="store_true")
    parser.add_argument(
        "--representation",
        type=str,
        default=REPRESENTATION_WAVEFORM,
        choices=[REPRESENTATION_WAVEFORM, REPRESENTATION_KLVAE_LATENT],
    )
    parser.add_argument("--autoencoder-ckpt", dest="autoencoder_ckpt", type=str, default=None)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--latent-length", type=int, default=375)
    parser.add_argument("--latent-patch-size", type=int, default=1)

    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp16", "fp32"],
        help="Training precision. 'auto' resolves to CUDA AMP mixed precision (fp16) on the supported GPU path.",
    )
    parser.add_argument(
        "--abs-geo-encoder-type",
        dest="abs_geo_encoder_type",
        type=str,
        default="none",
        choices=["none", "epicentral_distance", "relative_geometry"],
    )
    parser.add_argument("--patch_size", type=int, default=6)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
