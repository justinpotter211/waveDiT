import argparse
import json
from pathlib import Path

import torch

from train_compare_waveform_vae import (
    VAEExperiment,
    apply_conditioning_config,
    apply_waveform_scale_config,
    cleanup_distributed,
    make_dataset,
    resolve_conditioning_config,
    resolve_waveform_scale_config,
    setup_distributed,
    train_one_experiment,
)


BEST_VAE_NAME = "light_r8_c8_nonadv"
BEST_LATENT_CHANNELS = 8
BEST_STRIDES = (2, 2, 2)
BEST_BASE_CHANNELS = 64
BEST_USE_ADV = False


def build_best_experiment() -> VAEExperiment:
    return VAEExperiment(
        name=BEST_VAE_NAME,
        latent_channels=BEST_LATENT_CHANNELS,
        strides=BEST_STRIDES,
        base_channels=BEST_BASE_CHANNELS,
        use_adv=BEST_USE_ADV,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected best seismic waveform KL-VAE configuration. "
            "Default architecture is light_r8_c8_nonadv: latent 8x375, R=8, no adversarial loss."
        )
    )
    parser.add_argument("--wf_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/waveform_vae_best")
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
    exp = build_best_experiment()
    use_ddp, rank, world_size, device = setup_distributed(args)
    torch.manual_seed(args.seed + rank)

    try:
        conditioning_config = resolve_conditioning_config(args, require_ranges=True)
        apply_conditioning_config(args, conditioning_config)
        scale_config = resolve_waveform_scale_config(args, require_scale=True)
        apply_waveform_scale_config(args, scale_config)

        train_dataset = make_dataset(args, "train")
        val_dataset = make_dataset(args, "val")
        if train_dataset is None or val_dataset is None:
            raise ValueError("--wf_dir and --val_dir are required")

        if rank == 0:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "selected_vae": exp.name,
                "selection_reason": "Lowest score in the existing VAE reconstruction comparison.",
                "latent_channels": exp.latent_channels,
                "strides": list(exp.strides),
                "downsample_factor": exp.downsample_factor,
                "base_channels": exp.base_channels,
                "use_adv": exp.use_adv,
                "args": vars(args),
            }
            (output_dir / "selected_vae_config.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

        row = train_one_experiment(exp, args, train_dataset, val_dataset, rank, world_size, device, use_ddp)
        if rank == 0 and row is not None:
            output_dir = Path(args.output_dir)
            (output_dir / "training_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"best VAE checkpoint saved to {output_dir / exp.name / 'checkpoints' / 'best_val.pt'}", flush=True)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
