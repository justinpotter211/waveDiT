from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from validation.run_baseline_eval import (
    AUTO_VARIANT,
    DEFAULT_SUBSET_SEED,
    MODEL_VARIANTS,
    apply_checkpoint_model_args,
    batched,
    build_sample_decoder,
    configure_representation,
    generate_predictions_for_records,
    load_checkpoint_payload,
    load_model_and_diffusion,
    resolve_model_variant,
)
from validation.subset import (
    CaseRecord,
    collect_case_records,
    select_stratified_subset,
    write_selected_cases_csv,
)
from waveform_scaling import (
    apply_conditioning_config,
    apply_waveform_scale_config,
    get_default_runtime_stats_path,
    resolve_conditioning_config,
    resolve_waveform_scale_config,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class GenerationBatchError(RuntimeError):
    def __init__(self, failures: list[dict[str, object]]):
        self.failures = failures
        last_error = failures[-1]["error"] if failures else "unknown generation error"
        super().__init__(f"Waveform generation failed: {last_error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SWaG prediction waveforms as .npy files without plotting or reports."
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=256)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--subset-seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument(
        "--latent-seed",
        type=int,
        default=None,
        help="Base seed for record initial noise and deterministic reverse-process noise.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument(
        "--num-sampling-steps",
        "--num_sampling_steps",
        dest="num_sampling_steps",
        type=int,
        default=1000,
    )
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument(
        "--runtime-stats",
        type=Path,
        default=get_default_runtime_stats_path(),
        help="Dataset runtime stats JSON containing conditioning defaults and waveform Z-score stats.",
    )
    parser.add_argument("--model-variant", type=str, default=AUTO_VARIANT, choices=MODEL_VARIANTS)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume safely by skipping only complete batches whose saved predictions are valid.",
    )

    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--hidden_size", "--hidden-size", dest="hidden_size", type=int, default=768)
    parser.add_argument("--num_heads", "--num-heads", dest="num_heads", type=int, default=12)
    parser.add_argument("--hidden_feature", "--hidden-feature", dest="hidden_feature", type=int, default=168)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=6)
    parser.add_argument(
        "--abs-geo-encoder-type",
        dest="abs_geo_encoder_type",
        default="none",
        choices=["none", "epicentral_distance", "relative_geometry"],
        type=str,
    )
    parser.add_argument(
        "--position-embedding",
        dest="position_embedding",
        default="sincos",
        choices=["sincos", "rope"],
        type=str,
    )
    parser.add_argument("--rope-base", dest="rope_base", default=10000.0, type=float)
    parser.add_argument("--min-get", type=float, default=0.0)
    parser.add_argument("--max-get", type=float, default=1000.0)
    parser.add_argument("--cond-embedding-scale", default=1.0, type=float)

    parser.add_argument("--sta-lat-min", default=18.0, type=float)
    parser.add_argument("--sta-lat-max", default=55.0, type=float)
    parser.add_argument("--sta-lon-min", default=122.0, type=float)
    parser.add_argument("--sta-lon-max", default=156.0, type=float)
    parser.add_argument("--src-lat-min", default=18.0, type=float)
    parser.add_argument("--src-lat-max", default=55.0, type=float)
    parser.add_argument("--src-lon-min", default=122.0, type=float)
    parser.add_argument("--src-lon-max", default=156.0, type=float)
    parser.add_argument("--dep-min", default=-100.0, type=float)
    parser.add_argument("--dep-max", default=800.0, type=float)
    parser.add_argument("--mag-min", default=-1.0, type=float)
    parser.add_argument("--mag-max", default=10.0, type=float)
    parser.add_argument("--epi-dist-min", default=-100.0, type=float)
    parser.add_argument("--epi-dist-max", default=2500.0, type=float)
    parser.add_argument("--waveform-mean-value", type=float, default=None)
    parser.add_argument("--waveform-std-value", type=float, default=None)
    parser.add_argument("--representation", choices=["waveform", "klvae_latent"], default="waveform")
    parser.add_argument("--autoencoder-ckpt", dest="autoencoder_ckpt", type=str, default=None)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--latent-length", type=int, default=375)
    parser.add_argument("--latent-patch-size", type=int, default=1)
    return parser


def normalize_args(args: argparse.Namespace) -> None:
    args.input_dir = args.input_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.ckpt = args.ckpt.resolve()
    args.runtime_stats = args.runtime_stats.resolve()
    args.waveform_scale_stats = str(args.runtime_stats)
    if args.latent_seed is None:
        args.latent_seed = args.subset_seed


def ensure_environment(args: argparse.Namespace) -> None:
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")
    if not args.runtime_stats.exists():
        raise FileNotFoundError(f"Runtime stats JSON not found: {args.runtime_stats}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.subset_size < 1:
        raise ValueError(f"--subset-size must be >= 1, got {args.subset_size}")
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard_index < num_shards, got {args.shard_index}/{args.num_shards}"
        )
    if args.num_sampling_steps < 2:
        raise ValueError(
            f"--num-sampling-steps must be >= 2 for the current diffusion implementation, got {args.num_sampling_steps}."
        )
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. This script is intended for GPU waveform generation; "
            "pass --allow-cpu only for a tiny local smoke test."
        )


def select_records(args: argparse.Namespace) -> list[CaseRecord]:
    records = collect_case_records(args.input_dir)
    if not records:
        raise RuntimeError(f"No .npy files found under {args.input_dir}")
    selected = records if args.full else select_stratified_subset(records, args.subset_size, args.subset_seed)
    return selected[args.shard_index :: args.num_shards]


def shard_suffix(args: argparse.Namespace) -> str:
    if args.num_shards == 1:
        return ""
    return f"_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"


def resolve_checkpoint_and_configs(args: argparse.Namespace) -> dict[str, object]:
    ckpt_payload = load_checkpoint_payload(args.ckpt)
    apply_checkpoint_model_args(args, ckpt_payload)
    if getattr(args, "abs_geo_encoder_type", "none") == "relative_geometry":
        schema = ckpt_payload.get("conditioning_schema")
        if schema != "coords_relative_geometry_v2":
            raise RuntimeError(
                "Relative-geometry checkpoint must declare conditioning_schema="
                f"'coords_relative_geometry_v2', got {schema!r}."
            )
    if ckpt_payload.get("prediction_type", "epsilon") != "epsilon":
        raise RuntimeError("Waveform generation currently supports epsilon-prediction checkpoints only.")
    state_dict = ckpt_payload["ema"]
    args.model_variant = resolve_model_variant(args.model_variant, state_dict)
    configure_representation(args)

    ckpt_args = ckpt_payload.get("args")
    conditioning_config = resolve_conditioning_config(
        args,
        checkpoint_args=ckpt_args,
        prefer_checkpoint=True,
        require_ranges=True,
    )
    apply_conditioning_config(args, conditioning_config)
    scale_config = resolve_waveform_scale_config(
        args,
        checkpoint_args=ckpt_args,
        prefer_checkpoint=True,
        require_scale=True,
    )
    apply_waveform_scale_config(args, scale_config)
    return ckpt_payload


def infer_device_and_precision() -> tuple[str, torch.dtype, str]:
    if torch.cuda.is_available():
        return "cuda", torch.float16, "fp16"
    return "cpu", torch.float32, "fp32"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_manifest(
    args: argparse.Namespace,
    selected_records: list[CaseRecord],
    device: str,
    precision_name: str,
) -> dict[str, object]:
    suffix = shard_suffix(args)
    return {
        "created_at_utc": utc_now_iso(),
        "script": str(Path(__file__).resolve()),
        "ckpt": str(args.ckpt),
        "input_dir": str(args.input_dir),
        "out_dir": str(args.out_dir),
        "pred_npy_dir": str((args.out_dir / "pred_npy").resolve()),
        "selected_cases_csv": str((args.out_dir / f"selected_cases{suffix}.csv").resolve()),
        "mode": "full" if args.full else "subset",
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_count": len(selected_records),
        "subset_size": args.subset_size,
        "subset_seed": args.subset_seed,
        "latent_seed": args.latent_seed,
        "reverse_noise_seed_rule": (
            "sha256(latent_seed:shard_index:first_record_name:batch_size), first 8 bytes little-endian"
        ),
        "skip_existing": args.skip_existing,
        "batch_size": args.batch_size,
        "cfg_scale": args.cfg_scale,
        "num_sampling_steps": args.num_sampling_steps,
        "sampling_hz": args.sampling_hz,
        "length": args.length,
        "device": device,
        "precision": precision_name,
        "model_variant": args.model_variant,
        "abs_geo_encoder_type": getattr(args, "abs_geo_encoder_type", "none"),
        "conditioning_dimension": getattr(args, "conditioning_dimension", 6),
        "position_embedding": getattr(args, "position_embedding", "sincos"),
        "rope_base": getattr(args, "rope_base", 10000.0),
        "runtime_stats": str(args.runtime_stats),
        "conditioning": {
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
            "min_get": args.min_get,
            "max_get": args.max_get,
            "cond_embedding_scale": args.cond_embedding_scale,
            "source": getattr(args, "conditioning_source", None),
            "stats_path": getattr(args, "conditioning_stats_path", None),
        },
        "waveform_normalization": {
            "mode": args.waveform_norm_mode,
            "mean_value": args.waveform_mean_value,
            "std_value": args.waveform_std_value,
            "source": getattr(args, "waveform_scale_source", None),
            "stats_path": getattr(args, "waveform_scale_stats", None),
        },
    }


def validate_prediction(record: CaseRecord, pred: np.ndarray, args: argparse.Namespace) -> None:
    expected_shape = (args.length, 3)
    if pred.shape != expected_shape:
        raise ValueError(f"Prediction shape mismatch for {record.name}: got {pred.shape}, expected {expected_shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"Prediction contains NaN or Inf for {record.name}")


def prediction_path(record: CaseRecord, args: argparse.Namespace) -> Path:
    return args.out_dir / "pred_npy" / f"{record.name}_pred.npy"


def saved_prediction_is_valid(record: CaseRecord, args: argparse.Namespace) -> bool:
    path = prediction_path(record, args)
    if not path.is_file():
        return False
    try:
        pred = np.load(path, allow_pickle=False)
        validate_prediction(record, pred, args)
    except Exception:
        return False
    return True


def reverse_noise_seed(args: argparse.Namespace, batch: list[CaseRecord]) -> int:
    payload = f"{args.latent_seed}:{args.shard_index}:{batch[0].name}:{len(batch)}".encode("utf-8")
    # torch.manual_seed accepts signed 64-bit-safe non-negative values.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False) % (2**63 - 1)


def generate_waveforms(
    args: argparse.Namespace,
    selected_records: list[CaseRecord],
    state_dict: dict[str, torch.Tensor],
    device: str,
    model_dtype: torch.dtype,
    decoder=None,
) -> tuple[int, list[dict[str, object]]]:
    pred_dir = args.out_dir / "pred_npy"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model, diffusion = load_model_and_diffusion(args, state_dict, device=device, model_dtype=model_dtype)

    failures: list[dict[str, object]] = []
    success_count = 0
    total = len(selected_records)
    for batch in batched(selected_records, args.batch_size):
        if args.skip_existing and all(saved_prediction_is_valid(record, args) for record in batch):
            success_count += len(batch)
            if success_count % 20 == 0 or success_count == total:
                print(f"available {success_count}/{total} (valid existing predictions reused)")
            continue
        try:
            batch_seed = reverse_noise_seed(args, batch)
            torch.manual_seed(batch_seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(batch_seed)
            pred_batch = generate_predictions_for_records(
                batch,
                args=args,
                model=model,
                diffusion=diffusion,
                device=device,
                model_dtype=model_dtype,
                decoder=decoder,
            )
            for record, pred in zip(batch, pred_batch):
                validate_prediction(record, pred, args)
                np.save(prediction_path(record, args), pred)
                success_count += 1
        except Exception as exc:
            failure = {
                "records": [record.name for record in batch],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            if not args.continue_on_error:
                raise GenerationBatchError(failures) from exc
        if success_count % 20 == 0 or success_count == total:
            print(f"generated {success_count}/{total}")
    return success_count, failures


def count_successful_records(args: argparse.Namespace, records: list[CaseRecord]) -> int:
    pred_dir = args.out_dir / "pred_npy"
    return sum(1 for record in records if (pred_dir / f"{record.name}_pred.npy").is_file())


def run(args: argparse.Namespace) -> Path:
    normalize_args(args)
    ensure_environment(args)
    ckpt_payload = resolve_checkpoint_and_configs(args)
    selected_records = select_records(args)
    suffix = shard_suffix(args)
    selected_cases_path = args.out_dir / f"selected_cases{suffix}.csv"
    manifest_path = args.out_dir / f"generation_manifest{suffix}.json"
    summary_path = args.out_dir / f"generation_summary{suffix}.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_selected_cases_csv(selected_records, selected_cases_path)

    device, model_dtype, precision_name = infer_device_and_precision()
    decoder = build_sample_decoder(args, ckpt_payload, device)
    manifest = build_manifest(args, selected_records, device, precision_name)
    write_json(manifest_path, manifest)

    time0 = time.time()
    failures: list[dict[str, object]] = []
    try:
        success_count, failures = generate_waveforms(
            args,
            selected_records=selected_records,
            state_dict=ckpt_payload["ema"],
            device=device,
            model_dtype=model_dtype,
            decoder=decoder,
        )
        status = "completed" if not failures else "completed_with_failures"
    except GenerationBatchError as exc:
        success_count = count_successful_records(args, selected_records)
        failures = exc.failures
        status = "failed"
        raise
    except Exception:
        success_count = count_successful_records(args, selected_records)
        failures.append(
            {
                "records": [],
                "error": "fatal_error",
                "traceback": traceback.format_exc(),
            }
        )
        status = "failed"
        raise
    finally:
        elapsed_seconds = time.time() - time0
        attempted_count = len(selected_records)
        summary = {
            "status": status if "status" in locals() else "failed",
            "created_at_utc": manifest["created_at_utc"],
            "completed_at_utc": utc_now_iso(),
            "out_dir": str(args.out_dir),
            "pred_npy_dir": str((args.out_dir / "pred_npy").resolve()),
            "selected_cases_csv": str(selected_cases_path.resolve()),
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "latent_seed": args.latent_seed,
            "attempted_count": attempted_count,
            "success_count": success_count if "success_count" in locals() else 0,
            "failure_count": len(failures),
            "failures": failures,
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_success": (
                elapsed_seconds / success_count
                if "success_count" in locals() and success_count > 0
                else None
            ),
            "device": device,
            "precision": precision_name,
        }
        write_json(summary_path, summary)
    return args.out_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = run(args)
    print(f"generation output saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
