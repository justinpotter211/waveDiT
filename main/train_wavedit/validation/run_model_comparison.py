from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TRAIN_WAVEDIT_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]

if str(TRAIN_WAVEDIT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_WAVEDIT_DIR))


DEFAULT_TEST_ROOT = REPO_ROOT / "data" / "test"
DEFAULT_RUNTIME_STATS = REPO_ROOT / "data" / "dataset_runtime_stats.json"
TEST_CATEGORY_NAMES = (
    "future_known_station",
    "future_unknown_station",
    "historical_unknown_station",
)
SUMMARY_METRIC_KEYS = [
    "pcc",
    "rmse",
    "mae",
    "pga_log_resid",
    "pgv_log_resid",
    "psa_0p2_log_resid",
    "psa_0p5_log_resid",
    "psa_1p0_log_resid",
    "psa_2p0_log_resid",
    "log_fas_mse_0p1_20hz",
]
SIGNED_BIAS_METRICS = {
    "pga_log_resid",
    "pgv_log_resid",
    "psa_0p2_log_resid",
    "psa_0p5_log_resid",
    "psa_1p0_log_resid",
    "psa_2p0_log_resid",
}
LOWER_IS_BETTER = {
    "rmse",
    "mae",
    "log_fas_mse_0p1_20hz",
    *SIGNED_BIAS_METRICS,
}
KEY_REPORT_METRICS = ("pcc", "rmse", "mae", "log_fas_mse_0p1_20hz")
CHECKPOINT_DIR_CANDIDATES = (
    Path("checkpoints") / "best_val.pt",
    Path("best_val.pt"),
    Path("checkpoint.pt"),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    ckpt: str
    variant: str = "auto"
    representation: str = "waveform"
    autoencoder: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SWaG ablation checkpoint comparison with one independent model worker per GPU. "
            "This is not a torchrun/DDP entrypoint."
        )
    )
    parser.add_argument("--model", action="append", default=None, help="Model spec: name=...,ckpt=...,variant=auto,repr=waveform")
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--runtime-stats", type=Path, default=DEFAULT_RUNTIME_STATS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--gpu-mode",
        choices=("sequential_sharded", "one_model_per_gpu"),
        default="one_model_per_gpu",
        help=(
            "one_model_per_gpu runs multiple models concurrently with one GPU per model. "
            "sequential_sharded is reserved and disabled to avoid using one model on all GPUs by mistake."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--position-embedding", dest="position_embedding", type=str, default="sincos", choices=["sincos", "rope"])
    parser.add_argument("--rope-base", dest="rope_base", type=float, default=10000.0)
    parser.add_argument("--cond-embedding-scale", default=1.0, type=float)
    parser.add_argument("--subset-seed", type=int, default=20260415)
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--repeat-subset-size", type=int, default=512)
    parser.add_argument("--repeat-latent-seed-start", type=int, default=20260501)
    parser.add_argument("--full-test-limit", type=int, default=None, help="Smoke-test limit per category; omit for full test.")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate prediction .npy files and selected_cases.csv; skip metrics, figures, and reports.",
    )
    parser.add_argument("--save-pred-npy", action="store_true")
    parser.add_argument("--all-sample-figures", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def parse_key_value_spec(raw: str) -> ModelSpec:
    values: dict[str, str] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid model spec fragment without '=': {part!r}")
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()

    name = values.get("name")
    ckpt = values.get("ckpt")
    if not name or not ckpt:
        raise ValueError(f"Model spec must include name and ckpt: {raw!r}")
    representation = values.get("repr", values.get("representation", "waveform"))
    autoencoder = values.get("autoencoder", values.get("autoencoder_ckpt"))
    if representation == "klvae_latent" and not autoencoder:
        raise ValueError(f"VAE latent model '{name}' must include autoencoder=/path/to/vae.pt")
    return ModelSpec(
        name=name,
        ckpt=ckpt,
        variant=values.get("variant", "auto"),
        representation=representation,
        autoencoder=autoencoder,
    )


def parse_gpus(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def resolve_path_string(raw: str) -> str:
    return str(Path(raw).expanduser().resolve())


def resolve_checkpoint_string(raw: str, model_name: str) -> str:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        return str(path)
    if not path.exists():
        return str(path)
    if not path.is_dir():
        return str(path)

    for relative_path in CHECKPOINT_DIR_CANDIDATES:
        candidate = path / relative_path
        if candidate.is_file():
            return str(candidate.resolve())

    pt_candidates = sorted(candidate for candidate in path.glob("*.pt") if candidate.is_file())
    if len(pt_candidates) == 1:
        return str(pt_candidates[0].resolve())
    if not pt_candidates:
        preferred = ", ".join(str(path / candidate) for candidate in CHECKPOINT_DIR_CANDIDATES)
        raise FileNotFoundError(
            f"Checkpoint directory for model '{model_name}' contains no checkpoint file. "
            f"Expected one of: {preferred}"
        )

    candidate_list = ", ".join(str(candidate) for candidate in pt_candidates)
    raise RuntimeError(
        f"Checkpoint directory for model '{model_name}' has multiple .pt files and no preferred checkpoint. "
        f"Add checkpoints/best_val.pt, best_val.pt, or checkpoint.pt, or pass one file explicitly. "
        f"Candidates: {candidate_list}"
    )


def normalize_model_paths(model: ModelSpec) -> ModelSpec:
    return ModelSpec(
        name=model.name,
        ckpt=resolve_checkpoint_string(model.ckpt, model.name),
        variant=model.variant,
        representation=model.representation,
        autoencoder=resolve_path_string(model.autoencoder) if model.autoencoder is not None else None,
    )


def validate_inputs(args: argparse.Namespace, models: list[ModelSpec], gpus: list[str]) -> None:
    if args.gpu_mode != "one_model_per_gpu":
        raise ValueError(
            "--gpu-mode sequential_sharded is disabled for this comparison script. "
            "Use --gpu-mode one_model_per_gpu so each model occupies exactly one GPU."
        )
    if not args.generate_only and len(gpus) < len(models):
        raise ValueError(f"Need at least {len(models)} GPU ids for {len(models)} models, got {len(gpus)}")
    if args.generate_only:
        if len(gpus) < 1:
            raise ValueError("At least one GPU id is required in generate-only mode")
        if args.repeat_count < 0:
            raise ValueError("--repeat-count must be >= 0 in generate-only mode")
    elif args.repeat_count < 1:
        raise ValueError("--repeat-count must be >= 1")
    if args.repeat_subset_size < 1:
        raise ValueError("--repeat-subset-size must be >= 1")
    if args.full_test_limit is not None and args.full_test_limit < 1:
        raise ValueError("--full-test-limit must be >= 1 when provided")
    if not args.test_root.exists():
        raise FileNotFoundError(f"Test root not found: {args.test_root}")
    if not args.runtime_stats.exists():
        raise FileNotFoundError(f"Runtime stats not found: {args.runtime_stats}")
    for category in TEST_CATEGORY_NAMES:
        category_dir = args.test_root / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Missing test category directory: {category_dir}")
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError(f"Model names must be unique, got: {names}")
    for model in models:
        ckpt_path = Path(model.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for model '{model.name}': {ckpt_path}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint path for model '{model.name}' must resolve to a file, got: {ckpt_path}"
            )
        if model.representation == "klvae_latent":
            if model.autoencoder is None:
                raise ValueError(f"VAE latent model '{model.name}' must include autoencoder path")
            if not Path(model.autoencoder).exists():
                raise FileNotFoundError(f"Autoencoder checkpoint not found for model '{model.name}': {model.autoencoder}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def runtime_eval_defaults(runtime_stats: Path) -> dict[str, object]:
    payload = read_json(runtime_stats)
    conditioning_defaults = payload.get("conditioning_defaults")
    if not isinstance(conditioning_defaults, dict):
        raise ValueError(f"conditioning_defaults missing in runtime stats: {runtime_stats}")
    if "waveform_mean_value" not in payload:
        raise ValueError(f"waveform_mean_value missing in runtime stats: {runtime_stats}")
    if "waveform_std_value" not in payload:
        raise ValueError(f"waveform_std_value missing in runtime stats: {runtime_stats}")
    return payload


def build_eval_args(
    config: dict[str, object],
    model: ModelSpec,
    category: str,
    out_dir: Path,
    full_test: bool,
    subset_size: int,
    latent_seed: int,
):
    import argparse as _argparse

    stats_payload = runtime_eval_defaults(Path(str(config["runtime_stats"])))
    conditioning_defaults = stats_payload["conditioning_defaults"]
    return _argparse.Namespace(
        ckpt=Path(model.ckpt),
        model_variant=model.variant,
        test_dir=Path(str(config["test_root"])) / category,
        out_dir=out_dir,
        subset_size=subset_size,
        full_test=full_test,
        batch_size=int(config["batch_size"]),
        cfg_scale=float(config["cfg_scale"]),
        condition_mode="normal",
        num_sampling_steps=int(config["num_sampling_steps"]),
        sampling_hz=float(config["sampling_hz"]),
        length=int(config["length"]),
        depth=12,
        hidden_size=384,
        num_heads=6,
        hidden_feature=168,
        patch_size=8,
        abs_geo_encoder_type="none",
        position_embedding=str(config.get("position_embedding", "sincos")),
        rope_base=float(config.get("rope_base", 10000.0)),
        subset_seed=int(config["subset_seed"]),
        latent_seed=latent_seed,
        result_tag=f"{model.name}_{category}",
        save_pred_npy=bool(config["save_pred_npy"]),
        all_sample_figures=bool(config["all_sample_figures"]),
        sta_lat_min=conditioning_defaults["sta_lat_min"],
        sta_lat_max=conditioning_defaults["sta_lat_max"],
        sta_lon_min=conditioning_defaults["sta_lon_min"],
        sta_lon_max=conditioning_defaults["sta_lon_max"],
        src_lat_min=conditioning_defaults["src_lat_min"],
        src_lat_max=conditioning_defaults["src_lat_max"],
        src_lon_min=conditioning_defaults["src_lon_min"],
        src_lon_max=conditioning_defaults["src_lon_max"],
        dep_min=conditioning_defaults["dep_min"],
        dep_max=conditioning_defaults["dep_max"],
        mag_min=conditioning_defaults["mag_min"],
        mag_max=conditioning_defaults["mag_max"],
        epi_dist_min=conditioning_defaults.get("epi_dist_min", -100.0),
        epi_dist_max=conditioning_defaults.get("epi_dist_max", 2500.0),
        min_get=0.0,
        max_get=1000.0,
        cond_embedding_scale=float(config.get("cond_embedding_scale", 1.0)),
        waveform_scale_stats=str(config["runtime_stats"]),
        waveform_mean_value=stats_payload["waveform_mean_value"],
        waveform_std_value=stats_payload["waveform_std_value"],
        representation=model.representation,
        autoencoder_ckpt=model.autoencoder,
        latent_channels=8,
        latent_length=375,
        latent_patch_size=1,
        update_history=False,
    )


def limited_records_if_needed(test_dir: Path, limit: int | None, seed: int):
    if limit is None:
        return None
    from validation.subset import collect_case_records, select_stratified_subset

    records = collect_case_records(test_dir)
    return select_stratified_subset(records, limit, seed)


def select_records_for_worker(config: dict[str, object]):
    from validation.subset import collect_case_records, select_stratified_subset

    test_dir = Path(str(config["test_root"])) / str(config["category"])
    records = collect_case_records(test_dir)
    if not records:
        raise RuntimeError(f"No .npy test files found under {test_dir}")

    if bool(config["full_test"]):
        limit = config.get("full_test_limit")
        selected = (
            records
            if limit is None
            else select_stratified_subset(records, int(limit), int(config["subset_seed"]))
        )
    else:
        selected = select_stratified_subset(
            records,
            int(config["subset_size"]),
            int(config["subset_seed"]),
        )

    shard_index = int(config.get("shard_index", 0))
    shard_count = int(config.get("shard_count", 1))
    shard_records = selected[shard_index::shard_count]
    if not shard_records:
        raise RuntimeError(
            f"Shard {shard_index}/{shard_count} received no records for {config['category']} "
            f"from {len(selected)} selected records."
        )
    return shard_records


def select_records_for_generation(
    test_dir: Path,
    full_test: bool,
    subset_size: int,
    subset_seed: int,
    full_test_limit: int | None,
):
    from validation.subset import collect_case_records, select_stratified_subset

    records = collect_case_records(test_dir)
    if not records:
        raise RuntimeError(f"No .npy test files found under {test_dir}")
    if full_test:
        if full_test_limit is None:
            return records
        return select_stratified_subset(records, full_test_limit, subset_seed)
    return select_stratified_subset(records, subset_size, subset_seed)


def prepare_generation_components(eval_args):
    from validation.run_baseline_eval import (
        apply_checkpoint_model_args,
        build_sample_decoder,
        configure_representation,
        infer_device_and_precision,
        load_checkpoint_payload,
        load_model_and_diffusion,
        resolve_model_variant,
    )
    from waveform_scaling import (
        apply_conditioning_config,
        apply_waveform_scale_config,
        resolve_conditioning_config,
        resolve_waveform_scale_config,
    )

    ckpt_payload = load_checkpoint_payload(eval_args.ckpt)
    apply_checkpoint_model_args(eval_args, ckpt_payload)
    state_dict = ckpt_payload["ema"]
    eval_args.model_variant = resolve_model_variant(eval_args.model_variant, state_dict)
    configure_representation(eval_args)
    conditioning_config = resolve_conditioning_config(
        eval_args,
        checkpoint_args=ckpt_payload.get("args"),
        prefer_checkpoint=True,
        require_ranges=True,
    )
    apply_conditioning_config(eval_args, conditioning_config)
    scale_config = resolve_waveform_scale_config(
        eval_args,
        checkpoint_args=ckpt_payload.get("args"),
        prefer_checkpoint=True,
        require_scale=True,
    )
    apply_waveform_scale_config(eval_args, scale_config)
    device, model_dtype, precision_name, warnings = infer_device_and_precision()
    decoder = build_sample_decoder(eval_args, ckpt_payload, device)
    model, diffusion = load_model_and_diffusion(eval_args, state_dict, device=device, model_dtype=model_dtype)
    return model, diffusion, device, model_dtype, precision_name, warnings, decoder


def generate_records_to_dir(
    records,
    eval_args,
    model,
    diffusion,
    device: str,
    model_dtype,
    decoder,
    out_dir: Path,
    phase: str,
    category: str,
    repeat_index: int | None,
) -> dict[str, object]:
    import numpy as np
    from validation.run_baseline_eval import generate_predictions_for_records
    from validation.subset import write_selected_cases_csv

    def has_reusable_prediction(path: Path) -> bool:
        """Allow resume to replace interrupted zero-byte prediction files."""
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "pred_npy"
    pred_dir.mkdir(parents=True, exist_ok=True)
    selected_cases_path = out_dir / "selected_cases.csv"
    write_selected_cases_csv(records, selected_cases_path)

    total = len(records)
    batch_size = max(1, int(eval_args.batch_size))
    existing_count = sum(
        1 for record in records if has_reusable_prediction(pred_dir / f"{record.name}_pred.npy")
    )
    processed = existing_count
    generated_this_run = 0
    if existing_count:
        print(
            f"[{eval_args.ckpt.stem}] {phase} {category} "
            f"repeat={repeat_index if repeat_index is not None else '-'} "
            f"resuming with {existing_count}/{total} existing",
            flush=True,
        )
    reported_existing_complete = False
    for start in range(0, total, batch_size):
        batch = [
            record
            for record in records[start : start + batch_size]
            if not has_reusable_prediction(pred_dir / f"{record.name}_pred.npy")
        ]
        if not batch:
            if processed == total and not reported_existing_complete:
                print(
                    f"[{eval_args.ckpt.stem}] {phase} {category} "
                    f"repeat={repeat_index if repeat_index is not None else '-'} "
                    f"generated {processed}/{total}",
                    flush=True,
                )
                reported_existing_complete = True
            continue
        pred_batch = generate_predictions_for_records(
            batch,
            args=eval_args,
            model=model,
            diffusion=diffusion,
            device=device,
            model_dtype=model_dtype,
            decoder=decoder,
        )
        for record, pred in zip(batch, pred_batch):
            np.save(pred_dir / f"{record.name}_pred.npy", pred)
            processed += 1
            generated_this_run += 1
        if processed % 100 == 0 or processed == total:
            print(
                f"[{eval_args.ckpt.stem}] {phase} {category} "
                f"repeat={repeat_index if repeat_index is not None else '-'} "
                f"generated {processed}/{total}",
                flush=True,
            )

    manifest = {
        "phase": phase,
        "category": category,
        "repeat_index": repeat_index,
        "latent_seed": int(eval_args.latent_seed),
        "count": total,
        "out_dir": str(out_dir),
        "selected_cases_path": str(selected_cases_path),
        "pred_dir": str(pred_dir),
        "prediction_shape": "[T, C]",
        "abs_geo_encoder_type": getattr(eval_args, "abs_geo_encoder_type", "none"),
        "conditioning_dimension": getattr(eval_args, "conditioning_dimension", 6),
        "epi_dist_min": getattr(eval_args, "epi_dist_min", None),
        "epi_dist_max": getattr(eval_args, "epi_dist_max", None),
        "existing_predictions_before_run": existing_count,
        "generated_predictions_this_run": generated_this_run,
    }
    write_json(out_dir / "generation_manifest.json", manifest)
    return manifest


def apply_prepared_eval_config(prepared_args, eval_args) -> None:
    keep_values = {
        "test_dir": eval_args.test_dir,
        "out_dir": eval_args.out_dir,
        "full_test": eval_args.full_test,
        "subset_size": eval_args.subset_size,
        "latent_seed": eval_args.latent_seed,
        "result_tag": eval_args.result_tag,
    }
    for key, value in vars(prepared_args).items():
        setattr(eval_args, key, value)
    for key, value in keep_values.items():
        setattr(eval_args, key, value)


def run_generate_only_worker(config_path: Path) -> int:
    config = read_json(config_path)
    model_spec = ModelSpec(**config["model"])
    model_dir = Path(str(config["model_dir"]))
    full_results: list[dict[str, object]] = []
    repeat_results: list[dict[str, object]] = []

    print(
        f"[{model_spec.name}] generate-only worker started on "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}",
        flush=True,
    )

    setup_args = build_eval_args(
        config,
        model=model_spec,
        category=TEST_CATEGORY_NAMES[0],
        out_dir=model_dir,
        full_test=True,
        subset_size=1,
        latent_seed=int(config["subset_seed"]),
    )
    model, diffusion, device, model_dtype, precision_name, warnings, decoder = prepare_generation_components(setup_args)
    for warning in warnings:
        print(f"[{model_spec.name}] warning: {warning}", flush=True)

    for category in TEST_CATEGORY_NAMES:
        out_dir = model_dir / "full_test" / category
        eval_args = build_eval_args(
            config,
            model=model_spec,
            category=category,
            out_dir=out_dir,
            full_test=True,
            subset_size=1,
            latent_seed=int(config["subset_seed"]),
        )
        apply_prepared_eval_config(setup_args, eval_args)
        records = select_records_for_generation(
            Path(str(config["test_root"])) / category,
            full_test=True,
            subset_size=1,
            subset_seed=int(config["subset_seed"]),
            full_test_limit=config.get("full_test_limit"),
        )
        manifest = generate_records_to_dir(
            records,
            eval_args,
            model,
            diffusion,
            device,
            model_dtype,
            decoder,
            out_dir,
            phase="full_test",
            category=category,
            repeat_index=None,
        )
        full_results.append(manifest)

    for repeat_index in range(int(config["repeat_count"])):
        latent_seed = int(config["repeat_latent_seed_start"]) + repeat_index
        for category in TEST_CATEGORY_NAMES:
            out_dir = model_dir / "repeat_subset" / f"repeat_{repeat_index:02d}" / category
            eval_args = build_eval_args(
                config,
                model=model_spec,
                category=category,
                out_dir=out_dir,
                full_test=False,
                subset_size=int(config["repeat_subset_size"]),
                latent_seed=latent_seed,
            )
            apply_prepared_eval_config(setup_args, eval_args)
            records = select_records_for_generation(
                Path(str(config["test_root"])) / category,
                full_test=False,
                subset_size=int(config["repeat_subset_size"]),
                subset_seed=int(config["subset_seed"]),
                full_test_limit=None,
            )
            manifest = generate_records_to_dir(
                records,
                eval_args,
                model,
                diffusion,
                device,
                model_dtype,
                decoder,
                out_dir,
                phase="repeat_subset",
                category=category,
                repeat_index=repeat_index,
            )
            repeat_results.append(manifest)

    result = {
        "mode": "generate_only",
        "model": asdict(model_spec),
        "gpu": config["gpu"],
        "device": device,
        "precision": precision_name,
        "full_results": full_results,
        "repeat_results": repeat_results,
    }
    write_json(model_dir / "model_results.json", result)
    print(f"[{model_spec.name}] generate-only worker completed", flush=True)
    return 0


def run_shard_worker(config_path: Path) -> int:
    from validation.run_baseline_eval import run_evaluation

    config = read_json(config_path)
    model = ModelSpec(**config["model"])
    records = select_records_for_worker(config)
    eval_args = build_eval_args(
        config,
        model=model,
        category=str(config["category"]),
        out_dir=Path(str(config["out_dir"])),
        full_test=bool(config["full_test"]),
        subset_size=int(config["subset_size"]),
        latent_seed=int(config["latent_seed"]),
    )
    print(
        f"[{model.name}] {config['phase']} {config['category']} "
        f"shard {config['shard_index']}/{config['shard_count']} "
        f"records={len(records)} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}",
        flush=True,
    )
    run_dir = run_evaluation(eval_args, selected_records_override=records)
    result = {
        "model": model.name,
        "category": config["category"],
        "phase": config["phase"],
        "repeat_index": config.get("repeat_index"),
        "latent_seed": config["latent_seed"],
        "shard_index": config["shard_index"],
        "shard_count": config["shard_count"],
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "overall_summary.json"),
        "metrics_path": str(run_dir / "per_sample_metrics.csv"),
        "selected_cases_path": str(run_dir / "selected_cases.csv"),
        "all_sample_figures_path": str(run_dir / "all_sample_figures.csv"),
    }
    write_json(Path(str(config["result_path"])), result)
    return 0


def run_worker(config_path: Path) -> int:
    config = read_json(config_path)
    if bool(config.get("generate_only", False)):
        return run_generate_only_worker(config_path)

    from validation.run_baseline_eval import run_evaluation

    model = ModelSpec(**config["model"])
    model_dir = Path(str(config["model_dir"]))
    full_results: list[dict[str, object]] = []
    repeat_results: list[dict[str, object]] = []

    print(f"[{model.name}] worker started on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}", flush=True)
    for category in TEST_CATEGORY_NAMES:
        out_dir = model_dir / "full_test" / category
        eval_args = build_eval_args(
            config,
            model=model,
            category=category,
            out_dir=out_dir,
            full_test=True,
            subset_size=1,
            latent_seed=int(config["subset_seed"]),
        )
        selected_override = limited_records_if_needed(
            Path(str(config["test_root"])) / category,
            config.get("full_test_limit"),
            int(config["subset_seed"]),
        )
        run_dir = run_evaluation(eval_args, selected_records_override=selected_override)
        full_results.append(
            {
                "model": model.name,
                "category": category,
                "run_dir": str(run_dir),
                "summary_path": str(run_dir / "overall_summary.json"),
            }
        )
        print(f"[{model.name}] full_test {category} completed: {run_dir}", flush=True)

    for repeat_index in range(int(config["repeat_count"])):
        latent_seed = int(config["repeat_latent_seed_start"]) + repeat_index
        for category in TEST_CATEGORY_NAMES:
            out_dir = model_dir / "repeat_sampling" / f"repeat_{repeat_index:02d}" / category
            eval_args = build_eval_args(
                config,
                model=model,
                category=category,
                out_dir=out_dir,
                full_test=False,
                subset_size=int(config["repeat_subset_size"]),
                latent_seed=latent_seed,
            )
            run_dir = run_evaluation(eval_args)
            repeat_results.append(
                {
                    "model": model.name,
                    "category": category,
                    "repeat_index": repeat_index,
                    "latent_seed": latent_seed,
                    "run_dir": str(run_dir),
                    "summary_path": str(run_dir / "overall_summary.json"),
                }
            )
            print(f"[{model.name}] repeat {repeat_index:02d} {category} completed: {run_dir}", flush=True)

    result = {
        "model": asdict(model),
        "gpu": config["gpu"],
        "full_results": full_results,
        "repeat_results": repeat_results,
    }
    write_json(model_dir / "model_results.json", result)
    print(f"[{model.name}] worker completed", flush=True)
    return 0


def launch_workers(args: argparse.Namespace, models: list[ModelSpec], gpus: list[str]) -> list[Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_result_paths = [args.out_dir / "models" / model.name / "model_results.json" for model in models]
    processes: list[tuple[ModelSpec, subprocess.Popen, Path, str]] = []

    def start_worker(model: ModelSpec, gpu: str) -> tuple[ModelSpec, subprocess.Popen, Path, str]:
        model_dir = args.out_dir / "models" / model.name
        config_path = model_dir / "worker_config.json"
        log_path = model_dir / "worker.log"
        config = {
            "model": asdict(model),
            "gpu": gpu,
            "model_dir": str(model_dir),
            "test_root": str(args.test_root),
            "runtime_stats": str(args.runtime_stats),
            "batch_size": args.batch_size,
            "cfg_scale": args.cfg_scale,
            "num_sampling_steps": args.num_sampling_steps,
            "sampling_hz": args.sampling_hz,
            "length": args.length,
            "position_embedding": args.position_embedding,
            "rope_base": args.rope_base,
            "cond_embedding_scale": args.cond_embedding_scale,
            "subset_seed": args.subset_seed,
            "repeat_count": args.repeat_count,
            "repeat_subset_size": args.repeat_subset_size,
            "repeat_latent_seed_start": args.repeat_latent_seed_start,
            "full_test_limit": args.full_test_limit,
            "generate_only": args.generate_only,
            "save_pred_npy": args.save_pred_npy,
            "all_sample_figures": args.all_sample_figures,
        }
        write_json(config_path, config)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        cmd = [args.python, str(Path(__file__).resolve()), "--worker-config", str(config_path)]
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(f"launched model={model.name} gpu={gpu} pid={process.pid} log={log_path}", flush=True)
        return model, process, log_path, gpu

    pending = list(models)
    for gpu in gpus:
        if not pending:
            break
        processes.append(start_worker(pending.pop(0), gpu))

    failures: list[str] = []
    while processes:
        remaining: list[tuple[ModelSpec, subprocess.Popen, Path, str]] = []
        freed_gpus: list[str] = []
        for model, process, log_path, gpu in processes:
            return_code = process.poll()
            if return_code is None:
                remaining.append((model, process, log_path, gpu))
                continue
            if return_code != 0:
                failures.append(f"{model.name} exited with {return_code}; see {log_path}")
            else:
                print(f"completed model={model.name} gpu={gpu} log={log_path}", flush=True)
                freed_gpus.append(gpu)

        if failures:
            for _, process, _, _ in remaining:
                process.terminate()
            processes = remaining
            break

        for gpu in freed_gpus:
            if not pending:
                continue
            remaining.append(start_worker(pending.pop(0), gpu))

        processes = remaining
        if processes:
            time.sleep(10)

    if failures:
        raise RuntimeError("Worker failures:\n" + "\n".join(failures))
    return model_result_paths


def metric_mean(summary: dict[str, object], metric: str) -> float:
    return float(summary["metric_stats"][metric]["mean"])


def metric_score(metric: str, value: float) -> float:
    if metric in SIGNED_BIAS_METRICS:
        return abs(value)
    return value


def flatten_summary_row(model_name: str, category: str, summary: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model_name,
        "category": category,
        "count": int(summary["count"]),
        "run_dir": summary["run_dir"],
    }
    for metric in SUMMARY_METRIC_KEYS:
        stats = summary["metric_stats"][metric]
        row[f"{metric}_mean"] = float(stats["mean"])
        row[f"{metric}_median"] = float(stats["median"])
        row[f"{metric}_p90"] = float(stats["p90"])
    return row


def build_full_summary_rows(model_results: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in model_results:
        model_name = str(result["model"]["name"])
        for item in result["full_results"]:
            summary = read_json(Path(str(item["summary_path"])))
            rows.append(flatten_summary_row(model_name, str(item["category"]), summary))
    return rows


def build_delta_rows(full_rows: list[dict[str, object]], baseline_name: str) -> list[dict[str, object]]:
    by_key = {(str(row["model"]), str(row["category"])): row for row in full_rows}
    categories = sorted({str(row["category"]) for row in full_rows})
    models = [str(row["model"]) for row in full_rows if str(row["model"]) != baseline_name]
    models = sorted(set(models), key=models.index)
    rows: list[dict[str, object]] = []
    for model_name in models:
        for category in categories:
            baseline = by_key[(baseline_name, category)]
            candidate = by_key[(model_name, category)]
            for metric in SUMMARY_METRIC_KEYS:
                baseline_mean = float(baseline[f"{metric}_mean"])
                candidate_mean = float(candidate[f"{metric}_mean"])
                if metric == "pcc":
                    baseline_score = baseline_mean
                    candidate_score = candidate_mean
                    score_delta = candidate_score - baseline_score
                    better = "higher"
                elif metric in LOWER_IS_BETTER:
                    baseline_score = metric_score(metric, baseline_mean)
                    candidate_score = metric_score(metric, candidate_mean)
                    score_delta = candidate_score - baseline_score
                    better = "lower"
                else:
                    baseline_score = baseline_mean
                    candidate_score = candidate_mean
                    score_delta = candidate_score - baseline_score
                    better = "lower"
                denom = max(abs(baseline_score), 1e-12)
                rows.append(
                    {
                        "baseline_model": baseline_name,
                        "model": model_name,
                        "category": category,
                        "metric": metric,
                        "better": better,
                        "baseline_mean": baseline_mean,
                        "candidate_mean": candidate_mean,
                        "delta_mean": candidate_mean - baseline_mean,
                        "baseline_score": baseline_score,
                        "candidate_score": candidate_score,
                        "score_delta": score_delta,
                        "score_delta_pct": score_delta / denom,
                    }
                )
    return rows


def summarize_float_values(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance ** 0.5


def build_repeat_rows(model_results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for result in model_results:
        model_name = str(result["model"]["name"])
        for item in result["repeat_results"]:
            grouped.setdefault((model_name, str(item["category"])), []).append(item)

    rows: list[dict[str, object]] = []
    for (model_name, category), items in sorted(grouped.items()):
        summaries = [read_json(Path(str(item["summary_path"]))) for item in sorted(items, key=lambda x: int(x["repeat_index"]))]
        row: dict[str, object] = {
            "model": model_name,
            "category": category,
            "repeat_count": len(summaries),
            "count_per_repeat": int(summaries[0]["count"]) if summaries else 0,
        }
        for metric in SUMMARY_METRIC_KEYS:
            values = [metric_mean(summary, metric) for summary in summaries]
            mean, std = summarize_float_values(values)
            row[f"{metric}_repeat_mean"] = mean
            row[f"{metric}_repeat_std"] = std
        rows.append(row)
    return rows


def full_summary_fieldnames() -> list[str]:
    fields = ["model", "category", "count", "run_dir"]
    for metric in SUMMARY_METRIC_KEYS:
        fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_p90"])
    return fields


def repeat_summary_fieldnames() -> list[str]:
    fields = ["model", "category", "repeat_count", "count_per_repeat"]
    for metric in SUMMARY_METRIC_KEYS:
        fields.extend([f"{metric}_repeat_mean", f"{metric}_repeat_std"])
    return fields


def format_float(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def build_report(
    out_dir: Path,
    models: list[ModelSpec],
    gpus: list[str],
    args: argparse.Namespace,
    full_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    repeat_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = [
        "# SWaG Model Comparison",
        "",
        f"- Created: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Test root: `{args.test_root}`",
        f"- Runtime stats: `{args.runtime_stats}`",
        f"- Full test limit: `{args.full_test_limit if args.full_test_limit is not None else 'none'}`",
        f"- Repeat count: `{args.repeat_count}`",
        f"- Repeat subset size per category: `{args.repeat_subset_size}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Sampling steps: `{args.num_sampling_steps}`",
        f"- CFG scale: `{args.cfg_scale}`",
        "",
        "## Model Configuration",
        "",
    ]
    model_rows = []
    for index, model in enumerate(models):
        model_rows.append(
            [
                model.name,
                gpus[index],
                model.variant,
                model.representation,
                model.ckpt,
                model.autoencoder or "-",
            ]
        )
    lines.extend(markdown_table(["Model", "GPU", "Variant", "Representation", "Checkpoint", "Autoencoder"], model_rows))

    lines.extend(["", "## Full Test Key Metrics", ""])
    full_key_rows = []
    for row in full_rows:
        full_key_rows.append(
            [
                row["model"],
                row["category"],
                row["count"],
                format_float(row["pcc_mean"]),
                format_float(row["rmse_mean"]),
                format_float(row["mae_mean"]),
                format_float(row["log_fas_mse_0p1_20hz_mean"]),
            ]
        )
    lines.extend(markdown_table(["Model", "Category", "Count", "PCC", "RMSE", "MAE", "log-FAS MSE"], full_key_rows))

    lines.extend(["", "## Delta vs Baseline", ""])
    delta_key_rows = []
    for row in delta_rows:
        if row["metric"] not in KEY_REPORT_METRICS:
            continue
        delta_key_rows.append(
            [
                row["model"],
                row["category"],
                row["metric"],
                format_float(row["baseline_mean"]),
                format_float(row["candidate_mean"]),
                format_float(row["score_delta"]),
                f"{float(row['score_delta_pct']) * 100:+.2f}%",
            ]
        )
    lines.extend(
        markdown_table(
            ["Model", "Category", "Metric", "Baseline", "Candidate", "Score delta", "Score delta pct"],
            delta_key_rows,
        )
    )

    lines.extend(["", "## Repeat Sampling Stability", ""])
    repeat_key_rows = []
    for row in repeat_rows:
        repeat_key_rows.append(
            [
                row["model"],
                row["category"],
                row["repeat_count"],
                row["count_per_repeat"],
                f"{format_float(row['pcc_repeat_mean'])} +/- {format_float(row['pcc_repeat_std'])}",
                f"{format_float(row['rmse_repeat_mean'])} +/- {format_float(row['rmse_repeat_std'])}",
                f"{format_float(row['mae_repeat_mean'])} +/- {format_float(row['mae_repeat_std'])}",
                f"{format_float(row['log_fas_mse_0p1_20hz_repeat_mean'])} +/- {format_float(row['log_fas_mse_0p1_20hz_repeat_std'])}",
            ]
        )
    lines.extend(
        markdown_table(
            ["Model", "Category", "Repeats", "N", "PCC mean +/- std", "RMSE mean +/- std", "MAE mean +/- std", "log-FAS mean +/- std"],
            repeat_key_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Full-test rows are the primary ablation comparison.",
            "- Repeat-sampling rows measure diffusion sampling variability on the same stratified subsets with different latent seeds.",
            "- Repeat sampling does not replace multiple training seeds; it only quantifies generation-side stochasticity.",
            "- Prediction arrays and all-sample figures are disabled unless `--save-pred-npy` or `--all-sample-figures` is used.",
            "",
        ]
    )
    (out_dir / "model_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_results(args: argparse.Namespace, models: list[ModelSpec], gpus: list[str], result_paths: list[Path]) -> None:
    model_results = [read_json(path) for path in result_paths]
    full_rows = build_full_summary_rows(model_results)
    baseline_name = models[0].name
    delta_rows = build_delta_rows(full_rows, baseline_name=baseline_name)
    repeat_rows = build_repeat_rows(model_results)

    write_csv_rows(args.out_dir / "full_test_summary.csv", full_summary_fieldnames(), full_rows)
    write_csv_rows(
        args.out_dir / "full_test_delta_vs_baseline.csv",
        [
            "baseline_model",
            "model",
            "category",
            "metric",
            "better",
            "baseline_mean",
            "candidate_mean",
            "delta_mean",
            "baseline_score",
            "candidate_score",
            "score_delta",
            "score_delta_pct",
        ],
        delta_rows,
    )
    write_csv_rows(args.out_dir / "repeat_sampling_summary.csv", repeat_summary_fieldnames(), repeat_rows)
    build_report(args.out_dir, models, gpus, args, full_rows, delta_rows, repeat_rows)


def run_controller(args: argparse.Namespace) -> int:
    if not args.model:
        raise ValueError("At least one --model spec is required in controller mode.")
    if args.out_dir is None:
        raise ValueError("--out-dir is required in controller mode.")
    args.test_root = args.test_root.expanduser().resolve()
    args.runtime_stats = args.runtime_stats.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    models = [normalize_model_paths(parse_key_value_spec(raw)) for raw in args.model]
    gpus = parse_gpus(args.gpus)
    validate_inputs(args, models, gpus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_name = "generation_config.json" if args.generate_only else "comparison_config.json"
    write_json(
        args.out_dir / config_name,
        {
            "models": [asdict(model) for model in models],
            "gpus": gpus,
            "gpu_mode": args.gpu_mode,
            "test_root": str(args.test_root),
            "runtime_stats": str(args.runtime_stats),
            "batch_size": args.batch_size,
            "cfg_scale": args.cfg_scale,
            "num_sampling_steps": args.num_sampling_steps,
            "sampling_hz": args.sampling_hz,
            "length": args.length,
            "subset_seed": args.subset_seed,
            "repeat_count": args.repeat_count,
            "repeat_subset_size": args.repeat_subset_size,
            "repeat_latent_seed_start": args.repeat_latent_seed_start,
            "full_test_limit": args.full_test_limit,
            "generate_only": args.generate_only,
            "save_pred_npy": args.save_pred_npy,
            "all_sample_figures": args.all_sample_figures,
        },
    )
    result_paths = launch_workers(args, models, gpus)
    if args.generate_only:
        print(f"generation outputs saved to {args.out_dir}", flush=True)
    else:
        aggregate_results(args, models, gpus, result_paths)
        print(f"comparison report saved to {args.out_dir / 'model_comparison_report.md'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_config is not None:
        return run_worker(args.worker_config)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
