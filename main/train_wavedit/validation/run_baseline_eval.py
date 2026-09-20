from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TRAIN_WAVEDIT_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]

if str(TRAIN_WAVEDIT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_WAVEDIT_DIR))

from validation.metrics import SUMMARY_METRIC_KEYS, adjust_length, assert_finite_rows, build_binned_rows, compute_sample_metrics, summarize_metric_rows
from validation.report import plot_spectrum_psa_compare, plot_waveform_overlay, select_representative_cases, write_markdown_report
from validation.subset import DEPTH_LABELS, DISTANCE_LABELS, MAGNITUDE_LABELS, SELECTED_CASE_FIELDNAMES, CaseRecord, collect_case_records, select_stratified_subset, write_selected_cases_csv
from waveform_scaling import (
    apply_conditioning_config,
    apply_waveform_scale_config,
    conditioning_dimension,
    denormalize_waveform_array,
    encode_relative_geometry,
    project_condition_value,
    resolve_conditioning_config,
    resolve_waveform_scale_config,
    uses_epicentral_distance,
    uses_relative_geometry,
)
from waveform_vae import latent_scale_from_checkpoint, latent_scale_view, load_waveform_vae_checkpoint


DEFAULT_TEST_DIR = REPO_ROOT / "data" / "test"
DEFAULT_RESULT_ROOT = REPO_ROOT / "result" / "validation"
DEFAULT_HISTORY_PATH = DEFAULT_RESULT_ROOT / "RUN_HISTORY.md"
DEFAULT_SUBSET_SEED = 20260415
MAINLINE_VARIANT = "mainline"
AUTO_VARIANT = "auto"
MODEL_VARIANTS = (AUTO_VARIANT, MAINLINE_VARIANT)
CONDITION_MODES = ("normal", "shuffle", "zero")
REPRESENTATION_WAVEFORM = "waveform"
REPRESENTATION_KLVAE_LATENT = "klvae_latent"
MAINLINE_STATE_PREFIXES = ("conditioner.", "input_stem.")
MODEL_ARG_INT_KEYS = (
    "depth",
    "hidden_size",
    "num_heads",
    "hidden_feature",
    "length",
    "patch_size",
    "latent_channels",
    "latent_length",
    "latent_patch_size",
    "model_in_channels",
    "model_length",
    "model_patch_size",
)
MODEL_ARG_STR_KEYS = (
    "abs_geo_encoder_type",
    "position_embedding",
    "representation",
    "autoencoder_ckpt",
)
MODEL_ARG_FLOAT_KEYS = (
    "class_dropout_prob",
    "cond_embedding_scale",
    "rope_base",
    "epi_dist_min",
    "epi_dist_max",
)
PER_SAMPLE_FIELDNAMES = [
    *SELECTED_CASE_FIELDNAMES,
    "pcc",
    "rmse",
    "mae",
    "pga_real",
    "pga_pred",
    "pga_log_resid",
    "pgv_real",
    "pgv_pred",
    "pgv_log_resid",
    "psa_0p2_real",
    "psa_0p2_pred",
    "psa_0p2_log_resid",
    "psa_0p5_real",
    "psa_0p5_pred",
    "psa_0p5_log_resid",
    "psa_1p0_real",
    "psa_1p0_pred",
    "psa_1p0_log_resid",
    "psa_2p0_real",
    "psa_2p0_pred",
    "psa_2p0_log_resid",
    "log_fas_mse_0p1_20hz",
]
ALL_SAMPLE_FIGURE_FIELDNAMES = [
    "name",
    "waveform_relpath",
    "spectrum_relpath",
]


def convert_to_range(
    x: float,
    min_num: float,
    max_num: float,
    min_get: float,
    max_get: float,
    label_name: str = "condition",
) -> float:
    return project_condition_value(x, min_num, max_num, min_get, max_get, label_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWaG baseline evaluation and generate a Markdown report.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--model-variant", type=str, default=AUTO_VARIANT, choices=MODEL_VARIANTS)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--subset-size", type=int, default=256)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.2)
    parser.add_argument("--num-sampling-steps", "--num_sampling_steps", dest="num_sampling_steps", type=int, default=1000)
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--hidden_feature", type=int, default=168)
    parser.add_argument("--patch-size", type=int, default=6)
    parser.add_argument(
        "--abs-geo-encoder-type",
        dest="abs_geo_encoder_type",
        type=str,
        default="none",
        choices=["none", "epicentral_distance", "relative_geometry"],
    )
    parser.add_argument(
        "--position-embedding",
        dest="position_embedding",
        type=str,
        default="sincos",
        choices=["sincos", "rope"],
    )
    parser.add_argument("--rope-base", dest="rope_base", type=float, default=10000.0)
    parser.add_argument("--subset-seed", type=int, default=DEFAULT_SUBSET_SEED)
    parser.add_argument(
        "--latent-seed",
        type=int,
        default=None,
        help=(
            "Seed used for deterministic diffusion initial latents. "
            "Defaults to --subset-seed for backwards-compatible behavior."
        ),
    )
    parser.add_argument("--result-tag", type=str, default="baseline")
    parser.add_argument("--condition-mode", type=str, default="normal", choices=CONDITION_MODES)
    parser.add_argument("--save-pred-npy", action="store_true")
    parser.add_argument("--all-sample-figures", action="store_true")
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
    parser.add_argument("--min-get", default=0, type=float)
    parser.add_argument("--max-get", default=1000, type=float)
    parser.add_argument("--cond-embedding-scale", default=1.0, type=float)
    parser.add_argument("--waveform-scale-stats", type=str, default=None)
    parser.add_argument("--waveform-mean-value", type=float, default=None)
    parser.add_argument("--waveform-std-value", type=float, default=None)
    parser.add_argument("--representation", type=str, default=REPRESENTATION_WAVEFORM, choices=(REPRESENTATION_WAVEFORM, REPRESENTATION_KLVAE_LATENT))
    parser.add_argument("--autoencoder-ckpt", dest="autoencoder_ckpt", type=str, default=None)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--latent-length", type=int, default=375)
    parser.add_argument("--latent-patch-size", type=int, default=1)
    return parser


def ensure_environment(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("timm") is None:
        raise RuntimeError(
            "Missing required dependency 'timm'. Activate the swag environment first, "
            "for example: `conda activate swag` or `conda run -n swag python ...`."
        )
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
    if args.num_sampling_steps < 2:
        raise ValueError(
            f"--num-sampling-steps must be >= 2 for the current diffusion implementation, got {args.num_sampling_steps}."
        )
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")


def load_checkpoint_payload(ckpt_path: Path) -> dict[str, object]:
    payload = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
    if not isinstance(payload, dict) or "ema" not in payload:
        raise RuntimeError(f"Checkpoint format error, missing 'ema' weights: {ckpt_path}")
    return payload


def _ckpt_value(ckpt_payload: dict[str, object], ckpt_args, key: str):
    if key in ckpt_payload and ckpt_payload[key] is not None:
        return ckpt_payload[key]
    if ckpt_args is not None and hasattr(ckpt_args, key):
        return getattr(ckpt_args, key)
    return None


def apply_checkpoint_model_args(args: argparse.Namespace, ckpt_payload: dict[str, object]) -> None:
    ckpt_args = ckpt_payload.get("args")
    for key in MODEL_ARG_INT_KEYS:
        value = _ckpt_value(ckpt_payload, ckpt_args, key)
        if value is not None:
            setattr(args, key, int(value))
    for key in MODEL_ARG_STR_KEYS:
        if key == "autoencoder_ckpt" and getattr(args, key, None):
            continue
        value = _ckpt_value(ckpt_payload, ckpt_args, key)
        if value is None:
            continue
        setattr(args, key, str(value))
    for key in MODEL_ARG_FLOAT_KEYS:
        value = _ckpt_value(ckpt_payload, ckpt_args, key)
        if value is not None:
            setattr(args, key, float(value))


def configure_representation(args: argparse.Namespace) -> None:
    representation = getattr(args, "representation", REPRESENTATION_WAVEFORM)
    if representation == REPRESENTATION_WAVEFORM:
        args.model_in_channels = 3
        args.model_length = args.length
        args.model_patch_size = args.patch_size
        return
    if representation != REPRESENTATION_KLVAE_LATENT:
        raise ValueError(f"Unsupported representation: {representation}")
    if args.model_variant != MAINLINE_VARIANT:
        raise ValueError("--representation klvae_latent is only supported for mainline checkpoints")
    if not getattr(args, "autoencoder_ckpt", None):
        raise ValueError("Latent checkpoint requires an autoencoder path in checkpoint metadata or --autoencoder-ckpt")
    args.model_in_channels = args.latent_channels
    args.model_length = args.latent_length
    args.model_patch_size = args.latent_patch_size


def build_sample_decoder(args: argparse.Namespace, ckpt_payload: dict[str, object], device: str):
    if getattr(args, "representation", REPRESENTATION_WAVEFORM) != REPRESENTATION_KLVAE_LATENT:
        return None
    vae, vae_payload = load_waveform_vae_checkpoint(args.autoencoder_ckpt, device=torch.device(device), dtype=torch.float32)
    if vae.latent_channels != args.latent_channels or vae.latent_length != args.latent_length:
        raise ValueError(
            f"VAE latent shape ({vae.latent_channels},{vae.latent_length}) does not match "
            f"checkpoint latent shape ({args.latent_channels},{args.latent_length})"
        )
    latent_scale = ckpt_payload.get("latent_scale")
    if latent_scale is None:
        latent_scale = latent_scale_from_checkpoint(vae_payload, args.latent_channels)
    else:
        latent_scale = torch.tensor(latent_scale, dtype=torch.float32).flatten()
    return {
        "model": vae,
        "scale": latent_scale_view(latent_scale, torch.device(device), torch.float32),
    }


@torch.no_grad()
def decode_samples_if_needed(samples: torch.Tensor, decoder) -> torch.Tensor:
    if decoder is None:
        return samples
    scale = decoder["scale"].to(device=samples.device, dtype=torch.float32)
    latent = samples.float() * scale
    return decoder["model"].decode_from_latent(latent)


def detect_model_variant_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str | None:
    keys = tuple(state_dict.keys())
    if any(key.startswith(prefix) for key in keys for prefix in MAINLINE_STATE_PREFIXES):
        return MAINLINE_VARIANT
    return None


def resolve_model_variant(requested_variant: str, state_dict: dict[str, torch.Tensor]) -> str:
    detected_variant = detect_model_variant_from_state_dict(state_dict)
    if requested_variant == AUTO_VARIANT:
        if detected_variant is None:
            raise RuntimeError(
                "Unable to infer model variant from checkpoint state_dict. "
                f"Use --model-variant with one of: {', '.join(MODEL_VARIANTS[1:])}."
            )
        return detected_variant
    if detected_variant is not None and requested_variant != detected_variant:
        raise RuntimeError(
            f"Checkpoint appears to be '{detected_variant}', but --model-variant was set to "
            f"'{requested_variant}'."
        )
    return requested_variant


def infer_device_and_precision() -> tuple[str, torch.dtype, str, list[str]]:
    warnings: list[str] = []
    if torch.cuda.is_available():
        return "cuda", torch.float16, "fp16", warnings

    warnings.append("当前运行使用 CPU fallback，速度口径不作为正式结果。")
    return "cpu", torch.float32, "fp32", warnings


def build_condition_tensor(
    records: list[CaseRecord],
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    condition_dim = conditioning_dimension(getattr(args, "abs_geo_encoder_type", "none"))
    if getattr(args, "condition_mode", "normal") == "zero":
        return torch.zeros((len(records), condition_dim), dtype=dtype, device=device)
    rows = []
    for record in records:
        row = [
            convert_to_range(record.sta_lat, args.sta_lat_min, args.sta_lat_max, args.min_get, args.max_get, "sta_lat"),
            convert_to_range(record.sta_lon, args.sta_lon_min, args.sta_lon_max, args.min_get, args.max_get, "sta_lon"),
            convert_to_range(record.src_lat, args.src_lat_min, args.src_lat_max, args.min_get, args.max_get, "src_lat"),
            convert_to_range(record.src_lon, args.src_lon_min, args.src_lon_max, args.min_get, args.max_get, "src_lon"),
            convert_to_range(record.depth_km, args.dep_min, args.dep_max, args.min_get, args.max_get, "dep"),
            convert_to_range(record.magnitude, args.mag_min, args.mag_max, args.min_get, args.max_get, "mag"),
        ]
        if uses_epicentral_distance(getattr(args, "abs_geo_encoder_type", "none")):
            row.append(
                convert_to_range(
                    record.epicentral_distance_km,
                    args.epi_dist_min,
                    args.epi_dist_max,
                    args.min_get,
                    args.max_get,
                    "epi_dist",
                )
            )
        elif uses_relative_geometry(getattr(args, "abs_geo_encoder_type", "none")):
            row.extend(
                encode_relative_geometry(
                    sta_lat=record.sta_lat,
                    sta_lon=record.sta_lon,
                    src_lat=record.src_lat,
                    src_lon=record.src_lon,
                    distance_max_km=args.epi_dist_max,
                    output_min=args.min_get,
                    output_max=args.max_get,
                )
            )
        rows.append(row)
    return torch.tensor(rows, dtype=dtype, device=device)


def build_condition_records_for_mode(
    records: list[CaseRecord],
    args: argparse.Namespace,
) -> list[CaseRecord]:
    mode = getattr(args, "condition_mode", "normal")
    if mode in {"normal", "zero"}:
        return list(records)
    if mode != "shuffle":
        raise ValueError(f"Unsupported condition_mode: {mode}")
    if len(records) <= 1:
        return list(records)
    payload = f"{args.subset_seed}:{args.test_dir}:condition-shuffle".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
    offset = offset % (len(records) - 1) + 1
    return list(records[offset:]) + list(records[:offset])


def load_model_and_diffusion(
    args: argparse.Namespace,
    state_dict: dict[str, torch.Tensor],
    device: str,
    model_dtype: torch.dtype,
):
    from diffusion import create_diffusion

    if args.model_variant == MAINLINE_VARIANT:
        from models_mainline import build_waveform_model

        model = build_waveform_model(args)
    else:
        raise ValueError(f"Unsupported model variant: {args.model_variant}")
    model.load_state_dict(state_dict)
    model.eval()
    if model_dtype == torch.float16:
        model.half()
    else:
        model.float()
    model.to(device)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    return model, diffusion


def batched(items: list[CaseRecord], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_deterministic_latents(
    records: list[CaseRecord],
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    latents = []
    latent_seed = getattr(args, "latent_seed", None)
    if latent_seed is None:
        latent_seed = args.subset_seed
    for record in records:
        payload = f"{latent_seed}:{record.name}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        latent = rng.standard_normal((args.model_in_channels, args.model_length)).astype(np.float32)
        latents.append(latent)
    latent_batch = np.stack(latents, axis=0)
    return torch.from_numpy(latent_batch).to(device=device, dtype=dtype)


@torch.no_grad()
def generate_predictions_for_records(
    records: list[CaseRecord],
    args: argparse.Namespace,
    model,
    diffusion,
    device: str,
    model_dtype: torch.dtype,
    condition_records: list[CaseRecord] | None = None,
    decoder=None,
) -> list[np.ndarray]:
    if not records:
        return []
    if condition_records is None:
        condition_records = records
    if len(condition_records) != len(records):
        raise ValueError(
            f"condition_records length ({len(condition_records)}) must match records length ({len(records)})."
        )

    y = build_condition_tensor(condition_records, args, device=device, dtype=model_dtype)
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device == "cuda" and model_dtype == torch.float16
        else nullcontext()
    )
    with amp_context:
        z = build_deterministic_latents(records, args, device=device, dtype=model_dtype)
        z = torch.cat([z, z], 0)
        y_ = torch.cat([y, y], 0)
        model_kwargs = dict(y=y_, cfg_scale=args.cfg_scale)
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
    samples = decode_samples_if_needed(samples[: len(records)].detach(), decoder)
    pred = samples.float().cpu().numpy().transpose(0, 2, 1)
    return [
        denormalize_waveform_array(
            pred[index],
            mean_value=args.waveform_mean_value,
            std_value=args.waveform_std_value,
        )
        for index in range(pred.shape[0])
    ]


def load_real_waveform(path: Path, target_length: int) -> np.ndarray:
    return adjust_length(np.load(path).astype(np.float32), target_length)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_case_title(record: CaseRecord) -> str:
    return (
        f"{record.name} | M {record.magnitude:.2f} | Depth {record.depth_km:.2f} km | "
        f"Dist {record.epicentral_distance_km:.2f} km"
    )


def metric_mean(summary: dict[str, object], key: str) -> float:
    return float(summary["metric_stats"][key]["mean"])


def format_bias_text(value: float, neutral: float = 0.05) -> str:
    if value > neutral:
        return "整体偏高"
    if value < -neutral:
        return "整体偏低"
    return "整体接近中性"


def format_pcc_text(value: float) -> str:
    if abs(value) < 0.01:
        return "几乎接近于零"
    if abs(value) < 0.05:
        return "仍然偏低"
    return "已有一定正相关"


def format_elapsed_compact(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds:.2f} s ({seconds / 3600.0:.2f} h)"
    return f"{seconds:.2f} s"


def default_history_intro() -> str:
    return (
        "# SWaG Validation Run History\n\n"
        "本文件持续记录 `result/validation/` 下每一次正式验证运行的参数、环境、核心指标、结果解读与后续判断。"
    )


def split_history_sections(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    intro_lines: list[str] = []
    sections: list[str] = []
    current_section: list[str] | None = None
    for line in lines:
        if line.startswith("## "):
            if current_section is not None:
                sections.append("\n".join(current_section).rstrip())
            current_section = [line]
            continue
        if current_section is None:
            intro_lines.append(line)
        else:
            current_section.append(line)
    if current_section is not None:
        sections.append("\n".join(current_section).rstrip())
    intro = "\n".join(intro_lines).rstrip()
    return intro, sections


def find_previous_summary(current_run_dir: Path) -> dict[str, object] | None:
    candidates: list[tuple[float, dict[str, object]]] = []
    for path in DEFAULT_RESULT_ROOT.iterdir():
        if not path.is_dir() or path == current_run_dir:
            continue
        summary_path = path / "overall_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates.append((path.stat().st_mtime, summary))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def build_compare_lines(summary: dict[str, object], previous_summary: dict[str, object] | None) -> list[str]:
    if previous_summary is None:
        return []

    lines = ["**与上一轮对照**", ""]
    compare_keys = [
        ("PCC", "pcc"),
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("PGA log residual", "pga_log_resid"),
        ("PGV log residual", "pgv_log_resid"),
        ("PSA@0.2s", "psa_0p2_log_resid"),
        ("PSA@0.5s", "psa_0p5_log_resid"),
        ("PSA@1.0s", "psa_1p0_log_resid"),
        ("PSA@2.0s", "psa_2p0_log_resid"),
    ]
    for label, key in compare_keys:
        prev_value = metric_mean(previous_summary, key)
        curr_value = metric_mean(summary, key)
        delta = curr_value - prev_value
        lines.append(
            f"- `{label}`: `{prev_value:.6f} -> {curr_value:.6f}` (`delta = {delta:+.6f}`)"
        )
    lines.append("")
    return lines


def build_history_entry(summary: dict[str, object], previous_summary: dict[str, object] | None = None) -> str:
    run_dir = Path(summary["run_dir"])
    run_id = run_dir.name
    pcc_mean = metric_mean(summary, "pcc")
    rmse_mean = metric_mean(summary, "rmse")
    mae_mean = metric_mean(summary, "mae")
    pga_mean = metric_mean(summary, "pga_log_resid")
    pgv_mean = metric_mean(summary, "pgv_log_resid")
    psa02_mean = metric_mean(summary, "psa_0p2_log_resid")
    psa05_mean = metric_mean(summary, "psa_0p5_log_resid")
    psa10_mean = metric_mean(summary, "psa_1p0_log_resid")
    psa20_mean = metric_mean(summary, "psa_2p0_log_resid")
    fas_mean = metric_mean(summary, "log_fas_mse_0p1_20hz")
    warnings = list(summary.get("warnings", []))
    cpu_fallback = any("CPU fallback" in item for item in warnings)
    run_date = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d")

    analysis_text = (
        f"本轮运行表明，模型在 `{summary['count']}` 个样本上仍能生成量级可比的三分量波形，"
        f"但 `PCC mean = {pcc_mean:.6f}` {format_pcc_text(pcc_mean)}。"
        f"`RMSE mean = {rmse_mean:.6f}`、`MAE mean = {mae_mean:.6f}` 说明幅值尺度并未完全失真，"
        f"但波形逐点一致性仍不足。工程指标方面，`PGA` {format_bias_text(pga_mean)}，"
        f"`PGV` {format_bias_text(pgv_mean)}；`PSA@0.2s/0.5s/1.0s/2.0s` 分别为 "
        f"`{psa02_mean:.6f} / {psa05_mean:.6f} / {psa10_mean:.6f} / {psa20_mean:.6f}`，"
        f"说明不同周期段的响应偏差并不一致。"
    )

    diagnosis_text = (
        f"从诊断角度看，当前模型主要问题仍是时序结构与相位关系没有学稳，而不是单纯的幅值完全错误。"
        f"`PCC` 很低但 `RMSE` 仍处于可比较范围，说明模型更像学到了某种统计包络或能量尺度。"
        f"同时，`PGA` 与不同周期 `PSA` 的偏差方向不完全一致，提示后续应重点检查高频尖峰、"
        f"中长周期包络以及条件注入后的频段平衡能力。"
    )

    lines = [
        f"## {run_id}",
        "",
        f"- 运行时间: `{run_date}`",
        "- 状态: `completed`",
        f"- 运行目录: [{run_id}/report.md]({(run_dir / 'report.md').as_posix()})",
        f"- 模型: `{summary['ckpt_path']}`",
        f"- 测试集: `{summary['test_dir']}`",
        f"- 样本数: `{summary['count']}`",
        f"- 采样步数: `{summary['num_sampling_steps']}`",
        f"- CFG: `{summary['cfg_scale']}`",
        f"- 采样率: `{summary['sampling_hz']} Hz`",
        f"- Batch size: `{summary['batch_size']}`",
        f"- Subset seed: `{summary['subset_seed']}`",
        f"- 设备: `{summary['device']}`",
        f"- 精度: `{summary['precision']}`",
        f"- CPU fallback: `{'yes' if cpu_fallback else 'no'}`",
        f"- 总耗时: `{format_elapsed_compact(float(summary['elapsed_seconds']))}`",
        "",
        "**核心指标**",
        "",
        f"- `PCC mean = {pcc_mean:.6f}`",
        f"- `RMSE mean = {rmse_mean:.6f}`",
        f"- `MAE mean = {mae_mean:.6f}`",
        f"- `PGA log residual mean = {pga_mean:.6f}`",
        f"- `PGV log residual mean = {pgv_mean:.6f}`",
        f"- `PSA@0.2/0.5/1.0/2.0s mean = {psa02_mean:.6f} / {psa05_mean:.6f} / {psa10_mean:.6f} / {psa20_mean:.6f}`",
        f"- `log-FAS MSE mean = {fas_mean:.6f}`",
        "",
        "**论文式结果分析**",
        "",
        analysis_text,
        "",
        "**模型诊断式解读**",
        "",
        diagnosis_text,
        "",
    ]
    lines.extend(build_compare_lines(summary, previous_summary))
    lines.extend(
        [
            "**限制说明**",
            "",
            f"- 当前为 `{'full test' if summary['full_test'] else 'subset'}` 结果，不能直接替代最终全测试集结论。",
            f"- 当前运行环境为 `{'CPU fallback' if cpu_fallback else summary['device']}`，速度口径{'非正式' if cpu_fallback else '正式'}。",
            "- 稀疏分桶和极端条件样本仍可能较少，分组结果应优先视作趋势观察。",
        ]
    )
    return "\n".join(lines).rstrip()


def update_run_history(summary: dict[str, object]) -> None:
    history_path = DEFAULT_HISTORY_PATH
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        text = history_path.read_text(encoding="utf-8")
        intro, sections = split_history_sections(text)
        if not intro:
            intro = default_history_intro()
    else:
        intro = default_history_intro()
        sections = []

    run_id = Path(summary["run_dir"]).name
    previous_summary = find_previous_summary(Path(summary["run_dir"]))
    new_entry = build_history_entry(summary, previous_summary=previous_summary)
    filtered_sections = [section for section in sections if not section.startswith(f"## {run_id}\n") and section != f"## {run_id}"]
    content_parts = [intro.rstrip(), new_entry]
    content_parts.extend(filtered_sections)
    history_path.write_text("\n\n".join(part for part in content_parts if part), encoding="utf-8")


def build_run_dir(args: argparse.Namespace, evaluated_count: int) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "full" if args.full_test else f"sub{evaluated_count}"
    return DEFAULT_RESULT_ROOT / f"{args.result_tag}_{args.ckpt.stem}_{mode}_{timestamp}"


def evaluate_records(
    records: list[CaseRecord],
    args: argparse.Namespace,
    model,
    diffusion,
    device: str,
    model_dtype: torch.dtype,
    pred_dir: Path | None,
    all_sample_figures_dir: Path | None = None,
    report_dir: Path | None = None,
    decoder=None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []
    processed = 0
    total = len(records)
    if all_sample_figures_dir is not None:
        all_sample_figures_dir.mkdir(parents=True, exist_ok=True)
    if all_sample_figures_dir is not None and report_dir is None:
        raise ValueError("report_dir is required when all_sample_figures_dir is enabled")
    condition_records = build_condition_records_for_mode(records, args)
    batch_size = max(1, args.batch_size)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        condition_batch = condition_records[start : start + batch_size]
        pred_batch = generate_predictions_for_records(
            batch,
            args=args,
            model=model,
            diffusion=diffusion,
            device=device,
            model_dtype=model_dtype,
            condition_records=condition_batch,
            decoder=decoder,
        )
        for record, pred in zip(batch, pred_batch):
            real = load_real_waveform(record.path, args.length)
            metrics = compute_sample_metrics(real, pred, args.sampling_hz)
            row = record.to_row()
            row.update(metrics)
            rows.append(row)
            if pred_dir is not None:
                np.save(pred_dir / f"{record.name}_pred.npy", pred)
            if all_sample_figures_dir is not None and report_dir is not None:
                waveform_path = all_sample_figures_dir / f"{record.name}_waveform.png"
                spectrum_path = all_sample_figures_dir / f"{record.name}_spectrum.png"
                title = build_case_title(record)
                plot_waveform_overlay(real, pred, waveform_path, args.sampling_hz, title)
                plot_spectrum_psa_compare(real, pred, spectrum_path, args.sampling_hz, title)
                figure_rows.append(
                    {
                        "name": record.name,
                        "waveform_relpath": waveform_path.relative_to(report_dir).as_posix(),
                        "spectrum_relpath": spectrum_path.relative_to(report_dir).as_posix(),
                    }
                )
            processed += 1
        if processed % 20 == 0 or processed == total:
            print(f"processed {processed}/{total}")
    return rows, figure_rows


def create_representative_figures(
    representative_rows: list[dict[str, object]],
    record_by_name: dict[str, CaseRecord],
    condition_record_by_name: dict[str, CaseRecord],
    args: argparse.Namespace,
    model,
    diffusion,
    device: str,
    model_dtype: torch.dtype,
    figures_dir: Path,
    report_dir: Path,
    decoder=None,
) -> list[dict[str, object]]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    for item in representative_rows:
        row = item["row"]
        record = record_by_name[str(row["name"])]
        condition_record = condition_record_by_name[str(row["name"])]
        pred = generate_predictions_for_records(
            [record],
            args=args,
            model=model,
            diffusion=diffusion,
            device=device,
            model_dtype=model_dtype,
            condition_records=[condition_record],
            decoder=decoder,
        )[0]
        real = load_real_waveform(record.path, args.length)

        waveform_path = figures_dir / f"{item['label']}_{record.name}_waveform.png"
        spectrum_path = figures_dir / f"{item['label']}_{record.name}_spectrum.png"
        title = build_case_title(record)
        plot_waveform_overlay(real, pred, waveform_path, args.sampling_hz, title)
        plot_spectrum_psa_compare(real, pred, spectrum_path, args.sampling_hz, title)
        outputs.append(
            {
                "label": item["label"],
                "row": row,
                "waveform_relpath": waveform_path.relative_to(report_dir).as_posix(),
                "spectrum_relpath": spectrum_path.relative_to(report_dir).as_posix(),
            }
        )
    return outputs


def run_evaluation(
    args: argparse.Namespace,
    selected_records_override: list[CaseRecord] | None = None,
) -> Path:
    ensure_environment(args)
    if getattr(args, "latent_seed", None) is None:
        args.latent_seed = args.subset_seed

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
        raise RuntimeError("Baseline evaluation currently supports epsilon-prediction checkpoints only.")
    state_dict = ckpt_payload["ema"]
    args.model_variant = resolve_model_variant(args.model_variant, state_dict)
    configure_representation(args)
    conditioning_config = resolve_conditioning_config(
        args,
        checkpoint_args=ckpt_payload.get("args"),
        prefer_checkpoint=True,
        require_ranges=True,
    )
    apply_conditioning_config(args, conditioning_config)
    scale_config = resolve_waveform_scale_config(
        args,
        checkpoint_args=ckpt_payload.get("args"),
        prefer_checkpoint=True,
        require_scale=True,
    )
    apply_waveform_scale_config(args, scale_config)
    device, model_dtype, precision_name, warnings = infer_device_and_precision()
    decoder = build_sample_decoder(args, ckpt_payload, device)
    for warning in warnings:
        print(f"warning: {warning}")

    if selected_records_override is None:
        records = collect_case_records(args.test_dir)
        if not records:
            raise RuntimeError(f"No .npy test files found under {args.test_dir}")

        if args.full_test:
            selected_records = records
        else:
            selected_records = select_stratified_subset(records, args.subset_size, args.subset_seed)
    else:
        selected_records = list(selected_records_override)
        if not selected_records:
            raise RuntimeError("selected_records_override was provided but is empty")

    run_dir = build_run_dir(args, len(selected_records))
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    all_sample_figures_dir = figures_dir / "all_samples" if args.all_sample_figures else None
    pred_dir = run_dir / "pred_npy" if args.save_pred_npy else None
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    selected_cases_path = run_dir / "selected_cases.csv"
    write_selected_cases_csv(selected_records, selected_cases_path)

    model, diffusion = load_model_and_diffusion(args, state_dict, device=device, model_dtype=model_dtype)

    # DDPM sampling injects transition noise at every reverse step in addition
    # to the record-specific initial latent. Bind both noise paths to the run
    # seed so that an archived configuration can be reproduced exactly.
    reverse_noise_seed = int(args.latent_seed)
    torch.manual_seed(reverse_noise_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(reverse_noise_seed)

    time0 = time.time()
    rows, all_sample_figure_rows = evaluate_records(
        selected_records,
        args=args,
        model=model,
        diffusion=diffusion,
        device=device,
        model_dtype=model_dtype,
        pred_dir=pred_dir,
        all_sample_figures_dir=all_sample_figures_dir,
        report_dir=run_dir,
        decoder=decoder,
    )
    elapsed_seconds = time.time() - time0
    assert_finite_rows(rows)

    per_sample_metrics_path = run_dir / "per_sample_metrics.csv"
    write_csv_rows(per_sample_metrics_path, PER_SAMPLE_FIELDNAMES, rows)
    all_sample_figures_path = run_dir / "all_sample_figures.csv"
    if args.all_sample_figures:
        write_csv_rows(all_sample_figures_path, ALL_SAMPLE_FIGURE_FIELDNAMES, all_sample_figure_rows)

    metric_stats = summarize_metric_rows(rows)
    magnitude_rows = build_binned_rows(rows, "magnitude_bucket", MAGNITUDE_LABELS)
    depth_rows = build_binned_rows(rows, "depth_bucket", DEPTH_LABELS)
    distance_rows = build_binned_rows(rows, "distance_bucket", DISTANCE_LABELS)

    binned_fieldnames = ["bucket", "count"]
    for metric_key in SUMMARY_METRIC_KEYS:
        binned_fieldnames.extend(
            [f"{metric_key}_mean", f"{metric_key}_median", f"{metric_key}_p90"]
        )

    write_csv_rows(run_dir / "binned_metrics_magnitude.csv", binned_fieldnames, magnitude_rows)
    write_csv_rows(run_dir / "binned_metrics_depth.csv", binned_fieldnames, depth_rows)
    write_csv_rows(run_dir / "binned_metrics_distance.csv", binned_fieldnames, distance_rows)

    summary = {
        "model_name": args.ckpt.stem,
        "ckpt_path": str(args.ckpt),
        "test_dir": str(args.test_dir),
        "run_dir": str(run_dir),
        "count": len(rows),
        "full_test": bool(args.full_test),
        "subset_seed": args.subset_seed,
        "latent_seed": args.latent_seed,
        "reverse_noise_seed": reverse_noise_seed,
        "sampling_rng_control": "initial_latent_and_ddpm_reverse_noise",
        "condition_mode": getattr(args, "condition_mode", "normal"),
        "device": device,
        "precision": precision_name,
        "cfg_scale": args.cfg_scale,
        "num_sampling_steps": args.num_sampling_steps,
        "sampling_hz": args.sampling_hz,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed_seconds,
        "warnings": warnings,
        "waveform_norm_mode": args.waveform_norm_mode,
        "waveform_mean_value": args.waveform_mean_value,
        "waveform_std_value": args.waveform_std_value,
        "waveform_scale_source": args.waveform_scale_source,
        "waveform_scale_stats": args.waveform_scale_stats,
        "conditioning_source": getattr(args, "conditioning_source", None),
        "conditioning_stats_path": getattr(args, "conditioning_stats_path", None),
        "variant": args.model_variant,
        "representation": getattr(args, "representation", REPRESENTATION_WAVEFORM),
        "autoencoder_ckpt": getattr(args, "autoencoder_ckpt", None),
        "latent_channels": getattr(args, "latent_channels", None),
        "latent_length": getattr(args, "latent_length", None),
        "latent_patch_size": getattr(args, "latent_patch_size", None),
        "model_in_channels": getattr(args, "model_in_channels", None),
        "model_length": getattr(args, "model_length", None),
        "model_patch_size": getattr(args, "model_patch_size", None),
        "abs_geo_encoder_type": getattr(args, "abs_geo_encoder_type", None),
        "conditioning_dimension": conditioning_dimension(getattr(args, "abs_geo_encoder_type", "none")),
        "epi_dist_min": getattr(args, "epi_dist_min", None),
        "epi_dist_max": getattr(args, "epi_dist_max", None),
        "position_embedding": getattr(args, "position_embedding", "sincos"),
        "rope_base": getattr(args, "rope_base", 10000.0),
        "metric_stats": metric_stats,
        "all_sample_figures": bool(args.all_sample_figures),
        "all_sample_figure_count": len(all_sample_figure_rows),
    }

    overall_summary_path = run_dir / "overall_summary.json"
    overall_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    representative_rows = select_representative_cases(rows)
    record_by_name = {record.name: record for record in selected_records}
    condition_records = build_condition_records_for_mode(selected_records, args)
    condition_record_by_name = {
        record.name: condition_record
        for record, condition_record in zip(selected_records, condition_records)
    }
    representative_outputs = create_representative_figures(
        representative_rows,
        record_by_name=record_by_name,
        condition_record_by_name=condition_record_by_name,
        args=args,
        model=model,
        diffusion=diffusion,
        device=device,
        model_dtype=model_dtype,
        figures_dir=figures_dir,
        report_dir=run_dir,
        decoder=decoder,
    )
    write_markdown_report(
        run_dir / "report.md",
        summary=summary,
        magnitude_rows=magnitude_rows,
        depth_rows=depth_rows,
        distance_rows=distance_rows,
        representative_cases=representative_outputs,
    )
    if getattr(args, "update_history", True):
        update_run_history(summary)

    print(f"selected cases saved to {selected_cases_path}")
    print(f"per-sample metrics saved to {per_sample_metrics_path}")
    if args.all_sample_figures:
        print(f"all-sample figure inventory saved to {all_sample_figures_path}")
    print(f"overall summary saved to {overall_summary_path}")
    print(f"report saved to {run_dir / 'report.md'}")
    return run_dir

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
