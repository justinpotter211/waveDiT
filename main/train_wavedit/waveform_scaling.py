import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TRAIN_WAVEDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_WAVEDIT_DIR.parents[1]
DEFAULT_RUNTIME_STATS_PATH = REPO_ROOT / "data" / "dataset_runtime_stats.json"
DEFAULT_SCALE_STATS_PATH = DEFAULT_RUNTIME_STATS_PATH
WAVEFORM_NORM_GLOBAL_ZSCORE = "global_zscore"
ABS_GEO_ENCODER_NONE = "none"
ABS_GEO_ENCODER_EPICENTRAL_DISTANCE = "epicentral_distance"
ABS_GEO_ENCODER_RELATIVE_GEOMETRY = "relative_geometry"
BASE_CONDITIONING_ARG_NAMES = (
    "sta_lat_min",
    "sta_lat_max",
    "sta_lon_min",
    "sta_lon_max",
    "src_lat_min",
    "src_lat_max",
    "src_lon_min",
    "src_lon_max",
    "dep_min",
    "dep_max",
    "mag_min",
    "mag_max",
)
EPICENTRAL_DISTANCE_ARG_NAMES = ("epi_dist_min", "epi_dist_max")
BASE_CONDITIONING_BOUND_PAIRS = (
    ("sta_lat_min", "sta_lat_max"),
    ("sta_lon_min", "sta_lon_max"),
    ("src_lat_min", "src_lat_max"),
    ("src_lon_min", "src_lon_max"),
    ("dep_min", "dep_max"),
    ("mag_min", "mag_max"),
)
EPICENTRAL_DISTANCE_BOUND_PAIRS = (("epi_dist_min", "epi_dist_max"),)
LABEL_EPS = 1e-6


def normalize_abs_geo_encoder_type(abs_geo_encoder_type: str | None) -> str:
    mode = ABS_GEO_ENCODER_NONE if abs_geo_encoder_type is None else str(abs_geo_encoder_type).lower()
    if mode not in {
        ABS_GEO_ENCODER_NONE,
        ABS_GEO_ENCODER_EPICENTRAL_DISTANCE,
        ABS_GEO_ENCODER_RELATIVE_GEOMETRY,
    }:
        raise ValueError(f"Unsupported abs_geo_encoder_type: {abs_geo_encoder_type}")
    return mode


def uses_epicentral_distance(abs_geo_encoder_type: str | None) -> bool:
    return normalize_abs_geo_encoder_type(abs_geo_encoder_type) == ABS_GEO_ENCODER_EPICENTRAL_DISTANCE


def uses_relative_geometry(abs_geo_encoder_type: str | None) -> bool:
    return normalize_abs_geo_encoder_type(abs_geo_encoder_type) == ABS_GEO_ENCODER_RELATIVE_GEOMETRY


def uses_distance_conditioning(abs_geo_encoder_type: str | None) -> bool:
    return uses_epicentral_distance(abs_geo_encoder_type) or uses_relative_geometry(abs_geo_encoder_type)


def conditioning_arg_names(abs_geo_encoder_type: str | None) -> tuple[str, ...]:
    if uses_distance_conditioning(abs_geo_encoder_type):
        return (*BASE_CONDITIONING_ARG_NAMES, *EPICENTRAL_DISTANCE_ARG_NAMES)
    return BASE_CONDITIONING_ARG_NAMES


def conditioning_bound_pairs(abs_geo_encoder_type: str | None) -> tuple[tuple[str, str], ...]:
    if uses_distance_conditioning(abs_geo_encoder_type):
        return (*BASE_CONDITIONING_BOUND_PAIRS, *EPICENTRAL_DISTANCE_BOUND_PAIRS)
    return BASE_CONDITIONING_BOUND_PAIRS


def conditioning_dimension(abs_geo_encoder_type: str | None) -> int:
    if uses_relative_geometry(abs_geo_encoder_type):
        return 9
    return 7 if uses_epicentral_distance(abs_geo_encoder_type) else 6


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def initial_bearing_radians(src_lat: float, src_lon: float, sta_lat: float, sta_lon: float) -> float:
    """Return the initial great-circle bearing from source to station in radians."""
    phi1 = math.radians(float(src_lat))
    phi2 = math.radians(float(sta_lat))
    dlambda = math.radians(float(sta_lon) - float(src_lon))
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.atan2(y, x)


def encode_relative_geometry(
    sta_lat: float,
    sta_lon: float,
    src_lat: float,
    src_lon: float,
    distance_max_km: float,
    output_min: float,
    output_max: float,
) -> tuple[float, float, float]:
    """Encode log epicentral distance plus a raw unit-circle source-to-station bearing."""
    distance_max_km = float(distance_max_km)
    if distance_max_km <= 0.0:
        raise ValueError(f"distance_max_km must be positive, got {distance_max_km}")
    distance_km = haversine_km(sta_lat, sta_lon, src_lat, src_lon)
    log_fraction = math.log1p(distance_km) / math.log1p(distance_max_km)
    if log_fraction > 1.0 + LABEL_EPS:
        raise ValueError(
            f"Epicentral distance {distance_km} km exceeds configured maximum {distance_max_km} km"
        )
    log_fraction = min(max(log_fraction, 0.0), 1.0)
    encoded_distance = log_fraction * (float(output_max) - float(output_min)) + float(output_min)
    azimuth = initial_bearing_radians(src_lat, src_lon, sta_lat, sta_lon)
    return encoded_distance, math.sin(azimuth), math.cos(azimuth)


def project_condition_value(
    value: float,
    min_num: float,
    max_num: float,
    min_get: float,
    max_get: float,
    label_name: str,
) -> float:
    min_num = float(min_num)
    max_num = float(max_num)
    value = float(value)
    if max_num <= min_num:
        raise ValueError(f"Invalid label range for {label_name}: min={min_num}, max={max_num}")
    if value < min_num - LABEL_EPS or value > max_num + LABEL_EPS:
        raise ValueError(f"Value {value} for {label_name} is outside configured range [{min_num}, {max_num}]")
    scaled = (value - min_num) / (max_num - min_num)
    return scaled * (float(max_get) - float(min_get)) + float(min_get)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_default_runtime_stats_path() -> Path:
    return DEFAULT_RUNTIME_STATS_PATH


def get_default_scale_stats_path() -> Path:
    return DEFAULT_SCALE_STATS_PATH


def load_stats_payload(path: str | Path) -> dict[str, object]:
    stats_path = Path(path)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    payload["stats_path"] = str(stats_path)
    return payload


def load_scale_stats(path: str | Path) -> dict[str, object]:
    payload = load_stats_payload(path)
    payload["waveform_norm_mode"] = WAVEFORM_NORM_GLOBAL_ZSCORE
    if "waveform_mean_value" not in payload:
        raise ValueError(f"Scale stats file is missing 'waveform_mean_value': {payload['stats_path']}")
    if "waveform_std_value" not in payload:
        raise ValueError(f"Scale stats file is missing 'waveform_std_value': {payload['stats_path']}")
    mean_value = float(payload["waveform_mean_value"])
    if not np.isfinite(mean_value):
        raise ValueError(f"Invalid waveform_mean_value in {payload['stats_path']}: {mean_value}")
    std_value = float(payload["waveform_std_value"])
    if not np.isfinite(std_value) or std_value <= 0:
        raise ValueError(f"Invalid waveform_std_value in {payload['stats_path']}: {std_value}")
    payload["waveform_mean_value"] = mean_value
    payload["waveform_std_value"] = std_value
    return payload


def load_conditioning_defaults(
    path: str | Path,
    abs_geo_encoder_type: str | None = ABS_GEO_ENCODER_NONE,
) -> dict[str, object]:
    payload = load_stats_payload(path)
    defaults = payload.get("conditioning_defaults")
    if defaults is None:
        raise ValueError(
            f"Dataset runtime stats file is missing 'conditioning_defaults': {payload['stats_path']}"
        )
    validated: dict[str, float] = {}
    for key in conditioning_arg_names(abs_geo_encoder_type):
        if key not in defaults:
            raise ValueError(
                f"Dataset runtime stats file is missing conditioning default '{key}': "
                f"{payload['stats_path']}"
            )
        value = float(defaults[key])
        if not np.isfinite(value):
            raise ValueError(
                f"Dataset runtime stats file has non-finite conditioning default '{key}': "
                f"{payload['stats_path']}"
            )
        validated[key] = value
    for min_key, max_key in conditioning_bound_pairs(abs_geo_encoder_type):
        if validated[min_key] >= validated[max_key]:
            raise ValueError(
                f"Invalid conditioning range in {payload['stats_path']}: "
                f"{min_key}={validated[min_key]} must be < {max_key}={validated[max_key]}"
            )
    payload["conditioning_defaults"] = validated
    return payload


def normalize_waveform_array(
    waveform: np.ndarray,
    mean_value: float = 0.0,
    std_value: float = 1.0,
) -> np.ndarray:
    std_value = float(std_value)
    mean_value = float(mean_value)
    if not np.isfinite(mean_value):
        raise ValueError(f"waveform mean_value must be finite, got {mean_value}")
    if not np.isfinite(std_value) or std_value <= 0:
        raise ValueError(f"waveform std_value must be > 0, got {std_value}")
    normalized = (np.asarray(waveform, dtype=np.float32) - mean_value) / std_value
    return normalized.astype(np.float32, copy=False)


def denormalize_waveform_array(
    waveform: np.ndarray,
    mean_value: float = 0.0,
    std_value: float = 1.0,
) -> np.ndarray:
    std_value = float(std_value)
    mean_value = float(mean_value)
    if not np.isfinite(mean_value):
        raise ValueError(f"waveform mean_value must be finite, got {mean_value}")
    if not np.isfinite(std_value) or std_value <= 0:
        raise ValueError(f"waveform std_value must be > 0, got {std_value}")
    restored = np.asarray(waveform, dtype=np.float32) * std_value + mean_value
    return restored.astype(np.float32, copy=False)


def _get_namespace_value(namespace: argparse.Namespace | object | None, key: str):
    if namespace is None:
        return None
    return getattr(namespace, key, None)


def _looks_like_path(value) -> bool:
    if value in (None, ""):
        return False
    try:
        return Path(value).is_file()
    except TypeError:
        return False


def _build_stats_candidates(
    args,
    checkpoint_args: argparse.Namespace | object | None = None,
    prefer_checkpoint: bool = False,
) -> list[Path]:
    cli_stats_path = _get_namespace_value(args, "waveform_scale_stats")
    ckpt_stats_path = _get_namespace_value(checkpoint_args, "waveform_scale_stats")

    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(value) -> None:
        if value in (None, ""):
            return
        candidate = Path(value)
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)

    add_candidate(cli_stats_path)
    if prefer_checkpoint:
        add_candidate(ckpt_stats_path)
    add_candidate(get_default_runtime_stats_path())
    if not prefer_checkpoint:
        add_candidate(ckpt_stats_path)
    return candidates


def _load_first_available_payload(candidates: list[Path], loader) -> tuple[dict[str, object] | None, str | None]:
    for candidate in candidates:
        if candidate.is_file():
            payload = loader(candidate)
            return payload, str(candidate)
    return None, None


def resolve_waveform_scale_config(
    args,
    checkpoint_args: argparse.Namespace | object | None = None,
    prefer_checkpoint: bool = False,
    require_scale: bool = True,
) -> dict[str, object]:
    cli_mean_value = _get_namespace_value(args, "waveform_mean_value")
    cli_std_value = _get_namespace_value(args, "waveform_std_value")
    ckpt_mean_value = _get_namespace_value(checkpoint_args, "waveform_mean_value")
    ckpt_std_value = _get_namespace_value(checkpoint_args, "waveform_std_value")
    stats_payload, stats_path_str = _load_first_available_payload(
        _build_stats_candidates(args, checkpoint_args=checkpoint_args, prefer_checkpoint=prefer_checkpoint),
        load_scale_stats,
    )

    if cli_mean_value is not None:
        mean_value = float(cli_mean_value)
        scale_source = "cli"
    elif prefer_checkpoint and ckpt_mean_value is not None:
        mean_value = float(ckpt_mean_value)
        scale_source = "checkpoint"
    elif stats_payload is not None and stats_payload.get("waveform_mean_value") is not None:
        mean_value = float(stats_payload["waveform_mean_value"])
        scale_source = f"stats:{stats_path_str}"
    elif ckpt_mean_value is not None:
        mean_value = float(ckpt_mean_value)
        scale_source = "checkpoint"
    elif require_scale:
        raise FileNotFoundError(
            "Unable to resolve waveform mean. Provide dataset runtime stats with "
            "'waveform_mean_value', --waveform-mean-value, or a checkpoint that stores it."
        )
    else:
        mean_value = None
        scale_source = None

    if cli_std_value is not None:
        std_value = float(cli_std_value)
    elif prefer_checkpoint and ckpt_std_value is not None:
        std_value = float(ckpt_std_value)
    elif stats_payload is not None and stats_payload.get("waveform_std_value") is not None:
        std_value = float(stats_payload["waveform_std_value"])
    elif ckpt_std_value is not None:
        std_value = float(ckpt_std_value)
    elif require_scale:
        raise FileNotFoundError(
            "Unable to resolve waveform std. Provide dataset runtime stats with "
            "'waveform_std_value', --waveform-std-value, or a checkpoint that stores it."
        )
    else:
        std_value = None

    if mean_value is not None and not np.isfinite(mean_value):
        raise ValueError(f"waveform_mean_value must be finite, got {mean_value}")
    if require_scale and (std_value is None or not np.isfinite(std_value) or std_value <= 0):
        raise ValueError(f"waveform_std_value must be > 0 for global_zscore, got {std_value}")

    return {
        "waveform_norm_mode": WAVEFORM_NORM_GLOBAL_ZSCORE,
        "waveform_mean_value": mean_value,
        "waveform_std_value": std_value,
        "waveform_scale_source": scale_source,
        "waveform_scale_stats": stats_path_str,
        "waveform_scale_stats_payload": stats_payload,
    }


def resolve_conditioning_config(
    args,
    checkpoint_args: argparse.Namespace | object | None = None,
    prefer_checkpoint: bool = False,
    require_ranges: bool = True,
) -> dict[str, object]:
    abs_geo_encoder_type = normalize_abs_geo_encoder_type(
        _get_namespace_value(args, "abs_geo_encoder_type")
    )
    stats_payload = None
    stats_path_str = None
    for candidate in _build_stats_candidates(
        args,
        checkpoint_args=checkpoint_args,
        prefer_checkpoint=prefer_checkpoint,
    ):
        if not candidate.is_file():
            continue
        try:
            stats_payload = load_conditioning_defaults(candidate, abs_geo_encoder_type=abs_geo_encoder_type)
            stats_path_str = str(candidate)
            break
        except ValueError:
            continue
    stats_defaults = None if stats_payload is None else stats_payload["conditioning_defaults"]

    resolved: dict[str, float] = {}
    sources: list[str] = []
    for key in conditioning_arg_names(abs_geo_encoder_type):
        cli_value = _get_namespace_value(args, key)
        ckpt_value = _get_namespace_value(checkpoint_args, key)
        stats_value = None if stats_defaults is None else stats_defaults[key]
        if cli_value is not None:
            value = float(cli_value)
            source = "cli"
        elif prefer_checkpoint and ckpt_value is not None:
            value = float(ckpt_value)
            source = "checkpoint"
        elif stats_value is not None:
            value = float(stats_value)
            source = f"stats:{stats_path_str}"
        elif ckpt_value is not None:
            value = float(ckpt_value)
            source = "checkpoint"
        elif require_ranges:
            raise FileNotFoundError(
                "Unable to resolve conditioning ranges. Provide dataset runtime stats with "
                "'conditioning_defaults', explicit CLI bounds, or a checkpoint that stores them."
            )
        else:
            continue
        if not np.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value}")
        resolved[key] = value
        sources.append(source)

    for min_key, max_key in conditioning_bound_pairs(abs_geo_encoder_type):
        if min_key not in resolved or max_key not in resolved:
            continue
        if resolved[min_key] >= resolved[max_key]:
            raise ValueError(
                f"Invalid conditioning range: {min_key}={resolved[min_key]} must be < "
                f"{max_key}={resolved[max_key]}"
            )

    unique_sources = sorted(set(sources))
    if not unique_sources:
        conditioning_source = None
    elif len(unique_sources) == 1:
        conditioning_source = unique_sources[0]
    else:
        conditioning_source = "mixed"

    return {
        **resolved,
        "conditioning_dimension": conditioning_dimension(abs_geo_encoder_type),
        "conditioning_source": conditioning_source,
        "conditioning_stats_path": stats_path_str,
        "conditioning_stats_payload": stats_payload,
    }


def apply_waveform_scale_config(args, config: dict[str, object]) -> None:
    args.waveform_norm_mode = config["waveform_norm_mode"]
    args.waveform_mean_value = config["waveform_mean_value"]
    args.waveform_std_value = config["waveform_std_value"]
    args.waveform_scale_source = config["waveform_scale_source"]
    args.waveform_scale_stats = config["waveform_scale_stats"]


def apply_conditioning_config(args, config: dict[str, object]) -> None:
    for key in conditioning_arg_names(getattr(args, "abs_geo_encoder_type", ABS_GEO_ENCODER_NONE)):
        setattr(args, key, config[key])
    args.conditioning_dimension = config["conditioning_dimension"]
    args.conditioning_source = config["conditioning_source"]
    args.conditioning_stats_path = config["conditioning_stats_path"]
