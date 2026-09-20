from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from waveform_scaling import haversine_km


MAG_EDGES = np.asarray((3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 9.1), dtype=np.float64)
DISTANCE_EDGES_KM = np.asarray(
    (0.0, 10.0, 30.0, 60.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 2500.0),
    dtype=np.float64,
)
DEFAULT_ALPHA = 0.75
DEFAULT_WEIGHT_MIN = 0.25
DEFAULT_WEIGHT_MAX = 4.0
TOTAL_JOINT_CELLS = (len(MAG_EDGES) - 1) * (len(DISTANCE_EDGES_KM) - 1)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ordered_filename_sha256(filenames: list[str]) -> str:
    digest = hashlib.sha256()
    for name in filenames:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_waveform_filename(name: str) -> dict[str, object]:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 8:
        raise ValueError(f"Waveform filename does not contain eight metadata fields: {name}")
    sta_lat, sta_lon, src_lat, src_lon, depth, magnitude = map(float, parts[:6])
    event_time = parts[6]
    station_code = "_".join(parts[7:])
    distance_km = haversine_km(sta_lat, sta_lon, src_lat, src_lon)
    event_id = f"{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}_{event_time}"
    return {
        "sta_lat": sta_lat,
        "sta_lon": sta_lon,
        "src_lat": src_lat,
        "src_lon": src_lon,
        "depth": depth,
        "magnitude": magnitude,
        "event_time": event_time,
        "station_code": station_code,
        "distance_km": distance_km,
        "event_id": event_id,
    }


def assign_joint_cell(magnitude: float, distance_km: float) -> int:
    mag_bin = int(np.searchsorted(MAG_EDGES, float(magnitude), side="right") - 1)
    distance_bin = int(np.searchsorted(DISTANCE_EDGES_KM, float(distance_km), side="right") - 1)
    if mag_bin < 0 or mag_bin >= len(MAG_EDGES) - 1:
        raise ValueError(f"Magnitude {magnitude} is outside [{MAG_EDGES[0]}, {MAG_EDGES[-1]}).")
    if distance_bin < 0 or distance_bin >= len(DISTANCE_EDGES_KM) - 1:
        raise ValueError(
            f"Epicentral distance {distance_km} km is outside "
            f"[{DISTANCE_EDGES_KM[0]}, {DISTANCE_EDGES_KM[-1]})."
        )
    return mag_bin * (len(DISTANCE_EDGES_KM) - 1) + distance_bin


def build_weight_manifest(
    waveform_dir: str | Path,
    output_path: str | Path,
    alpha: float = DEFAULT_ALPHA,
    weight_min: float = DEFAULT_WEIGHT_MIN,
    weight_max: float = DEFAULT_WEIGHT_MAX,
) -> tuple[Path, Path]:
    waveform_dir = Path(waveform_dir)
    output_path = Path(output_path)
    if not waveform_dir.is_dir():
        raise FileNotFoundError(f"Training waveform directory not found: {waveform_dir}")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if not 0.0 < float(weight_min) <= float(weight_max):
        raise ValueError(f"Invalid weight bounds: [{weight_min}, {weight_max}]")

    filenames = sorted(path.name for path in waveform_dir.glob("*.npy"))
    if not filenames:
        raise ValueError(f"No .npy waveform files found in {waveform_dir}")

    sample_count = len(filenames)
    magnitudes = np.empty(sample_count, dtype=np.float32)
    distances_km = np.empty(sample_count, dtype=np.float32)
    cell_ids = np.empty(sample_count, dtype=np.int16)
    events_by_cell: dict[int, set[str]] = defaultdict(set)

    for index, name in enumerate(filenames):
        metadata = parse_waveform_filename(name)
        magnitude = float(metadata["magnitude"])
        distance_km = float(metadata["distance_km"])
        cell_id = assign_joint_cell(magnitude, distance_km)
        magnitudes[index] = magnitude
        distances_km[index] = distance_km
        cell_ids[index] = cell_id
        events_by_cell[cell_id].add(str(metadata["event_id"]))

    cell_counts = np.bincount(cell_ids.astype(np.int64), minlength=TOTAL_JOINT_CELLS)
    nonempty_mask = cell_counts > 0
    nonempty_cell_count = int(nonempty_mask.sum())
    record_counts = cell_counts[cell_ids.astype(np.int64)].astype(np.float64)
    raw_weights = (sample_count / (nonempty_cell_count * record_counts)) ** float(alpha)
    weights = np.clip(raw_weights, float(weight_min), float(weight_max)).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        weights=weights,
        cell_ids=cell_ids,
        magnitudes=magnitudes,
        distances_km=distances_km,
    )

    audit_path = output_path.with_suffix(".audit.json")
    nonempty_ids = np.flatnonzero(nonempty_mask)
    audit = {
        "computed_at_utc": utc_now_iso(),
        "waveform_dir": str(waveform_dir.resolve()),
        "manifest_path": str(output_path.resolve()),
        "sample_count": sample_count,
        "alpha": float(alpha),
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
        "mag_edges": MAG_EDGES.tolist(),
        "distance_edges_km": DISTANCE_EDGES_KM.tolist(),
        "nonempty_cell_count": nonempty_cell_count,
        "cell_record_counts": {str(int(cell_id)): int(cell_counts[cell_id]) for cell_id in nonempty_ids},
        "cell_unique_event_counts": {
            str(int(cell_id)): len(events_by_cell[int(cell_id)]) for cell_id in nonempty_ids
        },
        "weight_quantiles": {
            key: float(value)
            for key, value in zip(
                ("min", "p01", "p05", "median", "p95", "p99", "max"),
                np.quantile(weights, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)),
            )
        },
        "ordered_filename_sha256": ordered_filename_sha256(filenames),
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path, audit_path


def load_weight_manifest(
    manifest_path: str | Path,
    expected_filenames: list[str],
    expected_alpha: float | None = None,
    expected_weight_min: float | None = None,
    expected_weight_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    manifest_path = Path(manifest_path)
    audit_path = manifest_path.with_suffix(".audit.json")
    if not manifest_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError(f"Long-tail manifest or audit file is missing: {manifest_path}, {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_hash = ordered_filename_sha256(expected_filenames)
    if audit.get("ordered_filename_sha256") != expected_hash:
        raise ValueError("Long-tail manifest filename order/checksum does not match the training dataset.")
    if int(audit.get("sample_count", -1)) != len(expected_filenames):
        raise ValueError("Long-tail manifest sample count does not match the training dataset.")
    for key, expected in (
        ("alpha", expected_alpha),
        ("weight_min", expected_weight_min),
        ("weight_max", expected_weight_max),
    ):
        if expected is not None and not np.isclose(float(audit.get(key)), float(expected)):
            raise ValueError(f"Long-tail manifest {key}={audit.get(key)} does not match requested {expected}.")

    with np.load(manifest_path) as payload:
        weights = np.asarray(payload["weights"], dtype=np.float32)
        cell_ids = np.asarray(payload["cell_ids"], dtype=np.int64)
    if weights.shape != (len(expected_filenames),) or cell_ids.shape != weights.shape:
        raise ValueError(f"Invalid long-tail manifest shapes: weights={weights.shape}, cells={cell_ids.shape}")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Long-tail manifest contains non-finite or non-positive weights.")
    return weights, cell_ids, audit


class LongTailWeightedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        manifest_path: str | Path,
        alpha: float = DEFAULT_ALPHA,
        weight_min: float = DEFAULT_WEIGHT_MIN,
        weight_max: float = DEFAULT_WEIGHT_MAX,
    ):
        if not hasattr(base_dataset, "wf_path"):
            raise TypeError("LongTailWeightedDataset requires a dataset with a sorted wf_path list.")
        self.base_dataset = base_dataset
        self.weights, self.cell_ids, self.audit = load_weight_manifest(
            manifest_path,
            list(base_dataset.wf_path),
            expected_alpha=alpha,
            expected_weight_min=weight_min,
            expected_weight_max=weight_max,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        waveform, label = self.base_dataset[index]
        return (
            waveform,
            label,
            torch.tensor(self.weights[index], dtype=torch.float32),
            torch.tensor(self.cell_ids[index], dtype=torch.long),
        )


class JointCellDataset(Dataset):
    """Attach deterministic sample indices and joint cells for macro validation."""

    def __init__(self, base_dataset: Dataset):
        if not hasattr(base_dataset, "wf_path"):
            raise TypeError("JointCellDataset requires a dataset with a sorted wf_path list.")
        self.base_dataset = base_dataset
        cell_ids = []
        for name in base_dataset.wf_path:
            metadata = parse_waveform_filename(name)
            cell_ids.append(assign_joint_cell(float(metadata["magnitude"]), float(metadata["distance_km"])))
        self.cell_ids = np.asarray(cell_ids, dtype=np.int64)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        waveform, label = self.base_dataset[index]
        return waveform, label, torch.tensor(self.cell_ids[index], dtype=torch.long), torch.tensor(index, dtype=torch.long)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build joint magnitude-distance long-tail training weights.")
    parser.add_argument("--waveform-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--weight-min", type=float, default=DEFAULT_WEIGHT_MIN)
    parser.add_argument("--weight-max", type=float, default=DEFAULT_WEIGHT_MAX)
    return parser


if __name__ == "__main__":
    cli_args = create_parser().parse_args()
    manifest, audit = build_weight_manifest(
        cli_args.waveform_dir,
        cli_args.output,
        alpha=cli_args.alpha,
        weight_min=cli_args.weight_min,
        weight_max=cli_args.weight_max,
    )
    print(f"Wrote {manifest}")
    print(f"Wrote {audit}")
