from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from waveform_scaling import haversine_km


MAG_BINS = [3.0, 4.0, 5.0, 6.0, 7.0]
DEPTH_BINS = [0.0, 30.0, 70.0, 150.0, 300.0]
DIST_BINS = [0.0, 50.0, 100.0, 200.0, 400.0]

MAGNITUDE_LABELS = ["[3,4)", "[4,5)", "[5,6)", "[6,7)", "[7,+)"]
DEPTH_LABELS = ["[0,30)", "[30,70)", "[70,150)", "[150,300)", "[300,+)"]
DISTANCE_LABELS = ["[0,50)", "[50,100)", "[100,200)", "[200,400)", "[400,+)"]

SELECTED_CASE_FIELDNAMES = [
    "name",
    "path",
    "station",
    "event_id",
    "sta_lat",
    "sta_lon",
    "src_lat",
    "src_lon",
    "depth_km",
    "magnitude",
    "epicentral_distance_km",
    "magnitude_bucket",
    "depth_bucket",
    "distance_bucket",
]


@dataclass(frozen=True)
class CaseRecord:
    path: Path
    name: str
    station: str
    event_id: str
    sta_lat: float
    sta_lon: float
    src_lat: float
    src_lon: float
    depth_km: float
    magnitude: float
    epicentral_distance_km: float
    magnitude_bucket: str
    depth_bucket: str
    distance_bucket: str

    @property
    def bucket_key(self) -> tuple[str, str, str]:
        return (
            self.magnitude_bucket,
            self.depth_bucket,
            self.distance_bucket,
        )

    def to_row(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "station": self.station,
            "event_id": self.event_id,
            "sta_lat": self.sta_lat,
            "sta_lon": self.sta_lon,
            "src_lat": self.src_lat,
            "src_lon": self.src_lon,
            "depth_km": self.depth_km,
            "magnitude": self.magnitude,
            "epicentral_distance_km": self.epicentral_distance_km,
            "magnitude_bucket": self.magnitude_bucket,
            "depth_bucket": self.depth_bucket,
            "distance_bucket": self.distance_bucket,
        }


def bucket_from_edges(value: float, edges: list[float], labels: list[str]) -> str:
    for index in range(len(labels) - 1):
        if edges[index] <= value < edges[index + 1]:
            return labels[index]
    return labels[-1]


def magnitude_bucket(magnitude: float) -> str:
    return bucket_from_edges(magnitude, MAG_BINS, MAGNITUDE_LABELS)


def depth_bucket(depth_km: float) -> str:
    return bucket_from_edges(depth_km, DEPTH_BINS, DEPTH_LABELS)


def distance_bucket(distance_km: float) -> str:
    return bucket_from_edges(distance_km, DIST_BINS, DISTANCE_LABELS)


def parse_case_record(path: Path) -> CaseRecord:
    parts = path.stem.split("_")
    if len(parts) < 8:
        raise ValueError(f"Invalid test filename: {path.name}")

    sta_lat = float(parts[0])
    sta_lon = float(parts[1])
    src_lat = float(parts[2])
    src_lon = float(parts[3])
    depth_km = float(parts[4])
    magnitude = float(parts[5])
    event_id = parts[6]
    station = "_".join(parts[7:])
    epi_dist = haversine_km(sta_lat, sta_lon, src_lat, src_lon)
    return CaseRecord(
        path=path,
        name=path.stem,
        station=station,
        event_id=event_id,
        sta_lat=sta_lat,
        sta_lon=sta_lon,
        src_lat=src_lat,
        src_lon=src_lon,
        depth_km=depth_km,
        magnitude=magnitude,
        epicentral_distance_km=epi_dist,
        magnitude_bucket=magnitude_bucket(magnitude),
        depth_bucket=depth_bucket(depth_km),
        distance_bucket=distance_bucket(epi_dist),
    )


def collect_case_records(test_dir: Path) -> list[CaseRecord]:
    return [parse_case_record(path) for path in sorted(test_dir.glob("*.npy"))]


def largest_remainder_allocation(
    bucket_sizes: dict[tuple[str, str, str], int], quota: int
) -> dict[tuple[str, str, str], int]:
    if quota <= 0:
        return {bucket: 0 for bucket in bucket_sizes}

    total = sum(bucket_sizes.values())
    if total == 0:
        return {bucket: 0 for bucket in bucket_sizes}

    raw = {bucket: size * quota / total for bucket, size in bucket_sizes.items()}
    alloc = {
        bucket: min(size, int(math.floor(raw[bucket])))
        for bucket, size in bucket_sizes.items()
    }
    assigned = sum(alloc.values())
    remainders = sorted(
        bucket_sizes,
        key=lambda bucket: (raw[bucket] - math.floor(raw[bucket]), bucket_sizes[bucket], str(bucket)),
        reverse=True,
    )
    remaining = quota - assigned
    while remaining > 0:
        progressed = False
        for bucket in remainders:
            if alloc[bucket] < bucket_sizes[bucket]:
                alloc[bucket] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return alloc


def select_stratified_subset(
    records: list[CaseRecord], subset_size: int, seed: int
) -> list[CaseRecord]:
    if subset_size <= 0 or subset_size >= len(records):
        return list(records)

    buckets: dict[tuple[str, str, str], list[CaseRecord]] = {}
    for record in records:
        buckets.setdefault(record.bucket_key, []).append(record)
    bucket_sizes = {bucket: len(items) for bucket, items in buckets.items()}
    alloc = largest_remainder_allocation(bucket_sizes, subset_size)

    selected: list[CaseRecord] = []
    for bucket in sorted(buckets, key=str):
        items = sorted(buckets[bucket], key=lambda item: item.name)
        keep_count = min(alloc.get(bucket, 0), len(items))
        rng = random.Random(f"{seed}-{bucket}")
        chosen = rng.sample(items, keep_count) if keep_count > 0 else []
        selected.extend(sorted(chosen, key=lambda item: item.name))
    return sorted(selected, key=lambda item: item.name)


def write_selected_cases_csv(records: list[CaseRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_CASE_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
