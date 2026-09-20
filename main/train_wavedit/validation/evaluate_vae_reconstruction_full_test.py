import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT_DIR = Path(__file__).resolve().parents[3]
TRAIN_WAVEDIT_DIR = ROOT_DIR / "main" / "train_wavedit"
if str(TRAIN_WAVEDIT_DIR) not in sys.path:
    sys.path.append(str(TRAIN_WAVEDIT_DIR))

from data_load import LoadWaveform
from validation.metrics import compute_sample_metrics, summarize_metric_rows
from waveform_scaling import denormalize_waveform_array
from waveform_vae import load_waveform_vae_checkpoint


CATEGORIES = ("future_known_station", "future_unknown_station", "historical_unknown_station")
EXPECTED_COMPLETE_TEST_COUNT = 43641
EXPECTED_CATEGORY_COUNTS = {
    "future_known_station": 8372,
    "future_unknown_station": 999,
    "historical_unknown_station": 34270,
}
LOG_RESIDUAL_KEYS = (
    "pga_log_resid",
    "pgv_log_resid",
    "psa_0p2_log_resid",
    "psa_0p5_log_resid",
    "psa_1p0_log_resid",
    "psa_2p0_log_resid",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VAE reconstruction on the complete three-scenario test set.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--runtime-stats", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--limit-per-category", type=int, default=0)
    return parser


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(waveform_dir: Path, stats: dict[str, object]) -> LoadWaveform:
    conditions = stats["conditioning_defaults"]
    return LoadWaveform(
        str(waveform_dir),
        conditions["sta_lat_min"],
        conditions["sta_lat_max"],
        conditions["sta_lon_min"],
        conditions["sta_lon_max"],
        conditions["src_lat_min"],
        conditions["src_lat_max"],
        conditions["src_lon_min"],
        conditions["src_lon_max"],
        conditions["dep_min"],
        conditions["dep_max"],
        conditions["mag_min"],
        conditions["mag_max"],
        0,
        1000,
        3000,
        stats["waveform_mean_value"],
        stats["waveform_std_value"],
    )


def flatten_summary(category: str, summary: dict[str, dict[str, float]], n: int) -> dict[str, object]:
    row: dict[str, object] = {
        "category": category,
        "sample_count": n,
        "reconstruction_mode": "posterior_mean_sample_false",
    }
    for metric, values in summary.items():
        for statistic, value in values.items():
            row[f"{metric}_{statistic}"] = float(value)
    return row


def summarize_reconstruction_metrics(metric_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = summarize_metric_rows(metric_rows)
    for key in LOG_RESIDUAL_KEYS:
        values = np.asarray([float(row[f"{key}_abs"]) for row in metric_rows], dtype=float)
        summary[f"{key}_abs"] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return summary


def denormalize(wave: np.ndarray, stats: dict[str, object]) -> np.ndarray:
    return denormalize_waveform_array(
        wave.transpose(1, 0),
        mean_value=stats["waveform_mean_value"],
        std_value=stats["waveform_std_value"],
    )


def add_absolute_residuals(metric: dict[str, float]) -> dict[str, float]:
    enriched = dict(metric)
    for key in LOG_RESIDUAL_KEYS:
        enriched[f"{key}_abs"] = abs(float(metric[key]))
    return enriched


def evaluate_category(
    category: str,
    dataset: LoadWaveform,
    model,
    stats: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    indices = list(range(len(dataset)))
    if args.limit_per_category > 0:
        indices = indices[: args.limit_per_category]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    metric_rows: list[dict[str, float]] = []
    record_rows: list[dict[str, object]] = []
    processed = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True).float()
            recon = model(x, sample=False)["recon"].detach().cpu().numpy()
            target = x.detach().cpu().numpy()
            for local_idx, (real_norm, recon_norm) in enumerate(zip(target, recon)):
                dataset_index = indices[processed + local_idx]
                real = denormalize(real_norm, stats)
                predicted = denormalize(recon_norm, stats)
                metric = add_absolute_residuals(compute_sample_metrics(real, predicted, args.sampling_hz))
                metric_rows.append(metric)
                record_rows.append(
                    {
                        "category": category,
                        "test_index": dataset_index,
                        "filename": dataset.wf_path[dataset_index],
                        **{key: float(value) for key, value in metric.items()},
                    }
                )
            processed += len(target)
            print(f"{category}: VAE reconstruction evaluated {processed}/{len(indices)}", flush=True)
    return metric_rows, record_rows


def composite_score(metric: dict[str, object]) -> float:
    return float(
        np.mean([float(metric[f"{key}_abs"]) for key in LOG_RESIDUAL_KEYS])
        + 0.25 * float(metric["log_fas_mse_0p1_20hz"])
    )


def save_quantile_examples(
    model,
    datasets: dict[str, LoadWaveform],
    record_rows: list[dict[str, object]],
    stats: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    scores = np.asarray([composite_score(row) for row in record_rows], dtype=float)
    quantiles = np.linspace(0.25, 0.75, max(1, args.num_examples))
    targets = np.quantile(scores, quantiles)
    selected_indices: list[int] = []
    for target in targets:
        candidates = np.argsort(np.abs(scores - target))
        index = next(int(item) for item in candidates if int(item) not in selected_indices)
        selected_indices.append(index)

    examples_dir = args.out_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    selected_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for number, record_index in enumerate(selected_indices, start=1):
            record = record_rows[record_index]
            category = str(record["category"])
            dataset_index = int(record["test_index"])
            x, _ = datasets[category][dataset_index]
            recon = model(x.unsqueeze(0).to(device), sample=False)["recon"][0].detach().cpu().numpy()
            real = denormalize(x.numpy(), stats)
            predicted = denormalize(recon, stats)
            stem = f"quantile_example_{number:02d}"
            np.save(examples_dir / f"{stem}_observed.npy", real)
            np.save(examples_dir / f"{stem}_reconstructed.npy", predicted)
            selected_rows.append(
                {
                    "example": stem,
                    "category": category,
                    "test_index": dataset_index,
                    "filename": record["filename"],
                    "target_composite_quantile": float(quantiles[number - 1]),
                    "composite_score": float(scores[record_index]),
                }
            )
    write_csv(args.out_dir / "vae_reconstruction_selected_examples.csv", selected_rows)


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(args.runtime_stats.read_text(encoding="utf-8"))
    datasets = {category: make_dataset(args.test_root / category, stats) for category in CATEGORIES}
    discovered_test_count = sum(len(dataset) for dataset in datasets.values())
    if args.limit_per_category <= 0:
        actual_category_counts = {category: len(dataset) for category, dataset in datasets.items()}
        if actual_category_counts != EXPECTED_CATEGORY_COUNTS:
            raise RuntimeError(
                f"Expected complete-test category counts {EXPECTED_CATEGORY_COUNTS}, found {actual_category_counts}."
            )
        if discovered_test_count != EXPECTED_COMPLETE_TEST_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_COMPLETE_TEST_COUNT} complete-test records, found {discovered_test_count}."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_waveform_vae_checkpoint(str(args.checkpoint), device=device)
    metrics_by_category: dict[str, list[dict[str, float]]] = {}
    per_record_rows: list[dict[str, object]] = []
    for category in CATEGORIES:
        metrics, records = evaluate_category(category, datasets[category], model, stats, args, device)
        metrics_by_category[category] = metrics
        per_record_rows.extend(records)

    summary_rows = [
        flatten_summary(category, summarize_reconstruction_metrics(metrics_by_category[category]), len(metrics_by_category[category]))
        for category in CATEGORIES
    ]
    all_metrics = [metric for category in CATEGORIES for metric in metrics_by_category[category]]
    summary_rows.append(flatten_summary("overall", summarize_reconstruction_metrics(all_metrics), len(all_metrics)))
    write_csv(args.out_dir / "vae_reconstruction_full_test_summary.csv", summary_rows)
    write_csv(args.out_dir / "vae_reconstruction_full_test_per_record_metrics.csv", per_record_rows)
    save_quantile_examples(model, datasets, per_record_rows, stats, args, device)

    checkpoint_args = payload.get("args", {})
    if hasattr(checkpoint_args, "__dict__"):
        checkpoint_args = vars(checkpoint_args)
    (args.out_dir / "vae_reconstruction_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "test_root": str(args.test_root),
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "categories": list(CATEGORIES),
                "discovered_complete_test_count": discovered_test_count,
                "evaluated_count": len(all_metrics),
                "limit_per_category": args.limit_per_category,
                "reconstruction_mode": "posterior mean (sample=False)",
                "selection_rule": "Examples nearest fixed overall-test composite-error quantiles; not best-case selection.",
                "checkpoint_args": checkpoint_args,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
