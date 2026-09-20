from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
TRAIN_WAVEDIT_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]

if str(TRAIN_WAVEDIT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_WAVEDIT_DIR))

from validation.metrics import summarize_metric_rows
from validation.subset import CaseRecord, collect_case_records
from validation.run_baseline_eval import (
    AUTO_VARIANT,
    CONDITION_MODES,
    DEFAULT_TEST_DIR,
    MODEL_VARIANTS,
    PER_SAMPLE_FIELDNAMES,
    run_evaluation,
)

DEFAULT_RESULT_ROOT = REPO_ROOT / "result"
DEFAULT_RUNTIME_STATS = REPO_ROOT / "data" / "dataset_runtime_stats.json"
DEFAULT_RESULT_TAG = "category_eval"
TEST_CATEGORY_NAMES = (
    "future_known_station",
    "future_unknown_station",
    "historical_unknown_station",
)
VAL_FOCUS_NAME = "val_focus"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run category-wise SWaG evaluation and aggregate results."
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--val-dir", type=Path, default=REPO_ROOT / "data" / "val")
    parser.add_argument("--samples-per-class", type=int, default=64)
    parser.add_argument("--val-focus-count", type=int, default=64)
    parser.add_argument("--val-mag-target", type=float, default=4.0)
    parser.add_argument("--val-distance-target", type=float, default=90.0)
    parser.add_argument("--subset-seed", type=int, default=20260424)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--result-tag", type=str, default=DEFAULT_RESULT_TAG)
    parser.add_argument("--runtime-stats", type=Path, default=DEFAULT_RUNTIME_STATS)
    parser.add_argument("--condition-mode", type=str, default="normal", choices=CONDITION_MODES)
    parser.add_argument("--model-variant", type=str, default=AUTO_VARIANT, choices=MODEL_VARIANTS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=384)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=6)
    parser.add_argument("--hidden-feature", type=int, default=168)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=8)
    parser.add_argument("--position-embedding", dest="position_embedding", type=str, default="sincos", choices=["sincos", "rope"])
    parser.add_argument("--rope-base", dest="rope_base", type=float, default=10000.0)
    parser.add_argument("--min-get", type=float, default=0.0)
    parser.add_argument("--max-get", type=float, default=1000.0)
    parser.add_argument("--cond-embedding-scale", default=1.0, type=float)
    return parser


def group_names() -> tuple[str, ...]:
    return (*TEST_CATEGORY_NAMES, VAL_FOCUS_NAME)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_runtime_defaults(path: Path) -> dict[str, object]:
    payload = read_json(path)
    conditioning_defaults = payload.get("conditioning_defaults")
    if not isinstance(conditioning_defaults, dict):
        raise ValueError(f"conditioning_defaults missing in runtime stats: {path}")
    if "waveform_mean_value" not in payload:
        raise ValueError(f"waveform_mean_value missing in runtime stats: {path}")
    if "waveform_std_value" not in payload:
        raise ValueError(f"waveform_std_value missing in runtime stats: {path}")
    return payload


def build_run_dir(args: argparse.Namespace) -> Path:
    date_tag = datetime.now().strftime("%Y%m%d")
    return args.result_root / f"{args.result_tag}_{args.ckpt.stem}_{date_tag}"


def build_manifest_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "ckpt": str(args.ckpt.resolve()),
        "test_root": str(args.test_root.resolve()),
        "val_dir": str(args.val_dir.resolve()),
        "samples_per_class": args.samples_per_class,
        "val_focus_count": args.val_focus_count,
        "val_mag_target": args.val_mag_target,
        "val_distance_target": args.val_distance_target,
        "subset_seed": args.subset_seed,
        "runtime_stats": str(args.runtime_stats.resolve()),
        "condition_mode": args.condition_mode,
        "model_variant": args.model_variant,
        "batch_size": args.batch_size,
        "cfg_scale": args.cfg_scale,
        "num_sampling_steps": args.num_sampling_steps,
        "sampling_hz": args.sampling_hz,
        "length": args.length,
        "depth": args.depth,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "hidden_feature": args.hidden_feature,
        "patch_size": args.patch_size,
        "position_embedding": args.position_embedding,
        "rope_base": args.rope_base,
        "min_get": args.min_get,
        "max_get": args.max_get,
        "cond_embedding_scale": args.cond_embedding_scale,
        "categories": list(group_names()),
    }


def build_initial_manifest(args: argparse.Namespace, run_dir: Path) -> dict[str, object]:
    categories: dict[str, dict[str, object]] = {}
    for category in group_names():
        if category == VAL_FOCUS_NAME:
            test_dir = args.val_dir
        else:
            test_dir = args.test_root / category
        categories[category] = {
            "status": "pending",
            "test_dir": str(test_dir.resolve()),
            "out_dir": str((run_dir / category).resolve()),
            "started_at": None,
            "completed_at": None,
            "selected_count": 0,
            "error": None,
            "summary_path": None,
        }
    now = utc_now_iso()
    return {
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "run_dir": str(run_dir.resolve()),
        "config": build_manifest_config(args),
        "categories": categories,
    }


def load_or_create_manifest(args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, object], Path]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        expected = build_manifest_config(args)
        if manifest.get("config") != expected:
            raise RuntimeError(
                "Existing run_manifest.json does not match the current command parameters. "
                f"Run dir: {run_dir}"
            )
        return manifest, manifest_path
    manifest = build_initial_manifest(args, run_dir)
    write_json(manifest_path, manifest)
    return manifest, manifest_path


def persist_manifest(manifest_path: Path, manifest: dict[str, object]) -> None:
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)


def select_val_focus_records(
    val_dir: Path,
    count: int,
    magnitude_target: float,
    distance_target: float,
) -> list[CaseRecord]:
    records = collect_case_records(val_dir)
    if not records:
        raise RuntimeError(f"No .npy files found under {val_dir}")
    ranked = sorted(
        records,
        key=lambda record: (
            abs(float(record.magnitude) - magnitude_target),
            abs(float(record.epicentral_distance_km) - distance_target),
            record.name,
        ),
    )
    return ranked[: min(count, len(ranked))]


def build_category_args(args: argparse.Namespace, category: str, out_dir: Path) -> argparse.Namespace:
    stats_payload = args.runtime_defaults
    conditioning_defaults = stats_payload["conditioning_defaults"]
    test_dir = args.val_dir if category == VAL_FOCUS_NAME else (args.test_root / category)
    subset_size = args.val_focus_count if category == VAL_FOCUS_NAME else args.samples_per_class
    return argparse.Namespace(
        ckpt=args.ckpt,
        model_variant=args.model_variant,
        test_dir=test_dir,
        out_dir=out_dir,
        subset_size=subset_size,
        full_test=False,
        batch_size=args.batch_size,
        cfg_scale=args.cfg_scale,
        condition_mode=args.condition_mode,
        num_sampling_steps=args.num_sampling_steps,
        sampling_hz=args.sampling_hz,
        length=args.length,
        depth=args.depth,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        hidden_feature=args.hidden_feature,
        patch_size=args.patch_size,
        abs_geo_encoder_type="none",
        position_embedding=args.position_embedding,
        rope_base=args.rope_base,
        subset_seed=args.subset_seed,
        result_tag=f"{args.result_tag}_{category}",
        save_pred_npy=True,
        all_sample_figures=True,
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
        min_get=args.min_get,
        max_get=args.max_get,
        cond_embedding_scale=args.cond_embedding_scale,
        waveform_scale_stats=str(args.runtime_stats),
        waveform_mean_value=stats_payload["waveform_mean_value"],
        waveform_std_value=stats_payload["waveform_std_value"],
        update_history=False,
    )


def run_category(
    args: argparse.Namespace,
    category: str,
    manifest: dict[str, object],
    manifest_path: Path,
    run_dir: Path,
) -> None:
    entry = manifest["categories"][category]
    out_dir = run_dir / category
    summary_path = out_dir / "overall_summary.json"
    if entry.get("status") == "completed" and summary_path.exists():
        return

    entry["status"] = "running"
    entry["started_at"] = utc_now_iso()
    entry["completed_at"] = None
    entry["error"] = None
    entry["out_dir"] = str(out_dir.resolve())
    persist_manifest(manifest_path, manifest)

    try:
        category_args = build_category_args(args, category, out_dir)
        selected_records_override = None
        if category == VAL_FOCUS_NAME:
            selected_records_override = select_val_focus_records(
                args.val_dir,
                args.val_focus_count,
                args.val_mag_target,
                args.val_distance_target,
            )
        resolved_run_dir = run_evaluation(category_args, selected_records_override=selected_records_override)
        summary = read_json(Path(resolved_run_dir) / "overall_summary.json")
        entry["status"] = "completed"
        entry["completed_at"] = utc_now_iso()
        entry["summary_path"] = str((Path(resolved_run_dir) / "overall_summary.json").resolve())
        entry["selected_count"] = int(summary["count"])
        entry["variant"] = summary.get("variant")
        entry["device"] = summary.get("device")
        entry["precision"] = summary.get("precision")
        entry["error"] = None
    except Exception as exc:
        entry["status"] = "failed"
        entry["completed_at"] = utc_now_iso()
        entry["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        persist_manifest(manifest_path, manifest)
        raise

    persist_manifest(manifest_path, manifest)


def relative_link(target: Path, base: Path) -> str:
    return target.relative_to(base).as_posix()


def build_summary_table_lines(metric_stats: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Metric | Mean | Median | P90 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric_name, stats in metric_stats.items():
        lines.append(
            f"| {metric_name} | {float(stats['mean']):.6f} | {float(stats['median']):.6f} | {float(stats['p90']):.6f} |"
        )
    return lines


def aggregate_completed_runs(
    args: argparse.Namespace,
    manifest: dict[str, object],
    run_dir: Path,
    manifest_path: Path,
) -> None:
    aggregate_rows: list[dict[str, object]] = []
    category_summaries: dict[str, dict[str, object]] = {}
    completed_categories: list[str] = []

    for category in group_names():
        entry = manifest["categories"][category]
        if entry.get("status") != "completed":
            continue
        category_dir = run_dir / category
        summary_path = category_dir / "overall_summary.json"
        metrics_path = category_dir / "per_sample_metrics.csv"
        if not summary_path.exists() or not metrics_path.exists():
            continue
        summary = read_json(summary_path)
        rows = read_csv_rows(metrics_path)
        for row in rows:
            aggregate_rows.append({"category": category, **row})
        category_summaries[category] = summary
        completed_categories.append(category)

    fieldnames = ["category", *PER_SAMPLE_FIELDNAMES]
    write_csv_rows(run_dir / "aggregate_per_sample_metrics.csv", fieldnames, aggregate_rows)

    overall_metric_stats = summarize_metric_rows(aggregate_rows) if aggregate_rows else {}
    aggregate_summary = {
        "status": "completed" if len(completed_categories) == len(group_names()) else "partial",
        "created_at": manifest["created_at"],
        "updated_at": utc_now_iso(),
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "ckpt_path": str(args.ckpt.resolve()),
        "test_root": str(args.test_root.resolve()),
        "val_dir": str(args.val_dir.resolve()),
        "samples_per_class": args.samples_per_class,
        "val_focus_count": args.val_focus_count,
        "val_mag_target": args.val_mag_target,
        "val_distance_target": args.val_distance_target,
        "subset_seed": args.subset_seed,
        "runtime_stats": str(args.runtime_stats.resolve()),
        "condition_mode": args.condition_mode,
        "model_variant": args.model_variant,
        "total_count": len(aggregate_rows),
        "completed_categories": completed_categories,
        "category_counts": {
            category: int(summary["count"]) for category, summary in category_summaries.items()
        },
        "metric_stats": overall_metric_stats,
        "categories": category_summaries,
    }
    write_json(run_dir / "aggregate_summary.json", aggregate_summary)

    lines = [
        "# Category Evaluation Summary",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Test root: `{args.test_root}`",
        f"- Val dir: `{args.val_dir}`",
        f"- Run dir: `{run_dir}`",
        f"- Runtime stats: `{args.runtime_stats}`",
        f"- Condition mode: `{args.condition_mode}`",
        f"- Requested samples per class: `{args.samples_per_class}`",
        f"- Requested val focus count: `{args.val_focus_count}`",
        f"- Val focus target: `M≈{args.val_mag_target:.2f}`, `Dist≈{args.val_distance_target:.2f} km`",
        f"- Completed categories: `{len(completed_categories)}/{len(group_names())}`",
        f"- Total selected samples: `{len(aggregate_rows)}`",
        "",
        "## Category Status",
        "",
        "| Category | Status | Count | Variant | Device | Report |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for category in group_names():
        entry = manifest["categories"][category]
        summary = category_summaries.get(category, {})
        report_path = run_dir / category / "report.md"
        report_cell = (
            f"[report]({relative_link(report_path, run_dir)})"
            if report_path.exists()
            else "-"
        )
        lines.append(
            f"| {category} | {entry.get('status', '-')} | {int(summary.get('count', 0))} | "
            f"{summary.get('variant', '-')} | {summary.get('device', '-')} | {report_cell} |"
        )

    lines.extend(["", "## Overall Metrics", ""])
    if overall_metric_stats:
        lines.extend(build_summary_table_lines(overall_metric_stats))
    else:
        lines.append("No completed category results were available.")

    lines.extend(["", "## Per-Category Counts", ""])
    for category in group_names():
        if category not in category_summaries:
            continue
        summary = category_summaries[category]
        lines.append(
            f"- `{category}`: `{int(summary['count'])}` samples, variant `{summary.get('variant', '-')}`, "
            f"device `{summary.get('device', '-')}`"
        )

    (run_dir / "aggregate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_inputs(args: argparse.Namespace) -> None:
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.test_root.exists():
        raise FileNotFoundError(f"Test root not found: {args.test_root}")
    if not args.val_dir.exists():
        raise FileNotFoundError(f"Val dir not found: {args.val_dir}")
    if not args.runtime_stats.exists():
        raise FileNotFoundError(f"Runtime stats not found: {args.runtime_stats}")
    if args.samples_per_class <= 0:
        raise ValueError(f"--samples-per-class must be > 0, got {args.samples_per_class}")
    if args.val_focus_count <= 0:
        raise ValueError(f"--val-focus-count must be > 0, got {args.val_focus_count}")
    missing = [category for category in TEST_CATEGORY_NAMES if not (args.test_root / category).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing category directories under {args.test_root}: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_inputs(args)
    args.runtime_defaults = load_runtime_defaults(args.runtime_stats)

    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_path = load_or_create_manifest(args, run_dir)
    manifest["status"] = "running"
    persist_manifest(manifest_path, manifest)

    try:
        for category in group_names():
            run_category(args, category, manifest, manifest_path, run_dir)
        aggregate_completed_runs(args, manifest, run_dir, manifest_path)
    except Exception:
        manifest["status"] = "failed"
        persist_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    persist_manifest(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
