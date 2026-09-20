from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evaluate_generation_six_models import (
    CATEGORIES,
    DEFAULT_TEST_ROOT,
    MetricAccumulator,
    discover_models,
    evaluate_model_category,
    fmt,
    markdown_table,
    pred_path_for_row_unchecked,
    prediction_file_issue,
    ranking_rows,
    read_csv_rows,
    summary_fieldnames,
    write_csv_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOTS = (
    REPO_ROOT / "result" / "model_generation_6models_1000steps",
    REPO_ROOT / "result" / "model_generation_1models_1000steps",
)
DEFAULT_OUT_DIR = REPO_ROOT / "result" / "model_generation_combined_vae_rope_comparison" / "analysis"

CATEGORY_LABELS = {
    "future_known_station": "future_known_station",
    "future_unknown_station": "future_unknown_station",
    "historical_unknown_station": "historical_unknown_station",
    "overall": "overall",
}


@dataclass(frozen=True)
class ModelSource:
    model: str
    result_root: Path

    @property
    def model_dir(self) -> Path:
        return self.result_root / "models" / self.model

    def category_dir(self, category: str) -> Path:
        return self.model_dir / "full_test" / category

    def pred_dir(self, category: str) -> Path:
        return self.category_dir(category) / "pred_npy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generation outputs from multiple result roots on shared cases."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        action="append",
        default=None,
        help="Result root containing models/. Repeat to combine multiple roots.",
    )
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def selected_case_fingerprint(rows: list[dict[str, str]]) -> str:
    """Hash ordered sample identity, independent of path and derived float serialization."""
    canonical_rows = [row["name"] for row in rows]
    payload = json.dumps(
        canonical_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def discover_model_sources(result_roots: list[Path]) -> list[ModelSource]:
    sources: list[ModelSource] = []
    seen: set[str] = set()
    for result_root in result_roots:
        for model in discover_models(result_root):
            if model in seen:
                raise RuntimeError(
                    f"Duplicate model name {model!r}; use unique model directories before combining."
                )
            seen.add(model)
            sources.append(ModelSource(model=model, result_root=result_root))
    if not sources:
        raise RuntimeError("No model outputs found.")
    return sources


def validate_outputs_multi(
    sources: list[ModelSource],
    limit: int | None,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, object]], list[dict[str, object]]]:
    checks: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    rows_by_category: dict[str, list[dict[str, str]]] = {}
    reference_rows_by_category: dict[str, list[dict[str, str]]] = {}
    excluded_indices_by_category: dict[str, set[int]] = {category: set() for category in CATEGORIES}
    hashes_by_category: dict[str, list[str]] = {category: [] for category in CATEGORIES}

    for source in sources:
        for category in CATEGORIES:
            category_dir = source.category_dir(category)
            selected_path = category_dir / "selected_cases.csv"
            pred_dir = source.pred_dir(category)
            rows_all = read_csv_rows(selected_path)
            rows = rows_all[:limit] if limit is not None else rows_all
            pred_count = len(list(pred_dir.glob("*_pred.npy")))
            selected_hash = selected_case_fingerprint(rows)
            hashes_by_category[category].append(selected_hash)

            invalid_count = 0
            for index, row in enumerate(rows):
                pred_path = pred_path_for_row_unchecked(pred_dir, row)
                reason = prediction_file_issue(pred_path)
                if reason is None:
                    continue
                invalid_count += 1
                excluded_indices_by_category[category].add(index)
                skipped.append(
                    {
                        "model": source.model,
                        "category": category,
                        "row_index": index,
                        "name": row["name"],
                        "reason": reason,
                        "path": str(pred_path),
                        "result_root": str(source.result_root),
                    }
                )

            checks.append(
                {
                    "model": source.model,
                    "result_root": str(source.result_root),
                    "category": category,
                    "selected_count": len(rows),
                    "prediction_count": pred_count,
                    "invalid_prediction_count": invalid_count,
                    "common_excluded_count": None,
                    "eval_count": None,
                    "selected_hash": selected_hash,
                    "counts_match": len(rows_all) == pred_count,
                }
            )

            if source == sources[0]:
                reference_rows_by_category[category] = rows
            if len(rows_all) != pred_count:
                raise RuntimeError(
                    f"Count mismatch for {source.model} {category}: "
                    f"selected={len(rows_all)}, predictions={pred_count}"
                )

    for category, hashes in hashes_by_category.items():
        if len(set(hashes)) != 1:
            raise RuntimeError(f"selected_cases.csv differs across model sources for {category}: {hashes}")

    for category in CATEGORIES:
        excluded = excluded_indices_by_category[category]
        rows_by_category[category] = [
            row for index, row in enumerate(reference_rows_by_category[category]) if index not in excluded
        ]
        for check in checks:
            if check["category"] == category:
                check["common_excluded_count"] = len(excluded)
                check["eval_count"] = len(rows_by_category[category])

    return rows_by_category, checks, skipped


def write_report(
    path: Path,
    result_roots: list[Path],
    test_root: Path,
    sources: list[ModelSource],
    summary_rows: list[dict[str, object]],
    rankings: list[dict[str, object]],
    checks: list[dict[str, object]],
    skipped: list[dict[str, object]],
    generation_configs: dict[str, dict[str, object]],
) -> None:
    models = [source.model for source in sources]
    row_by_key = {(str(row["model"]), str(row["category"])): row for row in summary_rows}
    lines: list[str] = [
        "# Combined generation evaluation",
        "",
        f"- Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Result roots: `{'; '.join(str(path) for path in result_roots)}`",
        f"- Test root: `{test_root}`",
        f"- Models: `{', '.join(models)}`",
        "",
        "## Generation configs",
        "",
    ]
    for root, config in generation_configs.items():
        lines.append(
            f"- `{root}`: cfg=`{config.get('cfg_scale', 'unknown')}`, "
            f"steps=`{config.get('num_sampling_steps', 'unknown')}`, "
            f"sampling_hz=`{config.get('sampling_hz', 'unknown')}`"
        )

    lines.extend(["", "## Integrity", ""])
    integrity_rows = []
    for category in CATEGORIES:
        category_checks = [row for row in checks if row["category"] == category]
        selected_count = int(category_checks[0]["selected_count"])
        eval_count = int(category_checks[0]["eval_count"])
        common_excluded = int(category_checks[0]["common_excluded_count"])
        invalid_detail = ", ".join(
            f"{row['model']}={int(row['invalid_prediction_count'])}" for row in category_checks
        )
        integrity_rows.append(
            [
                CATEGORY_LABELS[category],
                selected_count,
                common_excluded,
                eval_count,
                invalid_detail,
            ]
        )
    lines.extend(
        markdown_table(
            ["category", "selected", "common_excluded", "evaluated", "invalid_by_model"],
            integrity_rows,
        )
    )
    if skipped:
        lines.extend(
            [
                "",
                f"Found `{len(skipped)}` empty or missing prediction files. "
                "Any invalid case is excluded for all models within that category.",
            ]
        )

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "- Pointwise R2/RMSE/MAE/PCC are intentionally excluded because stochastic",
            "  conditional generation does not guarantee phase-aligned one-to-one waveforms.",
            "- PGA/PGV/PSA log bias is mean ln(pred / real). Positive means overprediction;",
            "  negative means underprediction.",
            "- PGA/PGV/PSA abs log residual is mean |ln(pred / real)|. Lower is better.",
            "- log-FAS is log Fourier amplitude spectrum MSE from 0.1 to 20 Hz. Lower is better.",
            "- Envelope correlation is zero-lag Pearson correlation of 1 s moving-average",
            "  three-component energy envelopes. Higher is better.",
            "",
            "## Summary tables",
            "",
        ]
    )

    for category in list(CATEGORIES) + ["overall"]:
        rows = []
        for model in models:
            row = row_by_key[(model, category)]
            rows.append(
                [
                    model,
                    int(row["sample_count"]),
                    fmt(row["pga_log_bias_mean"], 4),
                    fmt(row["pga_abs_log_resid_mean"], 4),
                    fmt(row["pgv_log_bias_mean"], 4),
                    fmt(row["pgv_abs_log_resid_mean"], 4),
                    fmt(row["psa_0p2_abs_log_resid_mean"], 4),
                    fmt(row["psa_0p5_abs_log_resid_mean"], 4),
                    fmt(row["psa_1p0_abs_log_resid_mean"], 4),
                    fmt(row["psa_2p0_abs_log_resid_mean"], 4),
                    fmt(row["log_fas_mse_0p1_20hz_mean"], 4),
                    fmt(row["envelope_corr_mean"], 4),
                ]
            )
        lines.append(f"### {CATEGORY_LABELS.get(category, category)}")
        lines.append("")
        lines.extend(
            markdown_table(
                [
                    "model",
                    "N",
                    "PGA bias",
                    "PGA abs",
                    "PGV bias",
                    "PGV abs",
                    "PSA0.2s abs",
                    "PSA0.5s abs",
                    "PSA1s abs",
                    "PSA2s abs",
                    "log-FAS",
                    "Envelope r",
                ],
                rows,
            )
        )
        lines.append("")

    lines.extend(["## Winners", ""])
    for category in list(CATEGORIES) + ["overall"]:
        subset = [row for row in rankings if row["category"] == category and int(row["rank"]) == 1]
        winners = {str(row["metric"]): row for row in subset}
        lines.append(
            "- "
            f"{CATEGORY_LABELS.get(category, category)}: "
            f"PGA={winners['pga_abs_log_resid_mean']['model']} "
            f"({fmt(winners['pga_abs_log_resid_mean']['value'], 4)}), "
            f"PGV={winners['pgv_abs_log_resid_mean']['model']} "
            f"({fmt(winners['pgv_abs_log_resid_mean']['value'], 4)}), "
            f"PSA1s={winners['psa_1p0_abs_log_resid_mean']['model']} "
            f"({fmt(winners['psa_1p0_abs_log_resid_mean']['value'], 4)}), "
            f"log-FAS={winners['log_fas_mse_0p1_20hz_mean']['model']} "
            f"({fmt(winners['log_fas_mse_0p1_20hz_mean']['value'], 4)}), "
            f"Envelope r={winners['envelope_corr_mean']['model']} "
            f"({fmt(winners['envelope_corr_mean']['value'], 4)})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    result_roots = [path.resolve() for path in (args.result_root or list(DEFAULT_RESULT_ROOTS))]
    test_root = args.test_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_model_sources(result_roots)
    rows_by_category, checks, skipped = validate_outputs_multi(sources=sources, limit=args.limit)

    write_csv_rows(
        out_dir / "integrity_checks.csv",
        [
            "model",
            "result_root",
            "category",
            "selected_count",
            "prediction_count",
            "invalid_prediction_count",
            "common_excluded_count",
            "eval_count",
            "selected_hash",
            "counts_match",
        ],
        checks,
    )
    write_csv_rows(
        out_dir / "skipped_predictions.csv",
        ["model", "category", "row_index", "name", "reason", "path", "result_root"],
        skipped,
    )

    summary_rows: list[dict[str, object]] = []
    accumulators_by_model: dict[str, dict[str, MetricAccumulator]] = {
        source.model: {} for source in sources
    }
    for source in sources:
        for category in CATEGORIES:
            rows = rows_by_category[category]
            print(f"Evaluating {source.model} {category}: {len(rows)} samples", flush=True)
            accumulator = evaluate_model_category(
                rows=rows,
                pred_dir=source.pred_dir(category),
                test_root=test_root,
                category=category,
                batch_size=args.batch_size,
                sampling_hz=args.sampling_hz,
                length=args.length,
                progress_name=f"{source.model}/{category}",
            )
            accumulators_by_model[source.model][category] = accumulator
            summary_rows.append(accumulator.to_row(source.model, category))

    for source in sources:
        overall = MetricAccumulator()
        for category in CATEGORIES:
            overall.merge(accumulators_by_model[source.model][category])
        summary_rows.append(overall.to_row(source.model, "overall"))

    write_csv_rows(out_dir / "evaluation_summary.csv", summary_fieldnames(), summary_rows)
    rankings = ranking_rows(summary_rows)
    write_csv_rows(
        out_dir / "metric_rankings.csv",
        ["category", "category_label", "metric", "direction", "rank", "model", "value"],
        rankings,
    )

    generation_configs = {
        str(root): read_json_if_exists(root / "generation_config.json") for root in result_roots
    }
    write_report(
        out_dir / "evaluation_report.md",
        result_roots=result_roots,
        test_root=test_root,
        sources=sources,
        summary_rows=summary_rows,
        rankings=rankings,
        checks=checks,
        skipped=skipped,
        generation_configs=generation_configs,
    )
    print(f"Wrote evaluation outputs to {out_dir}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
