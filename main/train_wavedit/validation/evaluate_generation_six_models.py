from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_ROOT = REPO_ROOT / "result" / "model_generation_6models_1000steps"
DEFAULT_TEST_ROOT = REPO_ROOT / "data" / "test"
DEFAULT_OUT_DIR = DEFAULT_RESULT_ROOT / "analysis"

CATEGORIES = (
    "future_known_station",
    "future_unknown_station",
    "historical_unknown_station",
)
CATEGORY_LABELS = {
    "future_known_station": "未来事件-已知台站",
    "future_unknown_station": "未来事件-未知台站",
    "historical_unknown_station": "历史事件-未知台站",
    "overall": "总体",
}
PSA_PERIODS = (0.2, 0.5, 1.0, 2.0)
FAS_FREQ_MIN = 0.1
FAS_FREQ_MAX = 20.0
EPS = 1e-6


def period_key(period: float) -> str:
    return str(period).replace(".", "p")


def im_keys() -> list[str]:
    keys = ["pga", "pgv"]
    keys.extend(f"psa_{period_key(period)}" for period in PSA_PERIODS)
    return keys


IM_KEYS = im_keys()


@dataclass
class MetricAccumulator:
    sample_count: int = 0
    log_fas_mse: list[float] = field(default_factory=list)
    envelope_corr: list[float] = field(default_factory=list)
    im_log_resid: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in IM_KEYS}
    )
    im_real: dict[str, list[float]] = field(default_factory=lambda: {key: [] for key in IM_KEYS})
    im_pred: dict[str, list[float]] = field(default_factory=lambda: {key: [] for key in IM_KEYS})

    def extend_waveform(self, real: np.ndarray, pred: np.ndarray) -> None:
        del pred
        self.sample_count += int(real.shape[0])

    def extend_fas(self, values: np.ndarray) -> None:
        self.log_fas_mse.extend(float(value) for value in values)

    def extend_envelope_corr(self, values: np.ndarray) -> None:
        self.envelope_corr.extend(float(value) for value in values)

    def extend_im(self, key: str, real_values: np.ndarray, pred_values: np.ndarray) -> None:
        real_values = np.asarray(real_values, dtype=np.float64)
        pred_values = np.asarray(pred_values, dtype=np.float64)
        log_resid = np.log((pred_values + EPS) / (real_values + EPS))
        log_resid[~np.isfinite(log_resid)] = 0.0
        self.im_log_resid[key].extend(float(value) for value in log_resid)
        self.im_real[key].extend(float(value) for value in real_values)
        self.im_pred[key].extend(float(value) for value in pred_values)

    def merge(self, other: "MetricAccumulator") -> None:
        self.sample_count += other.sample_count
        self.log_fas_mse.extend(other.log_fas_mse)
        self.envelope_corr.extend(other.envelope_corr)
        for key in IM_KEYS:
            self.im_log_resid[key].extend(other.im_log_resid[key])
            self.im_real[key].extend(other.im_real[key])
            self.im_pred[key].extend(other.im_pred[key])

    def to_row(self, model: str, category: str) -> dict[str, object]:
        if self.sample_count <= 0:
            raise ValueError(f"No metrics accumulated for {model} {category}")
        row: dict[str, object] = {
            "model": model,
            "category": category,
            "category_label": CATEGORY_LABELS.get(category, category),
            "sample_count": self.sample_count,
        }
        add_distribution(row, "log_fas_mse_0p1_20hz", self.log_fas_mse)
        add_distribution(row, "envelope_corr", self.envelope_corr)
        for key in IM_KEYS:
            add_im_distribution(row, key, self.im_real[key], self.im_pred[key], self.im_log_resid[key])
        return row


def add_distribution(row: dict[str, object], prefix: str, values: list[float]) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        for suffix in ("mean", "median", "p90"):
            row[f"{prefix}_{suffix}"] = math.nan
        return
    row[f"{prefix}_mean"] = float(np.mean(arr))
    row[f"{prefix}_median"] = float(np.median(arr))
    row[f"{prefix}_p90"] = float(np.percentile(arr, 90))


def add_im_distribution(
    row: dict[str, object],
    key: str,
    real_values: list[float],
    pred_values: list[float],
    log_resid_values: list[float],
) -> None:
    real = np.asarray(real_values, dtype=np.float64)
    pred = np.asarray(pred_values, dtype=np.float64)
    log_resid = np.asarray(log_resid_values, dtype=np.float64)
    abs_log = np.abs(log_resid)
    row[f"{key}_real_mean"] = float(np.mean(real))
    row[f"{key}_pred_mean"] = float(np.mean(pred))
    row[f"{key}_log_bias_mean"] = float(np.mean(log_resid))
    row[f"{key}_log_bias_median"] = float(np.median(log_resid))
    row[f"{key}_abs_log_resid_mean"] = float(np.mean(abs_log))
    row[f"{key}_abs_log_resid_median"] = float(np.median(abs_log))
    row[f"{key}_abs_log_resid_p90"] = float(np.percentile(abs_log, 90))
    row[f"{key}_log_resid_rmse"] = float(np.sqrt(np.mean(log_resid**2)))
    row[f"{key}_ratio_median"] = float(np.median(np.exp(log_resid)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved SWaG generation outputs for multiple models."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sampling-hz", type=float, default=50.0)
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def discover_models(result_root: Path) -> list[str]:
    models_dir = result_root / "models"
    models = [
        path.name
        for path in sorted(models_dir.iterdir())
        if path.is_dir() and (path / "full_test").is_dir()
    ]
    if not models:
        raise RuntimeError(f"No model full_test directories found in {models_dir}")
    return models


def real_path_for_row(row: dict[str, str], test_root: Path, category: str) -> Path:
    original = Path(row["path"])
    if original.exists():
        return original
    candidate = test_root / category / f"{row['name']}.npy"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve real waveform for {row['name']} in {category}")


def pred_path_for_row(pred_dir: Path, row: dict[str, str]) -> Path:
    candidate = pred_dir / f"{row['name']}_pred.npy"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing prediction: {candidate}")
    return candidate


def load_waveform(path: Path, target_length: int) -> np.ndarray:
    wave = np.load(path)
    if wave.ndim != 2 or wave.shape[1] != 3:
        raise ValueError(f"Expected waveform shape [T, 3] at {path}, got {wave.shape}")
    if wave.shape[0] >= target_length:
        wave = wave[:target_length, :]
    else:
        wave = np.pad(wave, ((0, target_length - wave.shape[0]), (0, 0)), mode="constant")
    if not np.isfinite(wave).all():
        raise ValueError(f"Non-finite waveform detected: {path}")
    return wave.astype(np.float32, copy=False)


def load_real_batch(rows: list[dict[str, str]], test_root: Path, category: str, length: int) -> np.ndarray:
    return np.stack([load_waveform(real_path_for_row(row, test_root, category), length) for row in rows], axis=0)


def load_pred_batch(rows: list[dict[str, str]], pred_dir: Path, length: int) -> np.ndarray:
    return np.stack([load_waveform(pred_path_for_row(pred_dir, row), length) for row in rows], axis=0)


def compute_pga_batch(wave: np.ndarray) -> np.ndarray:
    return np.max(np.abs(wave), axis=(1, 2))


def compute_pgv_batch(wave: np.ndarray, sampling_hz: float) -> np.ndarray:
    dt = 1.0 / sampling_hz
    demeaned = wave - wave.mean(axis=1, keepdims=True)
    velocity = np.cumsum(demeaned, axis=1) * dt
    return np.max(np.abs(velocity), axis=(1, 2))


def response_spectrum_batch(
    wave: np.ndarray,
    sampling_hz: float,
    period: float,
    damping_ratio: float = 0.05,
) -> np.ndarray:
    dt = 1.0 / sampling_hz
    omega = 2.0 * np.pi / period
    stiffness = omega * omega
    damping = 2.0 * damping_ratio * omega
    beta = 0.25
    gamma = 0.5

    disp = np.zeros((wave.shape[0], wave.shape[2]), dtype=np.float64)
    vel = np.zeros_like(disp)
    acc_rel = np.zeros_like(disp)
    max_disp = np.zeros_like(disp)

    a0 = 1.0 / (beta * dt * dt) + gamma * damping / (beta * dt)
    a1 = 1.0 / (beta * dt) + damping * (gamma / beta - 1.0)
    a2 = (1.0 / (2.0 * beta) - 1.0) + damping * dt * (gamma / (2.0 * beta) - 1.0)
    keff = stiffness + a0

    wave64 = wave.astype(np.float64, copy=False)
    for index in range(wave64.shape[1]):
        ground_acc = wave64[:, index, :]
        p_eff = -ground_acc + a0 * disp + a1 * vel + a2 * acc_rel
        disp_new = p_eff / keff
        acc_rel_new = (
            (disp_new - disp) / (beta * dt * dt)
            - vel / (beta * dt)
            - (1.0 / (2.0 * beta) - 1.0) * acc_rel
        )
        vel_new = vel + dt * ((1.0 - gamma) * acc_rel + gamma * acc_rel_new)
        disp = disp_new
        vel = vel_new
        acc_rel = acc_rel_new
        max_disp = np.maximum(max_disp, np.abs(disp))

    return np.max((omega * omega) * max_disp, axis=1)


def compute_intensity_measures(wave: np.ndarray, sampling_hz: float) -> dict[str, np.ndarray]:
    values = {
        "pga": compute_pga_batch(wave),
        "pgv": compute_pgv_batch(wave, sampling_hz),
    }
    for period in PSA_PERIODS:
        values[f"psa_{period_key(period)}"] = response_spectrum_batch(wave, sampling_hz, period)
    return values


def log_fas_mse_batch(real: np.ndarray, pred: np.ndarray, sampling_hz: float) -> np.ndarray:
    real_centered = real - real.mean(axis=1, keepdims=True)
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(real.shape[1], d=1.0 / sampling_hz)
    mask = (freqs >= FAS_FREQ_MIN) & (freqs <= FAS_FREQ_MAX)
    real_amp = np.abs(np.fft.rfft(real_centered, axis=1)[:, mask, :])
    pred_amp = np.abs(np.fft.rfft(pred_centered, axis=1)[:, mask, :])
    diff = np.log(real_amp + EPS) - np.log(pred_amp + EPS)
    return np.mean(diff**2, axis=(1, 2))


def smoothed_energy_envelope_batch(wave: np.ndarray, sampling_hz: float) -> np.ndarray:
    """Return 1 s moving-average three-component energy envelopes."""
    envelope = np.sqrt(np.sum(np.square(wave.astype(np.float64, copy=False)), axis=2))
    window = max(1, int(round(sampling_hz)))
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    padded = np.pad(envelope, ((0, 0), (left_pad, right_pad)), mode="constant")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)), mode="constant")
    return (cumulative[:, window:] - cumulative[:, :-window]) / window


def envelope_correlation_batch(real: np.ndarray, pred: np.ndarray, sampling_hz: float) -> np.ndarray:
    real_env = smoothed_energy_envelope_batch(real, sampling_hz)
    pred_env = smoothed_energy_envelope_batch(pred, sampling_hz)
    real_centered = real_env - np.mean(real_env, axis=1, keepdims=True)
    pred_centered = pred_env - np.mean(pred_env, axis=1, keepdims=True)
    numerator = np.sum(real_centered * pred_centered, axis=1)
    denominator = np.sqrt(
        np.sum(real_centered**2, axis=1) * np.sum(pred_centered**2, axis=1)
    )
    correlations = np.zeros(real.shape[0], dtype=np.float64)
    np.divide(numerator, denominator, out=correlations, where=denominator > EPS)
    return correlations


def validate_outputs(
    result_root: Path,
    test_root: Path,
    models: list[str],
    length: int,
    limit: int | None,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, object]], list[dict[str, object]]]:
    del test_root, length
    checks: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    rows_by_category: dict[str, list[dict[str, str]]] = {}
    original_rows_by_category: dict[str, list[dict[str, str]]] = {}
    excluded_indices_by_category: dict[str, set[int]] = {category: set() for category in CATEGORIES}
    hashes_by_category: dict[str, list[str]] = {category: [] for category in CATEGORIES}
    models_dir = result_root / "models"
    for model in models:
        for category in CATEGORIES:
            category_dir = models_dir / model / "full_test" / category
            selected_path = category_dir / "selected_cases.csv"
            pred_dir = category_dir / "pred_npy"
            rows_all = read_csv_rows(selected_path)
            rows = rows_all[:limit] if limit is not None else rows_all
            pred_count = len(list(pred_dir.glob("*_pred.npy")))
            selected_hash = sha256_short(selected_path)
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
                        "model": model,
                        "category": category,
                        "row_index": index,
                        "name": row["name"],
                        "reason": reason,
                        "path": str(pred_path),
                    }
                )
            checks.append(
                {
                    "model": model,
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
            if model == models[0]:
                original_rows_by_category[category] = rows
            if len(rows_all) != pred_count:
                raise RuntimeError(
                    f"Count mismatch for {model} {category}: selected={len(rows_all)}, "
                    f"predictions={pred_count}"
                )
    for category, hashes in hashes_by_category.items():
        if len(set(hashes)) != 1:
            raise RuntimeError(f"selected_cases.csv differs across models for {category}: {hashes}")
    for category in CATEGORIES:
        excluded = excluded_indices_by_category[category]
        rows_by_category[category] = [
            row for index, row in enumerate(original_rows_by_category[category]) if index not in excluded
        ]
        for check in checks:
            if check["category"] == category:
                check["common_excluded_count"] = len(excluded)
                check["eval_count"] = len(rows_by_category[category])
    return rows_by_category, checks, skipped


def pred_path_for_row_unchecked(pred_dir: Path, row: dict[str, str]) -> Path:
    return pred_dir / f"{row['name']}_pred.npy"


def prediction_file_issue(path: Path) -> str | None:
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "empty_file"
    return None


def evaluate_model_category(
    rows: list[dict[str, str]],
    pred_dir: Path,
    test_root: Path,
    category: str,
    batch_size: int,
    sampling_hz: float,
    length: int,
    progress_name: str,
) -> MetricAccumulator:
    accumulator = MetricAccumulator()
    total = len(rows)
    for start in range(0, total, batch_size):
        batch_rows = rows[start : start + batch_size]
        real = load_real_batch(batch_rows, test_root, category, length)
        pred = load_pred_batch(batch_rows, pred_dir, length)
        accumulator.extend_waveform(real, pred)
        accumulator.extend_fas(log_fas_mse_batch(real, pred, sampling_hz))
        accumulator.extend_envelope_corr(envelope_correlation_batch(real, pred, sampling_hz))
        real_im = compute_intensity_measures(real, sampling_hz)
        pred_im = compute_intensity_measures(pred, sampling_hz)
        for key in IM_KEYS:
            accumulator.extend_im(key, real_im[key], pred_im[key])
        done = min(start + batch_size, total)
        if done == total or done % (batch_size * 10) == 0:
            print(f"{progress_name}: {done}/{total}", flush=True)
    return accumulator


def summary_fieldnames() -> list[str]:
    fields = [
        "model",
        "category",
        "category_label",
        "sample_count",
        "log_fas_mse_0p1_20hz_mean",
        "log_fas_mse_0p1_20hz_median",
        "log_fas_mse_0p1_20hz_p90",
        "envelope_corr_mean",
        "envelope_corr_median",
        "envelope_corr_p90",
    ]
    for key in IM_KEYS:
        fields.extend(
            [
                f"{key}_real_mean",
                f"{key}_pred_mean",
                f"{key}_log_bias_mean",
                f"{key}_log_bias_median",
                f"{key}_abs_log_resid_mean",
                f"{key}_abs_log_resid_median",
                f"{key}_abs_log_resid_p90",
                f"{key}_ratio_median",
            ]
        )
    return fields


def ranking_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_specs = [
        ("log_fas_mse_0p1_20hz_mean", "lower"),
        ("envelope_corr_mean", "higher"),
        ("pga_abs_log_resid_mean", "lower"),
        ("pgv_abs_log_resid_mean", "lower"),
        ("psa_0p2_abs_log_resid_mean", "lower"),
        ("psa_0p5_abs_log_resid_mean", "lower"),
        ("psa_1p0_abs_log_resid_mean", "lower"),
        ("psa_2p0_abs_log_resid_mean", "lower"),
    ]
    rows: list[dict[str, object]] = []
    categories = list(CATEGORIES) + ["overall"]
    for category in categories:
        category_rows = [row for row in summary_rows if row["category"] == category]
        for metric, direction in metric_specs:
            reverse = direction == "higher"
            ranked = sorted(category_rows, key=lambda row: float(row[metric]), reverse=reverse)
            for rank, row in enumerate(ranked, start=1):
                rows.append(
                    {
                        "category": category,
                        "category_label": CATEGORY_LABELS.get(category, category),
                        "metric": metric,
                        "direction": direction,
                        "rank": rank,
                        "model": row["model"],
                        "value": row[metric],
                    }
                )
    return rows


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def fmt(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    result_root: Path,
    test_root: Path,
    models: list[str],
    summary_rows: list[dict[str, object]],
    rankings: list[dict[str, object]],
    checks: list[dict[str, object]],
    skipped: list[dict[str, object]],
    generation_config: dict[str, object],
) -> None:
    row_by_key = {(str(row["model"]), str(row["category"])): row for row in summary_rows}
    lines: list[str] = [
        "# 六模型生成测试集地震动指标评估",
        "",
        f"- 生成时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 结果目录: `{result_root}`",
        f"- 测试集目录: `{test_root}`",
        f"- 模型: `{', '.join(models)}`",
        f"- 采样设置: CFG=`{generation_config.get('cfg_scale', 1.5)}`, "
        f"steps=`{generation_config.get('num_sampling_steps', 1000)}`, "
        f"sampling_hz=`{generation_config.get('sampling_hz', 50.0)}`",
        "",
        "## 数据完整性",
        "",
    ]
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
            ["测试集", "原始样本数", "共同剔除", "评估样本数", "各模型无效预测数"],
            integrity_rows,
        )
    )
    if skipped:
        lines.extend(
            [
                "",
                f"发现 `{len(skipped)}` 个空或缺失预测文件。为保证六个模型在同一批样本上比较，凡任一模型无效的样本都在该测试集中共同剔除。",
                "",
            ]
        )
    lines.extend(
        [
        "## 指标定义",
        "",
        "- 已删除逐点 `R2/RMSE/MAE/PCC`：这些指标要求生成波形与唯一真实记录逐点、同相位、同到时对齐，不适合作为一对多条件生成模型的主评价。",
        "- `PGA/PGV/PSA log bias`: `ln(pred / real)` 的均值，正值表示强度整体偏大，负值表示偏小。",
        "- `PGA/PGV/PSA abs log resid`: `|ln(pred / real)|` 的均值，越低越好。",
        "- `log_fas_mse_0p1_20hz`: 0.1-20 Hz 傅里叶幅值谱对数 MSE，越低越好。",
        "- `envelope_corr`: 1 s 平滑三分量能量包络的零时移 Pearson 相关系数，越高越好。",
        "",
        "## 主表",
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

    lines.extend(["## 最优模型摘录", ""])
    for category in list(CATEGORIES) + ["overall"]:
        subset = [row for row in rankings if row["category"] == category and int(row["rank"]) == 1]
        winners = {str(row["metric"]): row for row in subset}
        lines.append(
            "- "
            f"{CATEGORY_LABELS.get(category, category)}: "
            f"PGA 绝对对数残差最优 `{winners['pga_abs_log_resid_mean']['model']}` "
            f"({fmt(winners['pga_abs_log_resid_mean']['value'], 4)}), "
            f"PGV 绝对对数残差最优 `{winners['pgv_abs_log_resid_mean']['model']}` "
            f"({fmt(winners['pgv_abs_log_resid_mean']['value'], 4)}), "
            f"PSA@1s 绝对对数残差最优 `{winners['psa_1p0_abs_log_resid_mean']['model']}` "
            f"({fmt(winners['psa_1p0_abs_log_resid_mean']['value'], 4)}), "
            f"log-FAS 最优 `{winners['log_fas_mse_0p1_20hz_mean']['model']}` "
            f"({fmt(winners['log_fas_mse_0p1_20hz_mean']['value'], 4)}), "
            f"包络相关最优 `{winners['envelope_corr_mean']['model']}` "
            f"({fmt(winners['envelope_corr_mean']['value'], 4)})。"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    result_root = args.result_root.resolve()
    test_root = args.test_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generation_config_path = result_root / "generation_config.json"
    generation_config = read_json(generation_config_path) if generation_config_path.exists() else {}
    models = discover_models(result_root)
    rows_by_category, checks, skipped = validate_outputs(
        result_root=result_root,
        test_root=test_root,
        models=models,
        length=args.length,
        limit=args.limit,
    )
    write_csv_rows(
        out_dir / "integrity_checks.csv",
        [
            "model",
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
        ["model", "category", "row_index", "name", "reason", "path"],
        skipped,
    )

    summary_rows: list[dict[str, object]] = []
    accumulators_by_model: dict[str, dict[str, MetricAccumulator]] = {model: {} for model in models}
    for model in models:
        for category in CATEGORIES:
            rows = rows_by_category[category]
            pred_dir = result_root / "models" / model / "full_test" / category / "pred_npy"
            print(f"Evaluating {model} {category}: {len(rows)} samples", flush=True)
            accumulator = evaluate_model_category(
                rows=rows,
                pred_dir=pred_dir,
                test_root=test_root,
                category=category,
                batch_size=args.batch_size,
                sampling_hz=args.sampling_hz,
                length=args.length,
                progress_name=f"{model}/{category}",
            )
            accumulators_by_model[model][category] = accumulator
            summary_rows.append(accumulator.to_row(model, category))

    for model in models:
        overall = MetricAccumulator()
        for category in CATEGORIES:
            overall.merge(accumulators_by_model[model][category])
        summary_rows.append(overall.to_row(model, "overall"))

    write_csv_rows(out_dir / "evaluation_summary.csv", summary_fieldnames(), summary_rows)
    rankings = ranking_rows(summary_rows)
    write_csv_rows(
        out_dir / "metric_rankings.csv",
        ["category", "category_label", "metric", "direction", "rank", "model", "value"],
        rankings,
    )
    write_report(
        out_dir / "evaluation_report.md",
        result_root=result_root,
        test_root=test_root,
        models=models,
        summary_rows=summary_rows,
        rankings=rankings,
        checks=checks,
        skipped=skipped,
        generation_config=generation_config,
    )
    print(f"Wrote evaluation outputs to {out_dir}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
