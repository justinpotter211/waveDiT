from __future__ import annotations

from collections import defaultdict

import numpy as np


EPS = 1e-6
PSA_PERIODS = (0.2, 0.5, 1.0, 2.0)
FAS_FREQ_MIN = 0.1
FAS_FREQ_MAX = 20.0


def period_key(period: float) -> str:
    return str(period).replace(".", "p")


def psa_residual_keys(periods: tuple[float, ...] = PSA_PERIODS) -> list[str]:
    return [f"psa_{period_key(period)}_log_resid" for period in periods]


SUMMARY_METRIC_KEYS = [
    "pcc",
    "rmse",
    "mae",
    "pga_log_resid",
    "pgv_log_resid",
    *psa_residual_keys(),
    "log_fas_mse_0p1_20hz",
]


def adjust_length(wave: np.ndarray, target_length: int) -> np.ndarray:
    if wave.ndim != 2:
        raise ValueError(f"Expected waveform shape [T, C], got {wave.shape}")
    if wave.shape[0] >= target_length:
        return wave[:target_length, :]
    return np.pad(wave, ((0, target_length - wave.shape[0]), (0, 0)), mode="constant")


def safe_corrcoef(real: np.ndarray, pred: np.ndarray) -> float:
    corr = float(np.corrcoef(real.reshape(-1), pred.reshape(-1))[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def compute_pga(wave: np.ndarray) -> float:
    per_channel_peak = np.max(np.abs(wave), axis=0)
    return float(np.max(per_channel_peak))


def compute_velocity(wave: np.ndarray, sampling_hz: float) -> np.ndarray:
    dt = 1.0 / sampling_hz
    demeaned = wave - wave.mean(axis=0, keepdims=True)
    return np.cumsum(demeaned, axis=0) * dt


def compute_pgv(wave: np.ndarray, sampling_hz: float) -> float:
    velocity = compute_velocity(wave, sampling_hz)
    per_channel_peak = np.max(np.abs(velocity), axis=0)
    return float(np.max(per_channel_peak))


def response_spectrum_max(
    accel: np.ndarray,
    sampling_hz: float,
    period: float,
    damping_ratio: float = 0.05,
) -> float:
    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")

    dt = 1.0 / sampling_hz
    omega = 2.0 * np.pi / period
    stiffness = omega * omega
    damping = 2.0 * damping_ratio * omega
    beta = 0.25
    gamma = 0.5

    disp = 0.0
    vel = 0.0
    acc_rel = 0.0
    max_disp = 0.0

    a0 = 1.0 / (beta * dt * dt) + gamma * damping / (beta * dt)
    a1 = 1.0 / (beta * dt) + damping * (gamma / beta - 1.0)
    a2 = (1.0 / (2.0 * beta) - 1.0) + damping * dt * (gamma / (2.0 * beta) - 1.0)
    keff = stiffness + a0

    for ground_acc in accel:
        p_eff = -float(ground_acc) + a0 * disp + a1 * vel + a2 * acc_rel
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
        max_disp = max(max_disp, abs(disp))

    return float((omega * omega) * max_disp)


def compute_psa(
    wave: np.ndarray,
    sampling_hz: float,
    period: float,
    damping_ratio: float = 0.05,
) -> float:
    values = [
        response_spectrum_max(wave[:, channel], sampling_hz, period, damping_ratio)
        for channel in range(wave.shape[1])
    ]
    return float(max(values))


def compute_log_fas_mse(
    real: np.ndarray,
    pred: np.ndarray,
    sampling_hz: float,
    eps: float = EPS,
) -> float:
    real_centered = real - real.mean(axis=0, keepdims=True)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(real.shape[0], d=1.0 / sampling_hz)
    mask = (freqs >= FAS_FREQ_MIN) & (freqs <= FAS_FREQ_MAX)
    if not np.any(mask):
        return 0.0
    real_amp = np.abs(np.fft.rfft(real_centered, axis=0))[mask, :]
    pred_amp = np.abs(np.fft.rfft(pred_centered, axis=0))[mask, :]
    diff = np.log(real_amp + eps) - np.log(pred_amp + eps)
    return float(np.mean(diff ** 2))


def compute_sample_metrics(
    real: np.ndarray,
    pred: np.ndarray,
    sampling_hz: float,
    periods: tuple[float, ...] = PSA_PERIODS,
    eps: float = EPS,
) -> dict[str, float]:
    err = pred - real
    metrics = {
        "pcc": safe_corrcoef(real, pred),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }

    pga_real = compute_pga(real)
    pga_pred = compute_pga(pred)
    metrics.update(
        {
            "pga_real": pga_real,
            "pga_pred": pga_pred,
            "pga_log_resid": float(np.log((pga_pred + eps) / (pga_real + eps))),
        }
    )

    pgv_real = compute_pgv(real, sampling_hz)
    pgv_pred = compute_pgv(pred, sampling_hz)
    metrics.update(
        {
            "pgv_real": pgv_real,
            "pgv_pred": pgv_pred,
            "pgv_log_resid": float(np.log((pgv_pred + eps) / (pgv_real + eps))),
        }
    )

    for period in periods:
        key = period_key(period)
        real_value = compute_psa(real, sampling_hz, period)
        pred_value = compute_psa(pred, sampling_hz, period)
        metrics[f"psa_{key}_real"] = real_value
        metrics[f"psa_{key}_pred"] = pred_value
        metrics[f"psa_{key}_log_resid"] = float(
            np.log((pred_value + eps) / (real_value + eps))
        )

    metrics["log_fas_mse_0p1_20hz"] = compute_log_fas_mse(real, pred, sampling_hz, eps=eps)
    return metrics


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def summarize_metric_rows(
    rows: list[dict[str, object]],
    metric_keys: list[str] | tuple[str, ...] = tuple(SUMMARY_METRIC_KEYS),
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric_key in metric_keys:
        summary[metric_key] = summarize_values([float(row[metric_key]) for row in rows])
    return summary


def build_binned_rows(
    rows: list[dict[str, object]],
    bucket_key: str,
    ordered_labels: list[str],
    metric_keys: list[str] | tuple[str, ...] = tuple(SUMMARY_METRIC_KEYS),
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[bucket_key])].append(row)

    binned_rows: list[dict[str, object]] = []
    for label in ordered_labels:
        group = grouped.get(label, [])
        if not group:
            continue
        row: dict[str, object] = {"bucket": label, "count": len(group)}
        for metric_key in metric_keys:
            stats = summarize_values([float(item[metric_key]) for item in group])
            row[f"{metric_key}_mean"] = stats["mean"]
            row[f"{metric_key}_median"] = stats["median"]
            row[f"{metric_key}_p90"] = stats["p90"]
        binned_rows.append(row)
    return binned_rows


def assert_finite_rows(
    rows: list[dict[str, object]],
    metric_keys: list[str] | tuple[str, ...] = tuple(SUMMARY_METRIC_KEYS),
) -> None:
    for row in rows:
        for metric_key in metric_keys:
            value = float(row[metric_key])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite metric detected: {row['name']} {metric_key}={value}")
