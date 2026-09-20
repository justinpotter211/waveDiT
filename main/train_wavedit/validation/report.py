from __future__ import annotations

from pathlib import Path

import numpy as np

from validation.metrics import FAS_FREQ_MAX, FAS_FREQ_MIN, PSA_PERIODS, compute_psa


DISPLAY_NAMES = {
    "pcc": "PCC",
    "rmse": "RMSE",
    "mae": "MAE",
    "pga_log_resid": "PGA log residual",
    "pgv_log_resid": "PGV log residual",
    "psa_0p2_log_resid": "PSA@0.2s log residual",
    "psa_0p5_log_resid": "PSA@0.5s log residual",
    "psa_1p0_log_resid": "PSA@1.0s log residual",
    "psa_2p0_log_resid": "PSA@2.0s log residual",
    "log_fas_mse_0p1_20hz": "log-FAS MSE (0.1-20 Hz)",
}


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def _mean_fas(wave: np.ndarray, sampling_hz: float) -> tuple[np.ndarray, np.ndarray]:
    centered = wave - wave.mean(axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(wave.shape[0], d=1.0 / sampling_hz)
    mask = (freqs >= FAS_FREQ_MIN) & (freqs <= FAS_FREQ_MAX)
    amp = np.abs(np.fft.rfft(centered, axis=0))
    return freqs[mask], amp[mask, :].mean(axis=1)


def plot_waveform_overlay(
    real: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    sampling_hz: float,
    title: str,
) -> None:
    plt = _import_matplotlib()
    channels = ["E", "N", "Z"]
    time_axis = np.arange(real.shape[0]) / sampling_hz

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for channel_index, axis in enumerate(axes):
        axis.plot(time_axis, real[:, channel_index], label="Real", linewidth=1.2, color="#1f1f1f")
        axis.plot(time_axis, pred[:, channel_index], label="Pred", linewidth=1.0, color="#d94801", alpha=0.85)
        axis.set_ylabel(channels[channel_index])
        axis.grid(True, alpha=0.25)
        if channel_index == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spectrum_psa_compare(
    real: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    sampling_hz: float,
    title: str,
    periods: tuple[float, ...] = PSA_PERIODS,
) -> None:
    plt = _import_matplotlib()

    freqs_real, fas_real = _mean_fas(real, sampling_hz)
    freqs_pred, fas_pred = _mean_fas(pred, sampling_hz)
    psa_real = [compute_psa(real, sampling_hz, period) for period in periods]
    psa_pred = [compute_psa(pred, sampling_hz, period) for period in periods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(freqs_real, fas_real, color="#1f1f1f", linewidth=1.4, label="Real")
    axes[0].plot(freqs_pred, fas_pred, color="#d94801", linewidth=1.2, label="Pred")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Mean FAS")
    axes[0].set_title("Frequency Spectrum")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(periods, psa_real, marker="o", color="#1f1f1f", linewidth=1.4, label="Real")
    axes[1].plot(periods, psa_pred, marker="o", color="#d94801", linewidth=1.2, label="Pred")
    axes[1].set_xlabel("Period (s)")
    axes[1].set_ylabel("PSA")
    axes[1].set_title("5% Damped PSA")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def select_representative_cases(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []

    rmse_sorted = sorted(rows, key=lambda row: (float(row["rmse"]), str(row["name"])))
    rmse_values = np.asarray([float(row["rmse"]) for row in rmse_sorted], dtype=np.float64)
    median_value = float(np.median(rmse_values))
    median_sorted = sorted(
        rows,
        key=lambda row: (abs(float(row["rmse"]) - median_value), float(row["rmse"]), str(row["name"])),
    )
    magnitude_sorted = sorted(
        rows,
        key=lambda row: (float(row["magnitude"]), str(row["name"])),
        reverse=True,
    )
    distance_sorted = sorted(
        rows,
        key=lambda row: (float(row["epicentral_distance_km"]), str(row["name"])),
        reverse=True,
    )

    pools = [
        ("best_rmse", rmse_sorted),
        ("median_rmse", median_sorted),
        ("worst_rmse", list(reversed(rmse_sorted))),
        ("max_magnitude", magnitude_sorted),
        ("max_distance", distance_sorted),
    ]

    used_names: set[str] = set()
    selected: list[dict[str, object]] = []
    for label, pool in pools:
        for row in pool:
            name = str(row["name"])
            if name in used_names:
                continue
            selected.append({"label": label, "row": row})
            used_names.add(name)
            break
    return selected


def _format_float(value: object, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _summary_table_lines(metric_stats: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Metric | Mean | Median | P90 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, stats in metric_stats.items():
        name = DISPLAY_NAMES.get(key, key)
        lines.append(
            f"| {name} | {_format_float(stats['mean'])} | {_format_float(stats['median'])} | {_format_float(stats['p90'])} |"
        )
    return lines


def _binned_table_lines(title: str, rows: list[dict[str, object]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Bucket | Count | RMSE Mean | PCC Mean | PGA Residual Mean | PGV Residual Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['bucket']} | {int(row['count'])} | {_format_float(row['rmse_mean'])} | "
            f"{_format_float(row['pcc_mean'])} | {_format_float(row['pga_log_resid_mean'])} | "
            f"{_format_float(row['pgv_log_resid_mean'])} |"
        )
    if len(rows) == 0:
        lines.append("| no_data | 0 | - | - | - | - |")
    lines.append("")
    return lines


def write_markdown_report(
    output_path: Path,
    summary: dict[str, object],
    magnitude_rows: list[dict[str, object]],
    depth_rows: list[dict[str, object]],
    distance_rows: list[dict[str, object]],
    representative_cases: list[dict[str, object]],
) -> None:
    lines: list[str] = ["# SWaG Evaluation Data Report", ""]
    lines.extend(
        [
            "## Run Info",
            "",
            f"- Model name: `{summary['model_name']}`",
            f"- Checkpoint: `{summary['ckpt_path']}`",
            f"- Test dir: `{summary['test_dir']}`",
            f"- Run dir: `{summary['run_dir']}`",
            f"- Count: `{summary['count']}`",
            f"- Mode: `{'full_test' if summary['full_test'] else 'subset'}`",
            f"- Subset seed: `{summary['subset_seed']}`",
            f"- Device: `{summary['device']}`",
            f"- Precision: `{summary['precision']}`",
            f"- Sampling steps: `{summary['num_sampling_steps']}`",
            f"- CFG scale: `{summary['cfg_scale']}`",
            f"- Sampling Hz: `{summary['sampling_hz']}`",
            f"- Batch size: `{summary['batch_size']}`",
            f"- Elapsed seconds: `{_format_float(summary['elapsed_seconds'], digits=2)}`",
            f"- Variant: `{summary.get('variant', '-')}`",
            f"- Waveform normalization: `{summary.get('waveform_norm_mode', '-')}`",
            f"- Waveform global mean: `{summary.get('waveform_mean_value', '-')}`",
            f"- Waveform global std: `{summary.get('waveform_std_value', '-')}`",
            f"- Waveform normalization source: `{summary.get('waveform_scale_source', '-')}`",
            f"- Waveform normalization stats: `{summary.get('waveform_scale_stats', '-')}`",
            f"- Conditioning source: `{summary.get('conditioning_source', '-')}`",
            f"- Conditioning stats path: `{summary.get('conditioning_stats_path', '-')}`",
            f"- All sample figures: `{summary.get('all_sample_figures', False)}`",
            f"- All sample figure count: `{summary.get('all_sample_figure_count', 0)}`",
            "",
        ]
    )

    warnings = list(summary.get("warnings", []))
    if warnings:
        lines.extend(["## Warnings", ""])
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")

    lines.extend(["## Overall Metrics", ""])
    lines.extend(_summary_table_lines(summary["metric_stats"]))
    lines.append("")

    lines.extend(["## Binned Metrics", ""])
    lines.extend(_binned_table_lines("By magnitude", magnitude_rows))
    lines.extend(_binned_table_lines("By depth", depth_rows))
    lines.extend(_binned_table_lines("By distance", distance_rows))

    lines.extend(["## Representative Figures", ""])
    if representative_cases:
        label_map = {
            "best_rmse": "best_rmse",
            "median_rmse": "median_rmse",
            "worst_rmse": "worst_rmse",
            "max_magnitude": "max_magnitude",
            "max_distance": "max_distance",
        }
        for item in representative_cases:
            row = item["row"]
            lines.extend(
                [
                    f"### {label_map.get(item['label'], item['label'])}",
                    "",
                    f"- Sample: `{row['name']}`",
                    f"- Station: `{row['station']}`",
                    f"- Event: `{row['event_id']}`",
                    f"- Magnitude: `{_format_float(row['magnitude'], digits=2)}`",
                    f"- Depth km: `{_format_float(row['depth_km'], digits=2)}`",
                    f"- Distance km: `{_format_float(row['epicentral_distance_km'], digits=2)}`",
                    f"- RMSE: `{_format_float(row['rmse'])}`",
                    f"- PCC: `{_format_float(row['pcc'])}`",
                    f"- PGA log residual: `{_format_float(row['pga_log_resid'])}`",
                    f"- PGV log residual: `{_format_float(row['pgv_log_resid'])}`",
                    f"- Waveform figure: ![]({item['waveform_relpath']})",
                    f"- Spectrum/PSA figure: ![]({item['spectrum_relpath']})",
                    "",
                ]
            )
    else:
        lines.extend(["No representative figures were generated.", ""])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
