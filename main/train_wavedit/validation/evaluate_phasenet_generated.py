import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[3]


CATEGORIES = ("future_known_station", "future_unknown_station", "historical_unknown_station")
SAMPLE_HZ = 50.0
PHASENET_HZ = 100.0
WINDOW_SIZE = 3000
EPS = 1e-12


class PhaseNetForwardOnly(nn.Module):
    """PhaseNet inference graph matching PhaseNet_Original.pt without SeisBench APIs."""

    def __init__(self, in_channels: int = 3, classes: int = 3) -> None:
        super().__init__()
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8

        self.inc = nn.Conv1d(in_channels, self.filters_root, self.kernel_size, padding="same")
        self.in_bn = nn.BatchNorm1d(self.filters_root, eps=1e-3)
        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                padding = 0 if i in [1, 2, 3] else self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters, filters, self.kernel_size, self.stride, padding=padding, bias=False
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)
            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        for i in range(self.depth - 1):
            filters = int(2 ** (3 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)
            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, classes, 1, padding="same")
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _merge_skip(skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        return torch.cat([skip, x[:, :, offset : offset + skip.shape[-1]]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.in_bn(self.inc(x)))
        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = F.relu(bn1(conv_same(x)))
            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)
                x = F.relu(bn2(conv_down(x)))
        for (conv_up, bn1, conv_same, bn2), skip in zip(self.up_branch, skips[::-1]):
            x = F.relu(bn1(conv_up(x)))
            x = x[:, :, 1:-2]
            x = self._merge_skip(skip, x)
            x = F.relu(bn2(conv_same(x)))
        return self.softmax(self.out(x))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-hoc PhaseNet P/S arrival preservation diagnostics for observed and generated waveforms."
    )
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--model-result-root", type=Path, required=True)
    parser.add_argument("--model-name", default="model_vae")
    parser.add_argument("--phasenet-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-records", type=int, default=64)
    parser.add_argument("--batch-windows", type=int, default=256)
    parser.add_argument("--window-step", type=int, default=500)
    parser.add_argument("--prob-threshold", type=float, default=0.3)
    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)
    return parser


def load_selected_rows(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit <= 0 else rows[:limit]


def load_waveform(path: Path) -> np.ndarray:
    wave = np.load(path)
    if wave.ndim != 2:
        raise ValueError(f"Expected [T,C] waveform in {path}, got shape {wave.shape}.")
    if wave.shape == (3, 3000):
        wave = wave.T
    if wave.shape != (3000, 3):
        raise ValueError(f"Expected waveform shape (3000,3) in {path}, got {wave.shape}.")
    return wave.astype(np.float32, copy=False)


def resample_to_phasenet(wave: np.ndarray) -> np.ndarray:
    old_t = np.arange(wave.shape[0], dtype=np.float64) / SAMPLE_HZ
    new_n = int(round(old_t[-1] * PHASENET_HZ)) + 1
    new_t = np.arange(new_n, dtype=np.float64) / PHASENET_HZ
    out = np.zeros((new_n, wave.shape[1]), dtype=np.float32)
    for channel in range(wave.shape[1]):
        out[:, channel] = np.interp(new_t, old_t, wave[:, channel]).astype(np.float32)
    return out


def window_starts(length: int, step: int) -> list[int]:
    if length <= WINDOW_SIZE:
        return [0]
    starts = list(range(0, length - WINDOW_SIZE + 1, step))
    if starts[-1] != length - WINDOW_SIZE:
        starts.append(length - WINDOW_SIZE)
    return starts


class SlidingPhasePicker:
    def __init__(self, checkpoint: Path, batch_windows: int, window_step: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=self.device)
        self.model = PhaseNetForwardOnly()
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.batch_windows = max(1, int(batch_windows))
        self.window_step = max(1, int(window_step))

    @torch.no_grad()
    def pick_many(self, waveforms: list[np.ndarray]) -> list[dict[str, float]]:
        resampled = [resample_to_phasenet(wave) for wave in waveforms]
        probability = [np.full((3, wave.shape[0]), -np.inf, dtype=np.float32) for wave in resampled]
        windows: list[tuple[int, int, int, np.ndarray]] = []
        for wave_idx, wave in enumerate(resampled):
            for start in window_starts(wave.shape[0], self.window_step):
                valid = min(WINDOW_SIZE, wave.shape[0] - start)
                segment = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
                segment[:valid] = wave[start : start + valid]
                windows.append((wave_idx, start, valid, segment.T))
        for offset in range(0, len(windows), self.batch_windows):
            part = windows[offset : offset + self.batch_windows]
            x = torch.from_numpy(np.stack([entry[3] for entry in part])).float().to(self.device)
            output = self.model(x).detach().cpu().numpy()
            for row, (wave_idx, start, valid, _) in zip(output, part):
                target = probability[wave_idx][:, start : start + valid]
                probability[wave_idx][:, start : start + valid] = np.maximum(target, row[:, :valid])
        picks: list[dict[str, float]] = []
        for scores in probability:
            p_idx = int(np.nanargmax(scores[1]))
            s_idx = int(np.nanargmax(scores[2]))
            picks.append(
                {
                    "p_pick_s": p_idx / PHASENET_HZ,
                    "p_prob": float(scores[1, p_idx]),
                    "s_pick_s": s_idx / PHASENET_HZ,
                    "s_prob": float(scores[2, s_idx]),
                }
            )
        return picks


def summarize(rows: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    summaries = []
    for category in [*CATEGORIES, "overall"]:
        selected = rows if category == "overall" else [row for row in rows if row["category"] == category]
        payload: dict[str, object] = {"category": category, "n": len(selected)}
        for phase in ("p", "s"):
            valid = [
                row for row in selected
                if float(row[f"{phase}_prob_observed"]) >= threshold
                and float(row[f"{phase}_prob_generated"]) >= threshold
            ]
            deltas = np.asarray(
                [float(row[f"{phase}_pick_generated_s"]) - float(row[f"{phase}_pick_observed_s"]) for row in valid],
                dtype=float,
            )
            payload[f"{phase}_paired_detected_n"] = len(valid)
            payload[f"{phase}_paired_detected_rate"] = len(valid) / max(1, len(selected))
            payload[f"{phase}_arrival_bias_s"] = float(deltas.mean()) if deltas.size else np.nan
            payload[f"{phase}_arrival_mae_s"] = float(np.abs(deltas).mean()) if deltas.size else np.nan
            payload[f"{phase}_arrival_sigma_s"] = float(deltas.std(ddof=0)) if deltas.size else np.nan
        summaries.append(payload)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_record_path = args.out_dir / "phasenet_arrival_per_record.csv"
    if per_record_path.exists() and not args.resume:
        raise FileExistsError(
            f"Per-record result already exists: {per_record_path}. "
            "Use --resume to continue it or choose a new output directory."
        )
    output_rows: list[dict[str, object]] = read_csv(per_record_path) if args.resume else []
    completed = {(str(row["category"]), str(row["name"])) for row in output_rows}
    picker = SlidingPhasePicker(args.phasenet_checkpoint, args.batch_windows, args.window_step)
    selected_total = 0
    for category in CATEGORIES:
        category_dir = args.model_result_root / "models" / args.model_name / "full_test" / category
        selected_rows = load_selected_rows(category_dir / "selected_cases.csv", args.limit_per_category)
        selected_total += len(selected_rows)
        pending_rows = [
            row for row in selected_rows if (category, str(row["name"])) not in completed
        ]
        processed_before = len(selected_rows) - len(pending_rows)
        if processed_before:
            print(f"{category}: resume found {processed_before}/{len(selected_rows)} completed records", flush=True)
        for start in range(0, len(pending_rows), max(1, args.batch_records)):
            batch = pending_rows[start : start + args.batch_records]
            observed = [load_waveform(args.test_root / category / f"{row['name']}.npy") for row in batch]
            generated = [load_waveform(category_dir / "pred_npy" / f"{row['name']}_pred.npy") for row in batch]
            obs_picks = picker.pick_many(observed)
            gen_picks = picker.pick_many(generated)
            new_rows: list[dict[str, object]] = []
            for row, obs, gen in zip(batch, obs_picks, gen_picks):
                new_rows.append(
                    {
                        "category": category,
                        "name": row["name"],
                        "station": row.get("station", ""),
                        "event_id": row.get("event_id", ""),
                        "magnitude": row.get("magnitude", ""),
                        "epicentral_distance_km": row.get("epicentral_distance_km", ""),
                        "p_pick_observed_s": obs["p_pick_s"],
                        "p_prob_observed": obs["p_prob"],
                        "p_pick_generated_s": gen["p_pick_s"],
                        "p_prob_generated": gen["p_prob"],
                        "p_delta_s": gen["p_pick_s"] - obs["p_pick_s"],
                        "s_pick_observed_s": obs["s_pick_s"],
                        "s_prob_observed": obs["s_prob"],
                        "s_pick_generated_s": gen["s_pick_s"],
                        "s_prob_generated": gen["s_prob"],
                        "s_delta_s": gen["s_pick_s"] - obs["s_pick_s"],
                    }
                )
            append_csv(per_record_path, new_rows)
            output_rows.extend(new_rows)
            completed.update((category, str(row["name"])) for row in batch)
            processed = processed_before + start + len(batch)
            print(f"{category}: PhaseNet processed {processed}/{len(selected_rows)}", flush=True)
    summary = summarize(output_rows, args.prob_threshold)
    write_csv(args.out_dir / "phasenet_arrival_summary.csv", summary)
    (args.out_dir / "phasenet_arrival_manifest.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "phasenet_checkpoint": str(args.phasenet_checkpoint),
                "input_sampling_hz": SAMPLE_HZ,
                "phasenet_sampling_hz": PHASENET_HZ,
                "window_size_points_at_100hz": WINDOW_SIZE,
                "window_step_points_at_100hz": args.window_step,
                "probability_threshold": args.prob_threshold,
                "selected_records_requested": selected_total,
                "evaluated_records": len(output_rows),
                "complete": len(output_rows) == selected_total,
                "resume_enabled": args.resume,
                "interpretation": (
                    "Post-hoc phase-arrival preservation diagnostic. Observed records were originally "
                    "windowed using PhaseNet P picks, so this is not an independent manual-pick accuracy test."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
