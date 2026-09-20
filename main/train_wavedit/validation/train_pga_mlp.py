from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EPS = 1.0e-6
FEATURE_COLUMNS = ["station_lat", "station_lon", "source_lat", "source_lon", "depth_km", "magnitude"]
TEST_CATEGORIES = (
    "future_known_station",
    "future_unknown_station",
    "historical_unknown_station",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a six-condition MLP PGA regressor.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target", choices=["all_component_abs", "horizontal_srss"], default="all_component_abs")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-test-records-per-category", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_condition(path: Path) -> list[float]:
    parts = path.stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Cannot parse six conditions from file name: {path.name}")
    return [float(parts[i]) for i in range(6)]


def load_waveform_pga(path: Path, target: str) -> float:
    wave = np.load(path)
    if wave.ndim != 2:
        raise ValueError(f"Expected 2-D waveform array, got {wave.shape} for {path}")
    if wave.shape[0] == 3:
        wave = wave.T
    if wave.shape[1] != 3:
        raise ValueError(f"Expected three components, got {wave.shape} for {path}")
    if target == "all_component_abs":
        return float(np.max(np.abs(wave)))
    horizontal = np.sqrt(np.square(wave[:, 0]) + np.square(wave[:, 1]))
    return float(np.max(horizontal))


def build_cache(input_dir: Path, cache_path: Path, category: str, target: str, limit: int | None) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.npy"))
    if limit is not None:
        files = files[:limit]
    rows: list[dict[str, object]] = []
    t0 = time.time()
    for idx, path in enumerate(files, start=1):
        features = parse_condition(path)
        pga = load_waveform_pga(path, target)
        row: dict[str, object] = {
            "file": str(path),
            "category": category,
            "pga": pga,
            "log_pga": math.log(pga + EPS),
        }
        row.update(dict(zip(FEATURE_COLUMNS, features)))
        rows.append(row)
        if idx % 10000 == 0:
            print(f"[cache] {category}: {idx}/{len(files)} records in {time.time() - t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def load_or_build(
    input_dir: Path,
    cache_dir: Path,
    split_name: str,
    category: str,
    target: str,
    limit: int | None,
    reuse_cache: bool,
) -> pd.DataFrame:
    suffix = f"{split_name}_{category}_{target}"
    if limit is not None:
        suffix += f"_n{limit}"
    cache_path = cache_dir / f"{suffix}.csv"
    if reuse_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    return build_cache(input_dir, cache_path, category, target, limit)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["log_pga"].to_numpy(dtype=np.float32)
    x = (x - mean) / std
    return x.astype(np.float32), y.astype(np.float32)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.float32)


def metrics_for(name: str, category: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    residual = y_pred - y_true
    abs_resid = np.abs(residual)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    ratio = np.exp(residual)
    return {
        "split": name,
        "category": category,
        "n": int(len(y_true)),
        "log_rmse": float(np.sqrt(np.mean(np.square(residual)))) if len(y_true) else float("nan"),
        "log_mae": float(np.mean(abs_resid)) if len(y_true) else float("nan"),
        "abs_log_resid_median": float(np.median(abs_resid)) if len(y_true) else float("nan"),
        "abs_log_resid_p90": float(np.quantile(abs_resid, 0.90)) if len(y_true) else float("nan"),
        "log_bias_mean": float(np.mean(residual)) if len(y_true) else float("nan"),
        "log_corr": corr,
        "pga_ratio_median": float(np.median(ratio)) if len(y_true) else float("nan"),
        "pga_ratio_p90": float(np.quantile(ratio, 0.90)) if len(y_true) else float("nan"),
    }


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "cache"

    train_df = load_or_build(args.train_dir, cache_dir, "train", "train", args.target, args.max_train_records, args.reuse_cache)
    val_df = load_or_build(args.val_dir, cache_dir, "val", "val", args.target, args.max_val_records, args.reuse_cache)
    test_frames = []
    for category in TEST_CATEGORIES:
        test_frames.append(
            load_or_build(
                args.test_root / category,
                cache_dir,
                "test",
                category,
                args.target,
                args.max_test_records_per_category,
                args.reuse_cache,
            )
        )

    x_train_raw = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    x_mean = x_train_raw.mean(axis=0)
    x_std = x_train_raw.std(axis=0)
    x_std[x_std == 0] = 1.0
    x_train, y_train = standardize(train_df, x_mean, x_std)
    x_val, y_val = standardize(val_df, x_mean, x_std)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = MLP(len(FEATURE_COLUMNS), args.hidden_sizes, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = predict(model, x_val, device, args.batch_size)
        val_loss = float(np.mean(np.square(val_pred - y_val)))
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"[train] epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": x_mean,
                    "feature_std": x_std,
                    "feature_columns": FEATURE_COLUMNS,
                    "target": args.target,
                    "hidden_sizes": args.hidden_sizes,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "val_mse": val_loss,
                },
                args.out_dir / "pga_mlp_model.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[train] early stop at epoch {epoch}; best_epoch={best_epoch}", flush=True)
                break

    checkpoint = torch.load(args.out_dir / "pga_mlp_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    summary_rows = []
    prediction_outputs: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] = []
    for split_name, category, df in [("train", "train", train_df), ("val", "val", val_df)]:
        x, y = standardize(df, x_mean, x_std)
        pred = predict(model, x, device, args.batch_size)
        summary_rows.append(metrics_for(split_name, category, y, pred))
        prediction_outputs.append((f"{split_name}_{category}", df, y, pred))

    all_test_y = []
    all_test_pred = []
    for df in test_frames:
        category = str(df["category"].iloc[0])
        x, y = standardize(df, x_mean, x_std)
        pred = predict(model, x, device, args.batch_size)
        summary_rows.append(metrics_for("test", category, y, pred))
        prediction_outputs.append((f"test_{category}", df, y, pred))
        all_test_y.append(y)
        all_test_pred.append(pred)
    if all_test_y:
        summary_rows.append(metrics_for("test", "overall", np.concatenate(all_test_y), np.concatenate(all_test_pred)))

    pd.DataFrame(summary_rows).to_csv(args.out_dir / "pga_mlp_summary.csv", index=False)
    pd.DataFrame(history).to_csv(args.out_dir / "pga_mlp_training_history.csv", index=False)

    if args.save_predictions:
        pred_dir = args.out_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        for name, df, y, pred in prediction_outputs:
            out = df[["file", "category", *FEATURE_COLUMNS, "pga", "log_pga"]].copy()
            out["pred_log_pga"] = pred
            out["pred_pga"] = np.exp(pred) - EPS
            out["log_residual"] = pred - y
            out.to_csv(pred_dir / f"{name}.csv", index=False)

    manifest = {
        "target": args.target,
        "train_records": int(len(train_df)),
        "val_records": int(len(val_df)),
        "test_records": {str(df["category"].iloc[0]): int(len(df)) for df in test_frames},
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": x_mean.tolist(),
        "feature_std": x_std.tolist(),
        "hidden_sizes": args.hidden_sizes,
        "dropout": args.dropout,
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "seed": int(args.seed),
    }
    (args.out_dir / "pga_mlp_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    train_model(parse_args())
