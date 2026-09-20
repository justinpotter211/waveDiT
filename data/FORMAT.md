# Waveform data format

This repository ships **no waveform data**. This document describes the layout and
conventions the code expects so you can prepare your own.

## Directory layout

```
data/
├── train/       # training waveforms
├── val/         # validation waveforms
├── test/        # test waveforms, split into three category subdirectories
│   ├── future_known_station/
│   ├── future_unknown_station/
│   └── historical_unknown_station/
└── dataset_runtime_stats.json   # required by every entry point
```

`train/`, `val/`, and `test/historical_unknown_station/` are flat directories of `.npy`
files. The categories are described under [Test categories](#test-categories).

All three of these paths are overridable: `--wf_dir` / `--val_dir` for training,
`--test-dir` / `--test-root` for evaluation, `--input-dir` for generation.

## File format

Each `.npy` file holds one three-component record:

| Property | Value |
| --- | --- |
| Shape on disk | `(3000, 3)` — samples × components |
| Shape fed to the model | `(3, 3000)` (transposed by the loader) |
| dtype | float32 |
| Sampling rate | 50 Hz |
| Duration | 60 s |

The loader validates the on-disk shape and raises if it does not match.

## Filename grammar

The first six underscore-separated fields carry the conditioning labels. The remaining
fields identify the record.

```
sta_lat_sta_lon_src_lat_src_lon_dep_mag_event_id_station_id.npy
```

Example:

```
24.3413_124.1526_23.3200_123.6930_54.00_6.50_20100207151000_OKN009.npy
```

| Field | Value in the example | Meaning |
| --- | --- | --- |
| `sta_lat` | 24.3413 | station latitude (deg) |
| `sta_lon` | 124.1526 | station longitude (deg) |
| `src_lat` | 23.3200 | source latitude (deg) |
| `src_lon` | 123.6930 | source longitude (deg) |
| `dep` | 54.00 | hypocentral depth (km) |
| `mag` | 6.50 | magnitude |
| `event_id` | 20100207151000 | origin time, `YYYYMMDDHHMMSS` |
| `station_id` | OKN009 | station code |

Only the first six fields are read by `LoadWaveform`. Filenames with fewer than six
underscore-separated fields are rejected.

The epicentral distance and azimuth used by the `relative_geometry` conditioner are
derived from the station and source coordinates, not read from the filename.

## Conditioning ranges

Labels are linearly rescaled from their configured range into `[min_get, max_get]`
(default `[0, 1000]`) before entering the model. Values outside the configured range raise
an error rather than being silently clamped.

Ranges come from `conditioning_defaults` in `data/dataset_runtime_stats.json`:

| Label | Min | Max |
| --- | ---: | ---: |
| `sta_lat` | 18.0 | 55.0 |
| `sta_lon` | 122.0 | 156.0 |
| `src_lat` | 18.0 | 55.0 |
| `src_lon` | 122.0 | 156.0 |
| `dep` | -100.0 | 800.0 |
| `mag` | -1.0 | 10.0 |
| `epi_dist` | -100.0 | 2500.0 |

Override any of them with the matching CLI flags (`--sta-lat-min`, `--mag-max`, …). If
your dataset covers a different region or magnitude span, these **must** be widened,
otherwise loading fails on out-of-range records.

## Normalization

Waveforms use a single global Z-score computed over the training set:

```
z = (x - waveform_mean_value) / waveform_std_value
x = z * waveform_std_value + waveform_mean_value      # denormalization
```

Shipped values in `data/dataset_runtime_stats.json`:

| Field | Value |
| --- | --- |
| `waveform_norm_mode` | `global_zscore` |
| `waveform_mean_value` | `-3.941020839870379e-08` |
| `waveform_std_value` | `1.4593783015381432` |
| `waveform_sample_count` | 349126 |

Training writes the values it used into `waveform_scale_config.json` next to the
checkpoint, and inference reuses them so generated waveforms are rescaled correctly.

## Splits

The reference split is a strict 80 / 10 / 10 partition, with future-year events held out
into the test set:

| Split | Records |
| --- | ---: |
| train | 349,126 |
| val | 43,641 |
| test | 43,641 |
| total | 436,408 |

### Test categories

The test split is grouped by how far the evaluation conditions are from training, so
generalization can be reported separately from interpolation:

| Category | Records | Meaning |
| --- | ---: | --- |
| `historical_unknown_station` | 34,270 | Historical events at stations not seen in training |
| `future_known_station` | 8,372 | Future events at stations seen in training |
| `future_unknown_station` | 999 | Future events at stations not seen in training |

`run_eval_by_category.py` evaluates each category and aggregates the results.

## Long-tail cells

Long-tail re-weighting and macro-averaged reporting both use a 9 × 9 grid of joint
magnitude × distance cells (`TOTAL_JOINT_CELLS = 81`).

| Axis | Bin edges |
| --- | --- |
| magnitude | 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 9.1 |
| distance (km) | 0, 10, 30, 60, 100, 200, 400, 800, 1600, 2500 |

`data/long_tail_weights_train.npz` holds the precomputed per-cell weights for the
reference training split. Regenerate it with `main/train_wavedit/long_tail_weighting.py`
if your data differs.

Note that reporting-quality bins in `validation/subset.py` use a coarser 5 × 5 × 5
magnitude / depth / distance scheme — these are for readable tables and are separate from
the 81 weight cells.
