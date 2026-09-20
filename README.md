# waveDiT

Conditional latent diffusion transformer for generating three-component ground-motion
waveforms from earthquake and station metadata.

waveDiT pairs a 1D convolutional KL-VAE with a Diffusion Transformer (DiT). The VAE
compresses a `3 × 3000` waveform (three components, 50 Hz, 60 s) into a `32 × 125`
latent, and the DiT performs conditional denoising in that latent space. Conditioning is
a 9-dimensional relative-geometry vector built from the station and source coordinates,
hypocentral depth, magnitude, epicentral distance, and an azimuth sin/cos pair. The model
is trained with min-SNR loss weighting, classifier-free guidance dropout, and long-tail
re-weighting of the magnitude/distance/depth cells.

## Repository layout

```
waveDiT/
├── data/                              # metadata only — no waveform data is shipped
│   ├── dataset_runtime_stats.json     # global Z-score and conditioning ranges
│   ├── long_tail_weights_train.npz    # precomputed long-tail weight manifest
│   ├── Manual.txt                     # source K-NET ASCII format documentation
│   └── FORMAT.md                      # .npy layout, filename grammar, split counts
├── main/train_wavedit/                # all source code
│   ├── train_mainline.py              # training entry point (DDP, EMA, AMP, early stop)
│   ├── generate_waveforms_only.py     # batch inference -> .npy predictions
│   ├── models_mainline.py             # DiT backbone
│   ├── waveform_vae.py                # 1D KL-VAE encoder/decoder + checkpoint loader
│   ├── waveform_scaling.py            # normalization, conditioning ranges, config resolution
│   ├── data_load.py                   # LoadWaveform dataset
│   ├── long_tail_weighting.py         # weight manifest builder + weighted dataset
│   ├── train_compare_waveform_vae.py  # VAE training / configuration comparison
│   ├── train_waveform_vae.py          # single-VAE training wrapper
│   ├── diffusion/                     # Gaussian diffusion + timestep respacing
│   ├── validation/                    # metrics, reports, evaluation drivers
│   └── experiments/vae_pixel_shuffle_deployment/verify_install.py
├── model/                             # see "Released checkpoints" below
│   ├── waveDiT/waveDiT.pt
│   ├── VAE/VAE.pt
│   └── PhaseNet/PhaseNet_Original.pt
└── result/                            # training/inference output root (empty)
```

## Installation

Python 3.10 is the verified interpreter.

```bash
conda create -n wavedit python=3.10
conda activate wavedit
pip install -r requirements.txt
```

`requirements.txt` pins the versions this code was verified against. Install `torch` with
the CUDA wheel matching your driver (`torch==2.7.1+cu128` in the verified environment).

## Data

**No waveform data is distributed with this repository.** Place your own `.npy` files
under `data/train`, `data/val`, and `data/test` using the layout and naming convention
described in [`data/FORMAT.md`](data/FORMAT.md).

Each sample is a `(3000, 3)` float array. The loader reads it as `(3, 3000)` and applies
a global Z-score using the statistics in `data/dataset_runtime_stats.json`. The first six
underscore-separated fields of the filename supply the conditioning labels, which are
rescaled into `[min_get, max_get]` (default `[0, 1000]`).

`data/dataset_runtime_stats.json` is required by every entry point — pass it via
`--waveform-scale-stats` (training) or `--runtime-stats` (inference).

## Released checkpoints

The two large checkpoints are too big for git and are published as a separate download.
Place them at the paths below after downloading — the code and the commands in this
README resolve those exact paths.

| Path | Size | SHA256 | Download |
| --- | ---: | --- | --- |
| `model/waveDiT/waveDiT.pt` | 531 MB | `ae7d61e77b19e5d902e7fc9ef0beef9dcea797e69ea8b75ffca19288df8e56e4` | *link TBD* |
| `model/VAE/VAE.pt` | 56.6 MB | `c935031d84d6f7202bd501128abe55d709105a0d1e3199ab2f44954638c52bab` | *link TBD* |

Verify a download before using it:

```bash
sha256sum model/waveDiT/waveDiT.pt      # macOS: shasum -a 256
sha256sum model/VAE/VAE.pt
```

`model/PhaseNet/PhaseNet_Original.pt` (1.1 MB,
`7f0bd2f8ba5ee387875c044fb1dae28ef7acd2747925a6bab5473d7ce88bb89c`) ships inside the
repository, since it is small.

| Checkpoint | Description |
| --- | --- |
| `model/waveDiT/waveDiT.pt` | Mainline DiT, EMA weights (`ema` key), `mainline` variant |
| `model/VAE/VAE.pt` | All-pixel-shuffle KL-VAE decoder, latent `32 × 125`, compression R24 |
| `model/PhaseNet/PhaseNet_Original.pt` | PhaseNet, used for P/S phase-picking evaluation only |

`model/waveDiT/waveDiT.pt` configuration (stored in the checkpoint itself):

| Field | Value |
| --- | --- |
| `representation` | `klvae_latent` |
| `depth` / `hidden_size` / `num_heads` | 12 / 384 / 6 |
| `patch_size` | 8 (latent patch size 1) |
| `position_embedding` | `sincos` |
| `abs_geo_encoder_type` | `relative_geometry` (`conditioning_schema: coords_relative_geometry_v2`) |
| conditioning dimension | 9 (azimuth as raw sin/cos pair, `pair_mlp` embedder) |
| `prediction_type` | `epsilon` |
| `class_dropout_prob` / `min_snr_gamma` | 0.10 / 5.0 |
| train steps / epoch | 109,040 / 79 |
| best val macro MSE / micro MSE | 0.127726 / 0.132274 |
| parameters | 34,815,808 |

`model/VAE/MODEL_CARD.md` documents the VAE architecture, its SHA256, and how it was
derived from the original non-pixel-shuffle VAE.

## Quick start

### 1. Verify the VAE deployment

Checks the checkpoint hash, the architecture contract, and the latent scale. Runs on CPU.

```bash
python main/train_wavedit/experiments/vae_pixel_shuffle_deployment/verify_install.py --cpu
```

The encode/decode smoke test needs a waveform. Point it at your own data with
`--waveform-dir data/val`; it is skipped automatically when none is found.

### 2. Train

```bash
python main/train_wavedit/train_mainline.py \
  --model waveDiT \
  --representation klvae_latent \
  --autoencoder-ckpt model/VAE/VAE.pt \
  --wf_dir data/train \
  --val_dir data/val \
  --results_dir result \
  --waveform-scale-stats data/dataset_runtime_stats.json \
  --depth 12 --hidden_size 384 --num_heads 6 --patch_size 8 \
  --latent-channels 32 --latent-length 125 --latent-patch-size 1 \
  --position-embedding sincos \
  --abs-geo-encoder-type relative_geometry \
  --min-snr-gamma 5.0 --class-dropout-prob 0.10 \
  --long-tail-weight-manifest data/long_tail_weights_train.npz \
  --long-tail-alpha 0.75 --long-tail-weight-min 0.25 --long-tail-weight-max 4.0 \
  --precision fp32 --global-batch-size 256 --weight-decay 0.01 --min-lr 3e-6
```

For multi-GPU, wrap it in `torchrun` (the script reads `LOCAL_RANK` and initializes NCCL):

```bash
torchrun --nproc_per_node=4 main/train_wavedit/train_mainline.py <same arguments>
```

Runs land in `result/<index>-waveDiT/` with `log.txt`, `waveform_scale_config.json`, and
`checkpoints/`. `--global-batch-size` must be divisible by the world size; the per-GPU
batch size is derived automatically.

To regenerate the long-tail manifest instead of using the shipped one:

```bash
python main/train_wavedit/long_tail_weighting.py \
  --waveform-dir data/train \
  --output data/long_tail_weights_train.npz \
  --alpha 0.75 --weight-min 0.25 --weight-max 4.0
```

### 3. Generate waveforms

`generate_waveforms_only.py` writes predictions as `.npy` with no plots or reports — use
it for full-test generation. `--input-dir` is scanned for `.npy` files; `--full` selects
every record instead of a stratified subset.

```bash
python main/train_wavedit/generate_waveforms_only.py \
  --ckpt model/waveDiT/waveDiT.pt \
  --autoencoder-ckpt model/VAE/VAE.pt \
  --runtime-stats data/dataset_runtime_stats.json \
  --input-dir data/test/historical_unknown_station \
  --out-dir result/generation/waveDiT \
  --full --batch-size 32 \
  --num-sampling-steps 1000 --cfg-scale 1.5 --latent-seed 20260905
```

`--num-shards` / `--shard-index` split a run across GPUs, and `--skip-existing` makes a
shard safe to resume.

`run_baseline_eval.py` (below) renders waveform overlays and response/Fourier spectra for
the cases it evaluates, which covers visual inspection of generated waveforms.

### 4. Evaluate

```bash
python main/train_wavedit/validation/run_baseline_eval.py \
  --ckpt model/waveDiT/waveDiT.pt \
  --autoencoder-ckpt model/VAE/VAE.pt \
  --test-dir data/test \
  --waveform-scale-stats data/dataset_runtime_stats.json \
  --full-test --batch-size 1 \
  --num-sampling-steps 1000 --cfg-scale 1.5
```

Drop `--full-test` for a stratified subset (`--subset-size`, default 256). Add
`--save-pred-npy` to keep predicted waveforms. Outputs land in
`result/validation/<run_id>/`: `selected_cases.csv`, `per_sample_metrics.csv`,
`overall_summary.json`, binned CSVs by magnitude/depth/distance, `figures/`, and
`report.md`.

Related drivers in `validation/`:

| Script | Purpose |
| --- | --- |
| `run_eval_by_category.py` | Evaluate per test category and aggregate |
| `run_model_comparison.py` | Compare several checkpoints, one worker per GPU |
| `evaluate_generation_six_models.py` | Score saved predictions for multiple models |
| `evaluate_generation_combined.py` | Merge scores across several result roots |
| `evaluate_vae_reconstruction_full_test.py` | VAE reconstruction metrics, no diffusion |
| `evaluate_phasenet_generated.py` | Phase picking on generated waveforms (needs `--phasenet-checkpoint`) |
| `train_pga_mlp.py` | MLP baseline predicting PGA from conditioning only |

## Metrics

`validation/metrics.py` computes per-sample time-domain and engineering metrics (PCC,
RMSE, PGA, PGV) plus response spectra and Fourier amplitude spectra. `run_baseline_eval.py`
aggregates them per sample and per magnitude/depth/distance bin, and reports macro and
micro averages over the 81 joint conditioning cells (9 magnitude bins × 9 distance bins)
used by the long-tail weighting.

## Notes and gotchas

- **`--autoencoder-ckpt` must be passed explicitly.** The released `waveDiT.pt` records an
  absolute path from the cluster it was trained on, which will not exist on your machine.
  Every command above passes it. The same applies to any checkpoint you resume from.
- Training data is not shipped, so the training and evaluation commands above need your
  own `data/train`, `data/val`, and `data/test` populated first.
- The released weights are EMA weights, stored under the `ema` key. Inference loads that
  key; checkpoints also carry `model`, `opt`, `scheduler`, and `scaler` for resuming.
- `result/` is an empty output root, kept in git via `.gitkeep`.

## Third-party code

- `main/train_wavedit/diffusion/` — adapted from OpenAI's
  [guided-diffusion](https://github.com/openai/guided-diffusion) and GLIDE (MIT).
- `models_mainline.py` uses `timm`'s `Mlp` (Apache-2.0).

Both are MIT-compatible, so the license below covers the whole repository.

## Authors

| | |
| --- | --- |
| First author | Zhihui Zhu |
| Second author | Yanlong Li |
| Corresponding author | Jiexin Zhong |
| Affiliation | Central South University |

## License

MIT — see [LICENSE](LICENSE).
