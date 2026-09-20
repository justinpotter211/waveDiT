"""Verify the deployed Pixel Shuffle VAE and its diffusion-facing contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "main/train_wavedit"))
from waveform_vae import load_waveform_vae_checkpoint, latent_scale_from_checkpoint
from waveform_scaling import normalize_waveform_array

EXPECTED_SHA = "c935031d84d6f7202bd501128abe55d709105a0d1e3199ab2f44954638c52bab"
DEPLOYED = ROOT / "model/VAE/VAE.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def verify(
    checkpoint: Path,
    device: torch.device,
    reference_checkpoint: Path | None = None,
    waveform_dir: Path | None = None,
) -> dict:
    if sha(checkpoint) != EXPECTED_SHA:
        raise RuntimeError(f"Unexpected Pixel Shuffle checkpoint hash: {checkpoint}")
    selected, selected_payload = load_waveform_vae_checkpoint(str(checkpoint), device=device)
    config = selected.config()
    if config["decoder_upsample_modes"] != ["pixel_shuffle"] * 4:
        raise RuntimeError(f"Not the selected all-Pixel-Shuffle decoder: {config}")
    if (selected.latent_channels, selected.latent_length, selected.downsample_factor) != (32, 125, 24):
        raise RuntimeError("Unexpected deployed latent shape/compression")
    new_scale = latent_scale_from_checkpoint(selected_payload, 32)
    if not torch.isfinite(new_scale).all() or (new_scale <= 0).any():
        raise RuntimeError("Deployed checkpoint has an invalid latent scale")

    reference_available = reference_checkpoint is not None and reference_checkpoint.is_file()
    original = None
    if reference_available:
        original, original_payload = load_waveform_vae_checkpoint(str(reference_checkpoint), device=device)
        encoder_keys = [key for key in original_payload["model"] if key.startswith("encoder.")]
        if not all(torch.equal(original_payload["model"][key], selected_payload["model"][key]) for key in encoder_keys):
            raise RuntimeError("Deployed encoder differs from the reference VAE")
        old_scale = latent_scale_from_checkpoint(original_payload, 32)
        torch.testing.assert_close(old_scale, new_scale, rtol=0, atol=0)

    waveform_dir = waveform_dir if waveform_dir is not None else ROOT / "data/val"
    samples = sorted(waveform_dir.glob("*.npy")) if waveform_dir.is_dir() else []
    sample_name = None
    decoded_shape = None
    sample_latent_bitwise_equal = None
    if samples:
        stats = json.loads((ROOT / "data/dataset_runtime_stats.json").read_text(encoding="utf-8"))
        sample_name = samples[0].name
        raw = np.load(samples[0]).astype(np.float32, copy=False)
        waveform = normalize_waveform_array(raw, stats["waveform_mean_value"], stats["waveform_std_value"])
        x = torch.from_numpy(waveform).permute(1, 0).unsqueeze(0).to(device)
        new_z = selected.encode_to_latent(x, sample=False)
        if not torch.isfinite(new_z).all():
            raise RuntimeError("Pixel Shuffle encoder produced non-finite latent values")
        if original is not None:
            old_z = original.encode_to_latent(x, sample=False)
            torch.testing.assert_close(old_z, new_z, rtol=0, atol=0)
            sample_latent_bitwise_equal = True
        decoded = selected.decode_from_latent(new_z)
        if decoded.shape != (1, 3, 3000) or not torch.isfinite(decoded).all():
            raise RuntimeError("Pixel Shuffle decoder smoke test failed")
        decoded_shape = list(decoded.shape)
    return {
        "complete": True,
        "checkpoint": str(checkpoint),
        "sha256": EXPECTED_SHA,
        "device": str(device),
        "latent_shape": [32, 125],
        "compression": 24,
        "decoder_upsample_modes": config["decoder_upsample_modes"],
        "reference_checkpoint": str(reference_checkpoint) if reference_checkpoint is not None else None,
        "reference_checkpoint_available": reference_available,
        "encoder_bitwise_equal_to_reference": True if reference_available else None,
        "latent_scale_bitwise_equal_to_reference": True if reference_available else None,
        "sample_latent_bitwise_equal_to_reference": sample_latent_bitwise_equal,
        "waveform_smoke_test": "ran" if samples else "skipped (no .npy under waveform dir)",
        "waveform_dir": str(waveform_dir),
        "decoded_shape": decoded_shape,
        "sample": sample_name,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEPLOYED)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=None,
        help="Optional earlier VAE of the same family. When supplied, the encoder, latent "
        "scale, and posterior mean are cross-checked bitwise. Not shipped with this repo.",
    )
    parser.add_argument(
        "--waveform-dir",
        type=Path,
        default=None,
        help="Directory holding the .npy waveforms for the encode/decode smoke test. "
        "Defaults to data/val; the smoke test is skipped when no waveform is available.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    print(
        json.dumps(
            verify(
                args.checkpoint.resolve(),
                device,
                args.reference_checkpoint.resolve() if args.reference_checkpoint else None,
                waveform_dir=args.waveform_dir.resolve() if args.waveform_dir else None,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
