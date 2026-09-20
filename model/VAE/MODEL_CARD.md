# VAE

- Checkpoint: `VAE.pt`
- SHA256: `c935031d84d6f7202bd501128abe55d709105a0d1e3199ab2f44954638c52bab`
- Size: 59,333,899 bytes
- Input waveform: 3×3000 at 50 Hz
- Latent: 32×125, total compression R24
- Encoder strides: [2,2,2,3]
- Decoder strides: [3,2,2,2]
- Decoder modes: four stages all `pixel_shuffle`
- Training: initialized from an earlier VAE of the same family; encoder frozen;
  decoder-only 8,000 optimizer steps; no anti-alias or comb penalty.
- Public `model` weights are EMA weights selected by validation.

Verified contract with the original VAE:

- all 54 encoder tensors are bitwise equal;
- latent scale is bitwise equal;
- one held-out waveform produces a bitwise-equal posterior mean;
- the Pixel Shuffle decoded result is finite and 3×3000.

Use the loader in `main/train_wavedit/waveform_vae.py`.
