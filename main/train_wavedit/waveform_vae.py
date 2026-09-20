import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PortablePathUnpickler(pickle.Unpickler):
    """Load concrete pathlib paths as paths native to the current OS."""

    def find_class(self, module: str, name: str):
        if module == "pathlib" and name in {"WindowsPath", "PosixPath"}:
            return Path
        return super().find_class(module, name)


class _PortablePathPickleModule:
    """Minimal pickle-module interface required by torch.load."""

    __name__ = "pickle"
    Unpickler = _PortablePathUnpickler
    load = staticmethod(pickle.load)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.stride = int(stride)
        if self.stride % 2 == 0:
            kernel_size = 2 * self.stride
            padding = self.stride // 2
        else:
            kernel_size = 2 * self.stride - 1
            padding = (self.stride - 1) // 2
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class BlurPool1D(nn.Module):
    """Fixed binomial low-pass filter applied independently to each channel."""

    def __init__(self, channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")
        coefficients = torch.tensor(
            [math.comb(kernel_size - 1, index) for index in range(kernel_size)],
            dtype=torch.float32,
        )
        coefficients = coefficients / coefficients.sum()
        self.channels = int(channels)
        self.kernel_size = kernel_size
        self.stride = int(stride)
        self.register_buffer("kernel", coefficients.view(1, 1, -1).repeat(self.channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        mode = "reflect" if x.shape[-1] > pad else "replicate"
        x = F.pad(x, (pad, pad), mode=mode)
        return F.conv1d(x, self.kernel.to(dtype=x.dtype), stride=self.stride, groups=self.channels)


class Upsample1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        mode: str = "linear",
        blur_kernel_size: int = 5,
        blur_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.stride = int(stride)
        self.mode = str(mode)
        if self.mode not in {"linear", "anti_alias", "blended_anti_alias"}:
            raise ValueError(f"Unsupported upsample mode: {self.mode}")
        if blur_alpha is None:
            blur_alpha = 1.0 if self.mode == "anti_alias" else 0.0
        self.blur_alpha = float(blur_alpha)
        if not 0.0 <= self.blur_alpha <= 1.0:
            raise ValueError(f"blur_alpha must be in [0, 1], got {self.blur_alpha}")
        self.blur = (
            BlurPool1D(in_channels, kernel_size=blur_kernel_size, stride=1)
            if self.blur_alpha > 0.0
            else None
        )
        self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.stride, mode="linear", align_corners=False)
        if self.blur is not None:
            blurred = self.blur(x)
            x = (1.0 - self.blur_alpha) * x + self.blur_alpha * blurred
        return self.proj(x)


class PixelShuffle1D(nn.Module):
    """Interleave phase channels: (B, C*r, L) -> (B, C, L*r)."""

    def __init__(self, scale: int):
        super().__init__()
        if not isinstance(scale, int) or isinstance(scale, bool) or scale < 1:
            raise ValueError("scale must be a positive integer")
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] % self.scale:
            raise ValueError("PixelShuffle1D requires (B, C*scale, L)")
        batch, channels, length = x.shape
        return x.reshape(batch, channels // self.scale, self.scale, length).permute(
            0, 1, 3, 2
        ).reshape(batch, channels // self.scale, length * self.scale)


class SubpixelUpsample1D(nn.Module):
    """ICNR initialization ties phase kernels initially, not during training."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = int(stride)
        self.proj = nn.Conv1d(in_channels, out_channels * self.stride, 3, padding=1)
        self.shuffle = PixelShuffle1D(self.stride)
        with torch.no_grad():
            base = torch.empty(out_channels, in_channels, 3)
            nn.init.kaiming_uniform_(base, a=math.sqrt(5))
            self.proj.weight.copy_(base.repeat_interleave(self.stride, dim=0))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


class PhaseBalancedTransposeUpsample1D(nn.Module):
    """Stride-divisible kernel with equal phase sums at initialization.

    For odd stride, k=2*s and p=s//2 produce L*s+1: crop that extra
    rightmost sample explicitly. This is not a claim of permanent antialiasing.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = int(stride)
        self.proj = nn.ConvTranspose1d(in_channels, out_channels, 2 * self.stride,
                                      stride=self.stride, padding=self.stride // 2)
        with torch.no_grad():
            base = torch.empty(in_channels, out_channels, 2)
            nn.init.kaiming_uniform_(base, a=math.sqrt(5))
            self.proj.weight.copy_(base.repeat_interleave(self.stride, dim=2))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)[..., :x.shape[-1] * self.stride]


class WaveformKLVAE1D(nn.Module):
    """
    KL autoencoder for normalized 3-component waveforms.

    Default shape: (B, 3, 3000) -> z/mu/logvar (B, 8, 375) -> recon (B, 3, 3000).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 8,
        input_length: int = 3000,
        base_channels: int = 64,
        strides: Optional[Tuple[int, ...]] = None,
        upsample_mode: str = "linear",
        blur_kernel_size: int = 5,
        upsample_blur_alphas: Optional[Tuple[float, ...]] = None,
        pad_to_multiple: bool = False,
        pad_mode: str = "reflect",
        decoder_upsample_modes: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__()
        strides = tuple(int(s) for s in (strides or (2, 2, 2)))
        if not strides or any(s < 1 for s in strides):
            raise ValueError(f"strides must contain positive integers, got {strides}")
        downsample_factor = 1
        for stride in strides:
            downsample_factor *= stride
        padded_input_length = int(input_length)
        if input_length % downsample_factor != 0 and not pad_to_multiple:
            raise ValueError(
                f"input_length must be divisible by downsample_factor={downsample_factor}, got {input_length}"
            )
        if input_length % downsample_factor != 0:
            padded_input_length = int(math.ceil(input_length / downsample_factor) * downsample_factor)
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.input_length = input_length
        self.padded_input_length = padded_input_length
        self.pad_to_multiple = bool(pad_to_multiple)
        self.pad_mode = str(pad_mode)
        if self.pad_mode not in {"reflect", "replicate", "constant"}:
            raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")
        total_padding = self.padded_input_length - self.input_length
        self.pad_left = total_padding // 2
        self.pad_right = total_padding - self.pad_left
        self.strides = strides
        self.downsample_factor = downsample_factor
        self.latent_length = padded_input_length // downsample_factor
        self.base_channels = base_channels
        self.upsample_mode = str(upsample_mode)
        self.blur_kernel_size = int(blur_kernel_size)
        if upsample_blur_alphas is None:
            default_alpha = 1.0 if self.upsample_mode == "anti_alias" else 0.0
            upsample_blur_alphas = tuple(default_alpha for _ in strides)
        self.upsample_blur_alphas = tuple(float(alpha) for alpha in upsample_blur_alphas)
        if len(self.upsample_blur_alphas) != len(strides):
            raise ValueError(
                "upsample_blur_alphas is in decoder order and must have one value per stage: "
                f"expected {len(strides)}, got {len(self.upsample_blur_alphas)}"
            )
        if any(alpha < 0.0 or alpha > 1.0 for alpha in self.upsample_blur_alphas):
            raise ValueError(f"upsample_blur_alphas must be in [0, 1], got {self.upsample_blur_alphas}")
        self.decoder_upsample_modes = tuple(decoder_upsample_modes or (self.upsample_mode,) * len(strides))
        if len(self.decoder_upsample_modes) != len(strides):
            raise ValueError("decoder_upsample_modes must have one entry per decoder stage")
        if any(mode not in {"linear", "anti_alias", "blended_anti_alias", "pixel_shuffle", "transpose"}
               for mode in self.decoder_upsample_modes):
            raise ValueError(f"Unsupported decoder stage modes: {self.decoder_upsample_modes}")
        if any(mode in {"pixel_shuffle", "transpose"} and alpha != 0 for mode, alpha in
               zip(self.decoder_upsample_modes, self.upsample_blur_alphas)):
            raise ValueError("Learnable upsampling experiments do not support blur")

        channels = [base_channels * (i + 1) for i in range(len(strides) + 1)]
        encoder_layers = [nn.Conv1d(in_channels, channels[0], kernel_size=7, padding=3), ResidualBlock1D(channels[0])]
        for i, stride in enumerate(strides):
            encoder_layers.extend(
                [
                    Downsample1D(channels[i], channels[i + 1], stride=stride),
                    ResidualBlock1D(channels[i + 1]),
                ]
            )
        encoder_layers.extend(
            [
                nn.GroupNorm(_groups(channels[-1]), channels[-1]),
                nn.SiLU(),
                nn.Conv1d(channels[-1], latent_channels * 2, kernel_size=3, padding=1),
            ]
        )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [nn.Conv1d(latent_channels, channels[-1], kernel_size=3, padding=1), ResidualBlock1D(channels[-1])]
        for decoder_stage, (i, stride) in enumerate(reversed(list(enumerate(strides)))):
            stage_mode = self.decoder_upsample_modes[decoder_stage]
            upsample = (
                SubpixelUpsample1D(channels[i + 1], channels[i], stride)
                if stage_mode == "pixel_shuffle"
                else PhaseBalancedTransposeUpsample1D(channels[i + 1], channels[i], stride)
                if stage_mode == "transpose"
                else Upsample1D(channels[i + 1], channels[i], stride=stride,
                                mode=stage_mode, blur_kernel_size=self.blur_kernel_size,
                                blur_alpha=self.upsample_blur_alphas[decoder_stage])
            )
            decoder_layers.extend(
                [
                    upsample,
                    ResidualBlock1D(channels[i]),
                ]
            )
        decoder_layers.extend(
            [
                nn.GroupNorm(_groups(channels[0]), channels[0]),
                nn.SiLU(),
                nn.Conv1d(channels[0], in_channels, kernel_size=7, padding=3),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def config(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "input_length": self.input_length,
            "latent_length": self.latent_length,
            "padded_input_length": self.padded_input_length,
            "base_channels": self.base_channels,
            "downsample_factor": self.downsample_factor,
            "strides": list(self.strides),
            "upsample_mode": self.upsample_mode,
            "blur_kernel_size": self.blur_kernel_size,
            "upsample_blur_alphas": list(self.upsample_blur_alphas),
            "pad_to_multiple": self.pad_to_multiple,
            "pad_mode": self.pad_mode,
            "decoder_upsample_modes": list(self.decoder_upsample_modes),
        }

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pad_left or self.pad_right:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.pad_mode)
        moments = self.encoder(x)
        mu, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_to_latent(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar) if sample else mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(z)
        if self.pad_left or self.pad_right:
            recon = recon[..., self.pad_left : self.pad_left + self.input_length]
        elif recon.shape[-1] != self.input_length:
            recon = recon[..., : self.input_length]
            if recon.shape[-1] < self.input_length:
                recon = F.pad(recon, (0, self.input_length - recon.shape[-1]))
        return recon

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decode(z)
        return {"recon": recon, "z": z, "mu": mu, "logvar": logvar}


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def latent_scale_from_checkpoint(payload: Dict[str, Any], latent_channels: Optional[int] = None) -> torch.Tensor:
    scale = payload.get("latent_scale", None)
    if scale is None and isinstance(payload.get("vae", None), dict):
        scale = payload["vae"].get("latent_scale", None)
    if scale is None:
        args = payload.get("args", None)
        if isinstance(args, dict):
            latent_channels = latent_channels or int(args.get("latent_channels", 8))
        elif args is not None and hasattr(args, "latent_channels"):
            latent_channels = latent_channels or int(args.latent_channels)
        latent_channels = latent_channels or 8
        return torch.ones(latent_channels, dtype=torch.float32)
    if isinstance(scale, torch.Tensor):
        scale_t = scale.detach().float().flatten()
    else:
        scale_t = torch.tensor(scale, dtype=torch.float32).flatten()
    scale_t = torch.where(torch.isfinite(scale_t) & (scale_t > 0), scale_t, torch.ones_like(scale_t))
    return scale_t


def latent_scale_view(scale: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return scale.to(device=device, dtype=dtype).view(1, -1, 1)


def _args_value(args: Any, name: str, default: Any) -> Any:
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def vae_config_from_checkpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("vae_config", None), dict):
        cfg = dict(payload["vae_config"])
    elif isinstance(payload.get("config", None), dict):
        cfg = dict(payload["config"])
    else:
        args = payload.get("args", {})
        cfg = {
            "in_channels": int(_args_value(args, "in_channels", 3)),
            "latent_channels": int(_args_value(args, "latent_channels", 8)),
            "input_length": int(_args_value(args, "length", _args_value(args, "input_length", 3000))),
            "base_channels": int(_args_value(args, "base_channels", 64)),
            "strides": _args_value(args, "strides", (2, 2, 2)),
            "upsample_mode": _args_value(args, "upsample_mode", "linear"),
            "blur_kernel_size": int(_args_value(args, "blur_kernel_size", 5)),
            "upsample_blur_alphas": _args_value(args, "upsample_blur_alphas", None),
            "pad_to_multiple": bool(_args_value(args, "pad_to_multiple", False)),
            "pad_mode": _args_value(args, "pad_mode", "reflect"),
        }
    cfg.setdefault("in_channels", 3)
    cfg.setdefault("latent_channels", 8)
    cfg.setdefault("input_length", 3000)
    cfg.setdefault("base_channels", 64)
    cfg.setdefault("strides", (2, 2, 2))
    cfg.setdefault("upsample_mode", "linear")
    cfg.setdefault("blur_kernel_size", 5)
    cfg.setdefault("upsample_blur_alphas", None)
    cfg.setdefault("pad_to_multiple", False)
    cfg.setdefault("pad_mode", "reflect")
    out = {k: int(v) for k, v in cfg.items() if k in {"in_channels", "latent_channels", "input_length", "base_channels"}}
    out["strides"] = tuple(int(v) for v in cfg.get("strides", (2, 2, 2)))
    out["upsample_mode"] = str(cfg.get("upsample_mode", "linear"))
    out["blur_kernel_size"] = int(cfg.get("blur_kernel_size", 5))
    alphas = cfg.get("upsample_blur_alphas", None)
    out["upsample_blur_alphas"] = None if alphas is None else tuple(float(v) for v in alphas)
    out["pad_to_multiple"] = bool(cfg.get("pad_to_multiple", False))
    out["pad_mode"] = str(cfg.get("pad_mode", "reflect"))
    modes = cfg.get("decoder_upsample_modes")
    out["decoder_upsample_modes"] = None if modes is None else tuple(str(mode) for mode in modes)
    return out


def load_waveform_vae_checkpoint(
    ckpt_path: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    eval_mode: bool = True,
) -> Tuple[WaveformKLVAE1D, Dict[str, Any]]:
    device = device or torch.device("cpu")
    payload = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_PortablePathPickleModule,
    )
    cfg = vae_config_from_checkpoint(payload)
    model = WaveformKLVAE1D(**cfg)
    state = payload.get("model", payload.get("state_dict", payload))
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Failed to load VAE checkpoint cleanly: missing={missing}, unexpected={unexpected}")
    model.to(device=device, dtype=dtype)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, payload


def compute_latent_channel_std(
    model: WaveformKLVAE1D,
    loader,
    device: torch.device,
    max_batches: int = 32,
) -> torch.Tensor:
    sums = None
    sq_sums = None
    count = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            x = x.to(device, non_blocking=True).float()
            z = model.encode_to_latent(x, sample=False)
            z = z.detach().float()
            dims = (0, 2)
            batch_sum = z.sum(dim=dims)
            batch_sq_sum = (z * z).sum(dim=dims)
            batch_count = z.shape[0] * z.shape[2]
            sums = batch_sum if sums is None else sums + batch_sum
            sq_sums = batch_sq_sum if sq_sums is None else sq_sums + batch_sq_sum
            count += batch_count
    if count == 0:
        return torch.ones(model.latent_channels, dtype=torch.float32)
    mean = sums / count
    var = torch.clamp(sq_sums / count - mean.pow(2), min=1e-8)
    std = torch.sqrt(var)
    std = torch.where(torch.isfinite(std) & (std > 1e-6), std, torch.ones_like(std))
    return std.cpu()
