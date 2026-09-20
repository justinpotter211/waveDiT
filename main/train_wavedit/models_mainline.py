import math

import numpy as np
import torch
import torch.nn as nn

from waveform_scaling import conditioning_dimension, normalize_abs_geo_encoder_type, uses_relative_geometry

try:
    from timm.models.vision_transformer import Mlp
except ImportError:
    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values detected in {name}.")


def _build_log_frequency_bands(num_frequencies: int, min_freq: float = 1.0, max_freq: float = 64.0) -> torch.Tensor:
    if num_frequencies <= 0:
        raise ValueError(f"num_frequencies must be > 0, got {num_frequencies}.")
    min_value = float(min_freq) * math.pi
    max_value = float(max_freq) * math.pi
    return torch.exp(torch.linspace(math.log(min_value), math.log(max_value), steps=num_frequencies, dtype=torch.float32))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class AngularPairEmbedder(nn.Module):
    """Embed a raw [sin(azimuth), cos(azimuth)] unit-circle pair."""

    def __init__(self, hidden_size, embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, embedding_size, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_size, hidden_size, bias=True),
        )

    def forward(self, angle_pair):
        if angle_pair.ndim != 2 or angle_pair.shape[1] != 2:
            raise ValueError(f"Expected angle pair with shape [B, 2], got {tuple(angle_pair.shape)}.")
        angle_pair = angle_pair.float()
        _assert_finite("mainline_azimuth_pair", angle_pair)
        angle_pair = angle_pair / angle_pair.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return self.mlp(angle_pair)


class CFGMixin:
    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        half_batch = half.shape[0]
        force_drop_mask = torch.cat(
            [
                torch.zeros(half_batch, dtype=torch.bool, device=x.device),
                torch.ones(half_batch, dtype=torch.bool, device=x.device),
            ],
            dim=0,
        )
        model_out = self.forward(combined, t, y, force_drop_mask=force_drop_mask)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


POSITION_EMBEDDING_TYPES = ("sincos", "rope")


def _normalize_position_embedding(value: str) -> str:
    value = str(value).lower()
    if value not in POSITION_EMBEDDING_TYPES:
        raise ValueError(
            f"Unsupported position_embedding '{value}'. Expected one of: {', '.join(POSITION_EMBEDDING_TYPES)}."
        )
    return value


class RotaryPositionEmbedding1D(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even attention head dimension, got {dim}.")
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_pair = x.reshape(*x.shape[:-1], -1, 2)
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]
        x_rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return x_rotated.flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin(q.shape[-2], q.device, q.dtype)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        position_embedding="sincos",
        rope_base=10000.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        position_embedding = _normalize_position_embedding(position_embedding)
        self.rotary = (
            RotaryPositionEmbedding1D(self.head_dim, base=rope_base)
            if position_embedding == "rope"
            else None
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rotary is not None:
            q, k = self.rotary(q, k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """交叉注意力层，用于将辅助台站条件注入 DiT 主路径。

    Query 来自主序列（噪声波形 token），Key/Value 来自辅助台站 token。
    支持注意力掩码以处理变长台站数量。
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, context_mask=None):
        """前向传播。

        Args:
            x: 主序列特征 (B, N, D)，作为 Query。
            context: 辅助台站 token (B, S, D)，作为 Key/Value。
            context_mask: 有效位置掩码 (B, S)，True 表示有效台站。

        Returns:
            融合台站信息后的特征 (B, N, D)。
        """
        B, N, _ = x.shape
        S = context.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if context_mask is not None:
            # context_mask: (B, S) -> (B, 1, 1, S)，屏蔽 padding 台站
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        # 处理全掩码行（无有效台站时 softmax 产生 NaN）
        attn = attn.nan_to_num(nan=0.0)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, position_embedding="sincos", rope_base=10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            position_embedding=position_embedding,
            rope_base=rope_base,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ConvPatchStem1D(nn.Module):
    def __init__(self, signal_len=6000, patch_size=6, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        self.signal_len = int(signal_len)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.num_patches = self.signal_len // self.patch_size
        if self.num_patches * self.patch_size != self.signal_len:
            raise ValueError(
                f"signal_len ({self.signal_len}) must be divisible by patch_size ({self.patch_size})."
            )
        self.pre = nn.Sequential(
            nn.Conv1d(self.in_chans * self.patch_size, embed_dim, kernel_size=1, stride=1, bias=bias),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=bias),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=bias),
            nn.SiLU(),
        )
        self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        if x.shape[-1] != self.signal_len:
            raise ValueError(
                f"Input signal length ({x.shape[-1]}) does not match model ({self.signal_len})."
            )
        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"Input channel count ({x.shape[1]}) does not match model ({self.in_chans})."
            )
        x = x.permute(0, 2, 1).reshape(x.shape[0], self.num_patches, self.in_chans * self.patch_size)
        x = x.permute(0, 2, 1)
        x = self.pre(x)
        x = self.proj(x)
        if x.shape[-1] != self.num_patches:
            raise RuntimeError(
                f"Patch token length mismatch: got {x.shape[-1]}, expected {self.num_patches}."
            )
        return x.permute(0, 2, 1)


class ConfigurableGeoConditioner(nn.Module):
    def __init__(
        self,
        hidden_size,
        abs_geo_encoder_type="none",
        input_min=0.0,
        input_max=1000.0,
        cond_embedding_scale=1.0,
        frequency_embedding_size=256,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_min = float(input_min)
        self.input_max = float(input_max)
        self.cond_embedding_scale = float(cond_embedding_scale)
        self.abs_geo_encoder_type = normalize_abs_geo_encoder_type(abs_geo_encoder_type)
        self.num_condition_features = conditioning_dimension(self.abs_geo_encoder_type)
        self.num_scalar_features = 7 if uses_relative_geometry(self.abs_geo_encoder_type) else self.num_condition_features
        self.scalar_embedders = nn.ModuleList(
            [
                TimestepEmbedder(hidden_size, frequency_embedding_size=frequency_embedding_size)
                for _ in range(self.num_scalar_features)
            ]
        )
        self.angle_embedder = (
            AngularPairEmbedder(hidden_size, embedding_size=frequency_embedding_size)
            if uses_relative_geometry(self.abs_geo_encoder_type)
            else None
        )

    def forward(self, y):
        if y.ndim != 2 or y.shape[1] != self.num_condition_features:
            raise ValueError(
                f"Expected conditioning labels with shape [B, {self.num_condition_features}], got {tuple(y.shape)}."
            )
        y = y.float()
        _assert_finite("mainline_conditioning_labels", y)
        scalar_y = y[:, : self.num_scalar_features] * self.cond_embedding_scale
        conditioning = sum(embedder(scalar_y[:, i]) for i, embedder in enumerate(self.scalar_embedders))
        if self.angle_embedder is not None:
            conditioning = conditioning + self.angle_embedder(y[:, 7:9])
        _assert_finite("mainline_conditioning_vector", conditioning)
        return conditioning


class MainlineTransformerModel(CFGMixin, nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_size=768,
        depth=24,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=False,
        length=6000,
        patch_size=6,
        frequency_embedding_size=256,
        abs_geo_encoder_type="none",
        cond_input_min=0.0,
        cond_input_max=1000.0,
        cond_embedding_scale=1.0,
        position_embedding="sincos",
        rope_base=10000.0,
        **_ignored_kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.length = int(length)
        self.patch_size = int(patch_size)
        self.class_dropout_prob = class_dropout_prob
        self.hidden_size = int(hidden_size)
        self.num_patches = self.length // self.patch_size
        self.position_embedding = _normalize_position_embedding(position_embedding)
        self.rope_base = float(rope_base)
        self.use_abs_pos_embed = self.position_embedding == "sincos"
        if self.num_patches * self.patch_size != self.length:
            raise ValueError(f"length ({length}) must be divisible by patch_size ({patch_size}).")

        self.input_stem = ConvPatchStem1D(
            signal_len=self.length,
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.conditioner = ConfigurableGeoConditioner(
            hidden_size=hidden_size,
            abs_geo_encoder_type=abs_geo_encoder_type,
            input_min=cond_input_min,
            input_max=cond_input_max,
            cond_embedding_scale=cond_embedding_scale,
            frequency_embedding_size=frequency_embedding_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    position_embedding=self.position_embedding,
                    rope_base=self.rope_base,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        w = self.input_stem.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.input_stem.proj.bias is not None:
            nn.init.constant_(self.input_stem.proj.bias, 0)
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1], np.arange(self.num_patches, dtype=np.float32)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        for module in self.modules():
            if isinstance(module, TimestepEmbedder):
                nn.init.normal_(module.mlp[0].weight, std=0.02)
                nn.init.normal_(module.mlp[2].weight, std=0.02)
            elif isinstance(module, AngularPairEmbedder):
                nn.init.normal_(module.mlp[0].weight, std=0.02)
                nn.init.normal_(module.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _build_condition(self, t, y, force_drop_mask=None):
        h_cond = self.conditioner(y)
        if force_drop_mask is None:
            if self.training and self.class_dropout_prob > 0:
                drop_ids = torch.rand(y.shape[0], device=y.device) < self.class_dropout_prob
            else:
                drop_ids = torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        else:
            if force_drop_mask.shape != (y.shape[0],):
                raise ValueError(
                    f"Expected force_drop_mask shape {(y.shape[0],)}, got {tuple(force_drop_mask.shape)}."
                )
            drop_ids = force_drop_mask.to(device=y.device, dtype=torch.bool)
        h_cond = torch.where(drop_ids.unsqueeze(1), torch.zeros_like(h_cond), h_cond)
        return self.t_embedder(t) + h_cond

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(shape=(x.shape[0], -1, p, c)).reshape(shape=(x.shape[0], -1, c))
        return x.permute(0, 2, 1)

    def forward(self, x, t, y, force_drop_mask=None):
        x = self.input_stem(x)
        c = self._build_condition(t, y, force_drop_mask=force_drop_mask)
        if self.use_abs_pos_embed:
            x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if x.shape[-1] != self.length:
            raise RuntimeError(f"Output length mismatch: got {x.shape[-1]}, expected {self.length}.")
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def build_waveform_model(args):
    return MainlineTransformerModel(
        in_channels=getattr(args, "model_in_channels", getattr(args, "in_channels", 3)),
        hidden_size=getattr(args, "hidden_size", 768),
        depth=getattr(args, "depth", getattr(args, "bottleneck_depth", 24)),
        num_heads=getattr(args, "num_heads", 12),
        learn_sigma=True,
        length=getattr(args, "model_length", getattr(args, "length", 6000)),
        patch_size=getattr(args, "model_patch_size", getattr(args, "patch_size", 6)),
        class_dropout_prob=getattr(args, "class_dropout_prob", 0.1),
        frequency_embedding_size=256,
        abs_geo_encoder_type=getattr(args, "abs_geo_encoder_type", "none"),
        cond_input_min=getattr(args, "min_get", 0.0),
        cond_input_max=getattr(args, "max_get", 1000.0),
        cond_embedding_scale=getattr(args, "cond_embedding_scale", 1.0),
        position_embedding=getattr(args, "position_embedding", "sincos"),
        rope_base=getattr(args, "rope_base", 10000.0),
    )
