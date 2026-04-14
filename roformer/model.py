"""
BS-RoFormer model. Vendored from HiDolen/Mini-BS-RoFormer-V2-46.8M.
All transformers dependencies replaced with direct PyTorch equivalents.
Original license: CC-BY-NC-4.0
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import BSRoformerConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BSRoformerMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, out_size: int | None = None, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.out_size = out_size if out_size is not None else hidden_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_size, bias=bias)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class BSRoformerAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, attention_dropout, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.attention_dropout = attention_dropout
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=True)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        input_shape = hidden_states.size()[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]

        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class BSRoformerLayer(nn.Module):
    def __init__(self, config: BSRoformerConfig):
        super().__init__()
        self.self_attn = BSRoformerAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.attention_dropout,
            config.head_dim,
        )
        self.mlp = BSRoformerMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class BSRoformerAxialTransformer(nn.Module):
    def __init__(self, config: BSRoformerConfig, transformer_depth: int, is_time_transformer: bool):
        super().__init__()
        self.layers = nn.ModuleList([BSRoformerLayer(config) for _ in range(transformer_depth)])
        self.is_time_transformer = is_time_transformer

    def forward(self, hidden_states, position_embeddings, attention_mask):
        if self.is_time_transformer:
            hidden_states = rearrange(hidden_states, "b t f d -> b f t d")

        b, seq_len_1, seq_len_2, d = hidden_states.shape
        hidden_states = rearrange(hidden_states, "b n m d -> (b n) m d")

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, attention_mask)

        hidden_states = rearrange(hidden_states, "(b n) m d -> b n m d", b=b)

        if self.is_time_transformer:
            hidden_states = rearrange(hidden_states, "b f t d -> b t f d")

        return hidden_states


class BandSplit(nn.Module):
    def __init__(self, config: BSRoformerConfig):
        super().__init__()
        self.dim_inputs = tuple(2 * f * config.num_input_channels for f in config.freqs_per_bands)
        self.to_features = nn.ModuleList(
            [
                nn.Sequential(nn.RMSNorm(dim_in, eps=config.rms_norm_eps), nn.Linear(dim_in, config.band_proj_size))
                for dim_in in self.dim_inputs
            ]
        )

    def forward(self, x):
        x_split = x.split(self.dim_inputs, dim=-1)
        outs = [fn(chunk) for chunk, fn in zip(x_split, self.to_features)]
        return torch.stack(outs, dim=-2)


class MaskEstimator(nn.Module):
    def __init__(self, config: BSRoformerConfig):
        super().__init__()
        dim_inputs = tuple(f * config.num_input_channels * 2 for f in config.freqs_per_bands_out)
        self.to_freq_mlps = nn.ModuleList([nn.Linear(config.band_proj_size, dim) for dim in dim_inputs])
        self.to_gate_mlps = nn.ModuleList([nn.Linear(config.band_proj_size, dim // 2) for dim in dim_inputs])

    def forward(self, x):
        x_unbind = x.unbind(dim=-2)
        outs = []
        for band_features, freq_mlp, gate_mlp in zip(x_unbind, self.to_freq_mlps, self.to_gate_mlps):
            mask = freq_mlp(band_features)
            gate = gate_mlp(band_features)
            gate = gate.repeat_interleave(2, dim=-1)
            outs.append(mask * torch.sigmoid(gate))
        return torch.cat(outs, dim=-1)


class BSRoformerModel(nn.Module):
    """Core frequency-domain model."""

    def __init__(self, config: BSRoformerConfig):
        super().__init__()
        self.config = config

        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_base)
        self.band_split = BandSplit(config)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    BSRoformerAxialTransformer(config, config.time_transformer_depth, is_time_transformer=True),
                    BSRoformerAxialTransformer(config, config.freq_transformer_depth, is_time_transformer=False),
                ]
            )
            for _ in range(config.num_hidden_layers)
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mask_estimators = nn.ModuleList([MaskEstimator(config) for _ in range(config.num_stems)])

        self.time_conv_length = config.time_conv_length
        if self.time_conv_length is not None:
            self.time_conv = nn.Sequential(
                nn.RMSNorm(config.band_proj_size * self.time_conv_length, eps=config.rms_norm_eps),
                BSRoformerMLP(
                    hidden_size=config.band_proj_size * self.time_conv_length,
                    intermediate_size=config.hidden_size * self.time_conv_length,
                    out_size=config.hidden_size,
                    bias=True,
                ),
            )
            self.time_deconv = nn.Sequential(
                BSRoformerMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.hidden_size * self.time_conv_length,
                    out_size=config.band_proj_size * self.time_conv_length,
                    bias=True,
                ),
                nn.RMSNorm(config.band_proj_size * self.time_conv_length, eps=config.rms_norm_eps),
            )

        rn = config.register_token_num
        self.register_tokens = nn.Parameter(torch.normal(0, 0.02, size=(rn, rn, config.hidden_size)))

    def forward(self, x, position_ids=None):
        origin_dtype = x.dtype
        target_dtype = next(self.parameters()).dtype
        x = x.to(dtype=target_dtype)
        t_origin = x.shape[1]

        if self.time_conv_length is not None:
            pad_t = (self.time_conv_length - (t_origin % self.time_conv_length)) % self.time_conv_length
            if pad_t > 0:
                x = F.pad(x, (0, 0, 0, pad_t), value=0.0)

        hidden_states = self.band_split(x)

        if self.time_conv_length is not None:
            hidden_states = rearrange(hidden_states, "b (t t_c) n d -> b t n (d t_c)", t_c=self.time_conv_length)
            hidden_states = self.time_conv(hidden_states)

        b, t, n, h = hidden_states.shape

        if position_ids is None:
            position_ids = torch.arange(t, device=hidden_states.device).unsqueeze(0)
        pos_embeds = self.rotary_emb(hidden_states, position_ids)
        pos_embeds_for_freq = self.rotary_emb(
            hidden_states, torch.arange(n, device=hidden_states.device).unsqueeze(0)
        )

        rn = self.config.register_token_num
        hidden_states = F.pad(hidden_states, (0, 0, 0, rn, 0, rn))
        hidden_states[:, t:, n:, :] = self.register_tokens

        def pad_rope(cos, sin):
            return F.pad(cos, (0, 0, 0, rn), value=1.0), F.pad(sin, (0, 0, 0, rn), value=0.0)

        pos_embeds = pad_rope(*pos_embeds)
        pos_embeds_for_freq = pad_rope(*pos_embeds_for_freq)

        for time_transformer, freq_transformer in self.layers:
            hidden_states = time_transformer(hidden_states, position_embeddings=pos_embeds, attention_mask=None)
            hidden_states = freq_transformer(hidden_states, position_embeddings=pos_embeds_for_freq, attention_mask=None)

        hidden_states = hidden_states[:, :t, :n, :]
        hidden_states = self.final_norm(hidden_states)

        if self.time_conv_length is not None:
            hidden_states = self.time_deconv(hidden_states)
            hidden_states = rearrange(hidden_states, "b t n (d t_c) -> b (t t_c) n d", t_c=self.time_conv_length)
            hidden_states = hidden_states[:, :t_origin, :, :]

        mask = torch.stack([fn(hidden_states) for fn in self.mask_estimators], dim=1)
        return mask.to(dtype=origin_dtype)


class BSRoformerForMaskedEstimation(nn.Module):
    """Full model with STFT/iSTFT wrapping the frequency-domain core."""

    def __init__(self, config: BSRoformerConfig):
        super().__init__()
        self.freq_domain_model = BSRoformerModel(config)
        self.config = config

        self.register_buffer("stft_window", torch.hann_window(config.stft_n_fft), persistent=False)
        self.register_buffer("stft_out_window", torch.hann_window(config.stft_n_fft_out), persistent=False)

        self.stft_kwargs = dict(
            n_fft=config.stft_n_fft,
            hop_length=config.stft_hop_length,
            win_length=config.stft_n_fft,
            normalized=False,
        )
        self.stft_out_kwargs = dict(
            n_fft=config.stft_n_fft_out,
            hop_length=config.stft_hop_length,
            win_length=config.stft_n_fft_out,
            normalized=False,
        )
        self.wave_channels = config.num_input_channels

    def forward(self, raw_audio: torch.Tensor):
        device = raw_audio.device
        dtype = raw_audio.dtype
        b, c, t = raw_audio.shape

        with torch.autocast(device_type=device.type, enabled=False):
            raw_audio = raw_audio.to(dtype=torch.float32)
            raw_audio_packed = rearrange(raw_audio, "b c t -> (b c) t")

            stft_repr = torch.stft(raw_audio_packed, **self.stft_kwargs, window=self.stft_window, return_complex=True)
            stft_repr = torch.view_as_real(stft_repr)
            stft_repr = rearrange(stft_repr, "(b c) f t T -> b c f t T", c=c)
            stft_repr_merged = rearrange(stft_repr, "b c f t T -> b t (f c T)")

        stft_repr_merged = stft_repr_merged.to(dtype=dtype)

        mask = self.freq_domain_model(stft_repr_merged)
        mask = rearrange(mask, "b n t (f c T) -> b n c f t T", T=2, c=c)
        mask = mask.to(dtype=torch.float32)

        with torch.autocast(device_type=device.type, enabled=False):
            stft_repr = torch.stft(
                raw_audio_packed, **self.stft_out_kwargs, window=self.stft_out_window, return_complex=True
            )
            stft_repr = torch.view_as_real(stft_repr)
            stft_repr_expanded = rearrange(stft_repr, "(b c) f t T -> b 1 c f t T", c=c)
            stft_repr_complex = torch.view_as_complex(stft_repr_expanded)
            mask_complex = torch.view_as_complex(mask)
            masked_stft = stft_repr_complex * mask_complex

            masked_stft = rearrange(masked_stft, "b n c f t -> (b n c) f t")
            recon_audio = torch.istft(
                masked_stft,
                **self.stft_out_kwargs,
                window=self.stft_out_window,
                return_complex=False,
                length=raw_audio.shape[-1],
            )
            recon_audio = rearrange(recon_audio, "(b n c) t -> b n c t", c=self.wave_channels, n=self.config.num_stems)

        return recon_audio

    @torch.inference_mode()
    def separate(self, mixed_wave, chunk_size=None, overlap_size=None, batch_size=1, gap_size=0, verbose=True):
        """
        Separate [channels, time] waveform → [num_stems, channels, time].
        Stems order for this model: bass, drums, other, vocals.
        """
        assert mixed_wave.dim() == 2
        assert mixed_wave.size(0) == self.config.num_input_channels, (
            f"Expected {self.config.num_input_channels} channels, got {mixed_wave.size(0)}"
        )

        chunk_size = chunk_size or self.config.wave_chunk_size
        overlap_size = overlap_size or (chunk_size // 2)

        fade_size = chunk_size // 10
        window = torch.ones(chunk_size - 2 * gap_size)
        window[:fade_size] = torch.linspace(0, 1, fade_size)
        window[-fade_size:] = torch.linspace(1, 0, fade_size)
        window = F.pad(window, (gap_size, gap_size), value=0.0)
        window = window.to(mixed_wave.device)

        wave_length = mixed_wave.shape[-1]
        n = math.ceil(max(wave_length - chunk_size, 0) / overlap_size) + 1
        required_length = (n - 1) * overlap_size + chunk_size

        if verbose:
            print(f"[SloppyAudio] Input: {mixed_wave.shape}, padded: {required_length}, chunks: {n}, batch: {batch_size}")

        padded_wave = F.pad(mixed_wave, (0, required_length - wave_length), mode="constant")
        unfolded_chunks = padded_wave.unfold(dimension=-1, size=chunk_size, step=overlap_size)
        batch = unfolded_chunks.permute(1, 0, 2)

        outputs = []
        for i, chunk_batch in enumerate(batch.split(batch_size, dim=0)):
            if verbose:
                print(f"\r[SloppyAudio] Processing chunk {i * batch_size + chunk_batch.shape[0]} / {n}")
            outputs.append(self(chunk_batch))
        batch = torch.cat(outputs, dim=0)

        batch = batch * window

        _, num_stems, C, _ = batch.shape
        batch = batch.view(n, -1, chunk_size).permute(1, 2, 0)
        output_result_buffer = F.fold(
            batch,
            output_size=(1, required_length),
            kernel_size=(1, chunk_size),
            stride=(1, overlap_size),
        )
        output_result_buffer = output_result_buffer.view(num_stems, C, -1)

        window_for_fold = window.expand(1, 1, -1).repeat(1, n, 1)
        weighted_sum_counter = F.fold(
            window_for_fold.permute(0, 2, 1),
            output_size=(1, required_length),
            kernel_size=(1, chunk_size),
            stride=(1, overlap_size),
        )
        weighted_sum_counter = weighted_sum_counter.view(1, 1, -1)
        weighted_sum_counter.clamp_min_(1e-8)

        final_output = (output_result_buffer / weighted_sum_counter)[:, :, :wave_length]
        return final_output
