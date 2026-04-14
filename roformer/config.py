"""BS-RoFormer model configuration. Standalone, no transformers dependency."""


class BSRoformerConfig:
    model_type = "bs_roformer"

    def __init__(
        self,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        head_dim: int | None = None,
        attention_dropout: float = 0.0,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 8,
        intermediate_size: int = 384 * 4,
        register_token_num: int = 4,
        num_input_channels: int = 1,
        num_stems: int = 1,
        band_proj_size: int | None = None,
        time_conv_length: int | None = None,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        freqs_per_bands: tuple[int, ...] = (),
        freqs_per_bands_out: tuple[int, ...] = (),
        stft_n_fft: int = 4096,
        stft_n_fft_out: int = 2048,
        stft_hop_length: int = 512,
        wave_sample_rate: int = 44100,
        wave_chunk_size: int = 44100 * 6,
        rms_norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        initializer_range: float = 0.02,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.register_token_num = register_token_num

        self.num_input_channels = num_input_channels
        self.num_stems = num_stems
        self.band_proj_size = band_proj_size if band_proj_size is not None else hidden_size
        self.time_conv_length = time_conv_length
        self.time_transformer_depth = time_transformer_depth
        self.freq_transformer_depth = freq_transformer_depth
        self.freqs_per_bands = tuple(freqs_per_bands)
        self.freqs_per_bands_out = tuple(freqs_per_bands_out)

        assert len(self.freqs_per_bands) == len(self.freqs_per_bands_out)

        self.stft_n_fft = stft_n_fft
        self.stft_n_fft_out = stft_n_fft_out
        self.stft_hop_length = stft_hop_length
        self.wave_sample_rate = wave_sample_rate
        self.wave_chunk_size = wave_chunk_size

        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base
        self.initializer_range = initializer_range
        self.dtype = dtype

    @classmethod
    def from_dict(cls, d: dict) -> "BSRoformerConfig":
        filtered = {k: v for k, v in d.items() if not k.startswith("_")}
        filtered.pop("architectures", None)
        filtered.pop("auto_map", None)
        filtered.pop("model_type", None)
        filtered.pop("transformers_version", None)
        return cls(**filtered)
