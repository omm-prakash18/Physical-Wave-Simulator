"""
Encoder-Decoder Transformer for Latent Sequence Prediction.

Context frames are encoded to latents and processed by a Transformer Encoder.
Learned query tokens representing future timesteps are processed by a
Transformer Decoder, cross-attending back to the encoded context latents.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over the time dimension."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            offset: temporal index offset
        """
        return x + self.pe[:, offset : offset + x.size(1), :]


class PreNormEncoderLayer(nn.Module):
    """Pre-norm Transformer Encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # Pre-norm FFN
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class PreNormDecoderLayer(nn.Module):
    """Pre-norm Transformer Decoder layer with self-attention & cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 1. Pre-norm self-attention
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=self_attn_mask, need_weights=False)
        x = x + attn_out

        # 2. Pre-norm cross-attention (cross-attends to memory/encoder outputs)
        h = self.norm2(x)
        cross_out, cross_weights = self.cross_attn(
            h, memory, memory,
            need_weights=need_weights,
            average_attn_weights=False,  # return per-head weights
        )
        x = x + cross_out

        # 3. Pre-norm FFN
        h = self.norm3(x)
        x = x + self.ffn(h)

        return x, cross_weights


class CausalLatentTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for predicting future latents.

    Context frames pass through the Encoder.
    Learned query tokens pass through the Decoder, cross-attending to the Encoder output.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        t_in: int = 10,
        t_out: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.t_in = t_in
        self.t_out = t_out

        # Positional encodings
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=t_in + t_out + 10)

        # Learned query tokens for prediction horizon
        self.query_tokens = nn.Parameter(torch.randn(1, t_out, d_model) * 0.02)

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            PreNormDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def _build_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        """Causal self-attention mask for the decoder queries."""
        return torch.triu(
            torch.ones(length, length, device=device) * float("-inf"),
            diagonal=1,
        )

    def forward(
        self,
        context_latents: torch.Tensor,
        target_latents: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        need_weights: bool = False,
    ) -> dict:
        """
        Args:
            context_latents: (B, T_in, d_model)
            target_latents: (B, T_out, d_model) for teacher forcing
            teacher_forcing_ratio: probability of using target_latents instead of queries
            need_weights: if True, returns cross-attention weights from the last layer

        Returns:
            dict containing predicted_latents and optional cross-attention weights
        """
        B = context_latents.size(0)
        device = context_latents.device

        # ─── Encoder ────────────────────────────────────────
        # Add positional encoding to context latents (offset = 0)
        memory = self.encoder_norm(context_latents)
        memory = self.pos_enc(memory, offset=0)

        for layer in self.encoder_layers:
            memory = layer(memory)

        # ─── Decoder ────────────────────────────────────────
        # Prepare queries
        queries = self.query_tokens.expand(B, -1, -1)  # (B, T_out, d_model)

        # Teacher forcing
        if target_latents is not None and teacher_forcing_ratio > 0:
            tf_mask = (
                torch.rand(B, self.t_out, 1, device=device) < teacher_forcing_ratio
            ).float()
            # Feed context[-1] for step 0 prediction, target[t-1] for step t prediction.
            shifted_target = torch.cat([context_latents[:, -1:], target_latents[:, :-1]], dim=1)
            queries = tf_mask * shifted_target + (1.0 - tf_mask) * queries

        # Add positional encoding to decoder queries (offset = T_in)
        queries = self.decoder_norm(queries)
        queries = self.pos_enc(queries, offset=self.t_in)

        # Self-attention mask (causal)
        self_mask = self._build_causal_mask(self.t_out, device)

        # Run decoder layers
        cross_weights = None
        for i, layer in enumerate(self.decoder_layers):
            is_last = (i == len(self.decoder_layers) - 1)
            queries, weights = layer(
                queries, memory,
                self_attn_mask=self_mask,
                need_weights=(is_last and need_weights),
            )
            if is_last and need_weights:
                cross_weights = weights

        # Output projection
        predicted = self.output_norm(queries)
        predicted = self.output_proj(predicted)

        return {
            "predicted_latents": predicted,
            "attention_weights": [cross_weights] if cross_weights is not None else None,
        }
