"""
Full Model: LatentVideoPredictor.

Composes the CNN Encoder, Causal Transformer, and CNN Decoder
into a single end-to-end model for latent video prediction.
"""

import torch
import torch.nn as nn

from model.encoder import CNNEncoder
from model.decoder import CNNDecoder
from model.transformer import CausalLatentTransformer


class LatentVideoPredictor(nn.Module):
    """
    End-to-end latent video prediction model.

    Pipeline:
        1. Encode context frames → context latent tokens
        2. (Optionally) encode target frames → GT latent tokens (for training)
        3. Transformer predicts future latent tokens
        4. Decode predicted latents → predicted frames

    Args:
        in_channels: Number of input channels per frame.
        d_model: Latent dimension.
        encoder_channels: Channel progression for the CNN encoder.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_ff: Feed-forward dimension in transformer.
        dropout: Dropout rate.
        t_in: Number of context frames.
        t_out: Number of frames to predict.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 256,
        encoder_channels: tuple[int, ...] = (32, 64, 128, 256),
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        t_in: int = 10,
        t_out: int = 10,
        decoding_mode: str = "parallel",
    ):
        super().__init__()

        self.t_in = t_in
        self.t_out = t_out
        self.d_model = d_model

        # CNN Encoder
        self.encoder = CNNEncoder(
            in_channels=in_channels,
            channel_progression=encoder_channels,
            d_model=d_model,
        )

        # Causal Transformer
        self.transformer = CausalLatentTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            t_in=t_in,
            t_out=t_out,
            decoding_mode=decoding_mode,
        )

        # CNN Decoder
        decoder_channels = tuple(reversed(encoder_channels))  # (256, 128, 64, 32)
        self.decoder = CNNDecoder(
            out_channels=in_channels,
            channel_progression=decoder_channels,
            d_model=d_model,
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of frame sequences to latent tokens.

        Args:
            frames: (B, T, C, H, W)

        Returns:
            latents: (B, T, d_model)
        """
        B, T, C, H, W = frames.shape
        # Flatten batch and time for encoding
        flat = frames.reshape(B * T, C, H, W)
        latents = self.encoder(flat)           # (B*T, d_model)
        return latents.reshape(B, T, self.d_model)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tokens back to frames.

        Args:
            latents: (B, T, d_model)

        Returns:
            frames: (B, T, C, H, W)
        """
        B, T, D = latents.shape
        flat = latents.reshape(B * T, D)
        frames = self.decoder(flat)  # (B*T, C, H, W)
        C, H, W = frames.shape[1:]
        return frames.reshape(B, T, C, H, W)

    def forward(
        self,
        context_frames: torch.Tensor,
        target_frames: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        need_weights: bool = False,
    ) -> dict:
        """
        Full forward pass.

        Args:
            context_frames: (B, T_in, C, H, W) — input context sequence
            target_frames: (B, T_out, C, H, W) — GT future frames (training only)
            teacher_forcing_ratio: probability of using GT latents during training
            need_weights: if True, return transformer attention weights

        Returns:
            dict with keys:
                'predicted_frames': (B, T_out, C, H, W)
                'predicted_latents': (B, T_out, d_model)
                'context_latents': (B, T_in, d_model)
                'target_latents': (B, T_out, d_model) or None
                'attention_weights': list or None
        """
        # 1. Encode context frames
        context_latents = self.encode_frames(context_frames)  # (B, T_in, d_model)

        # 2. Encode target frames (for loss computation + teacher forcing)
        target_latents = None
        if target_frames is not None:
            target_latents = self.encode_frames(target_frames)  # (B, T_out, d_model)

        # 3. Predict future latents via transformer
        transformer_out = self.transformer(
            context_latents=context_latents,
            target_latents=target_latents,
            teacher_forcing_ratio=teacher_forcing_ratio,
            need_weights=need_weights,
        )
        predicted_latents = transformer_out["predicted_latents"]  # (B, T_out, d_model)

        # 4. Decode predicted latents to frames
        predicted_frames = self.decode_latents(predicted_latents)  # (B, T_out, C, H, W)

        return {
            "predicted_frames": predicted_frames,
            "predicted_latents": predicted_latents,
            "context_latents": context_latents,
            "target_latents": target_latents,
            "attention_weights": transformer_out["attention_weights"],
        }

    def load_state_dict(self, state_dict, strict=True):
        """Custom loader to prevent crashes if latent_proj weights are missing in checkpoints."""
        has_proj = any("latent_proj" in k for k in state_dict.keys())
        if not has_proj:
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def count_parameters(self) -> dict:
        """Count parameters per component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "encoder": _count(self.encoder),
            "transformer": _count(self.transformer),
            "decoder": _count(self.decoder),
            "total": _count(self),
        }

    @torch.no_grad()
    def predict_autoregressive(
        self,
        context_frames: torch.Tensor,
        need_weights: bool = False,
    ) -> dict:
        """
        Autoregressive prediction at inference time (no teacher forcing).
        """
        self.eval()
        return self.forward(
            context_frames=context_frames,
            target_frames=None,
            teacher_forcing_ratio=0.0,
            need_weights=need_weights,
        )

    def predict_uncertainty(
        self,
        context_frames: torch.Tensor,
        num_samples: int = 20,
    ) -> dict:
        """
        Perform Monte Carlo Dropout (MC-Dropout) stochastic forward passes
        to estimate epistemic uncertainty (mean and per-pixel standard deviation).

        Args:
            context_frames: (B, T_in, C, H, W)
            num_samples: Number of stochastic forward passes (default: 20)

        Returns:
            dict containing:
                'mean_predicted_frames': (B, T_out, C, H, W)
                'std_predicted_frames': (B, T_out, C, H, W)
                'sample_predictions': (N, B, T_out, C, H, W)
        """
        self.eval()
        # Enable dropout layers for stochastic inference
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                out = self.forward(context_frames=context_frames, teacher_forcing_ratio=0.0)
                preds.append(out["predicted_frames"])

        # Reset model back to full eval mode
        self.eval()

        # Stack predictions along new sample dimension (N, B, T_out, C, H, W)
        preds_stack = torch.stack(preds, dim=0)
        mean_preds = preds_stack.mean(dim=0)
        std_preds = preds_stack.std(dim=0)

        return {
            "mean_predicted_frames": mean_preds,
            "std_predicted_frames": std_preds,
            "sample_predictions": preds_stack,
        }

