"""
Loss functions for the Latent Video Prediction model.

Four components:
  1. L_recon:      Pixel-space L1 between predicted and GT frames
  2. L_latent:     MSE in latent space (primary Transformer training signal)
  3. L_perceptual: Feature-space L1 using frozen encoder features
  4. L_temporal:   Temporal consistency — penalizes motion inconsistency

All components are logged separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class ReconstructionLoss(nn.Module):
    """L1 loss between predicted and ground-truth frames."""

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, T, C, H, W)
            target: (B, T, C, H, W)
        """
        return F.l1_loss(predicted, target)


class LatentPredictionLoss(nn.Module):
    """MSE loss between predicted and encoded GT latent vectors."""

    def forward(
        self,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_latents: (B, T, d_model)
            target_latents: (B, T, d_model)
        """
        return F.mse_loss(predicted_latents, target_latents)


class PerceptualLoss(nn.Module):
    """
    Feature-space L1 loss to discourage blurry predictions.

    Uses a frozen copy of the encoder's first 2 convolutional stages
    as a feature extractor. Computes L1 distance in feature space.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        # Deep copy and freeze first 2 stages of the encoder
        self.feature_extractor = deepcopy(nn.Sequential(*list(encoder.stages[:2])))
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from frames (B*T, C, H, W)."""
        return self.feature_extractor(x)

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, T, C, H, W)
            target: (B, T, C, H, W)
        """
        B, T, C, H, W = predicted.shape
        pred_flat = predicted.reshape(B * T, C, H, W)
        tgt_flat = target.reshape(B * T, C, H, W)

        pred_feat = self._extract(pred_flat)
        tgt_feat = self._extract(tgt_flat)

        return F.l1_loss(pred_feat, tgt_feat)


class TemporalConsistencyLoss(nn.Module):
    """
    Penalizes inconsistent temporal dynamics.

    Computes the MSE between frame-to-frame differences in predicted
    vs. ground-truth sequences. This encourages the model to predict
    consistent motion rather than producing flickering frames.
    """

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted: (B, T, C, H, W)
            target: (B, T, C, H, W)
        """
        # Frame-to-frame differences
        pred_diff = predicted[:, 1:] - predicted[:, :-1]  # (B, T-1, C, H, W)
        tgt_diff = target[:, 1:] - target[:, :-1]

        return F.mse_loss(pred_diff, tgt_diff)


class CombinedLoss(nn.Module):
    """
    Weighted combination of all loss components.

    Supports both fixed hand-tuned weights and learned uncertainty weighting
    (Kendall et al., homoscedastic task uncertainty weighting).
    """

    def __init__(
        self,
        encoder: nn.Module,
        lambda_recon: float = 1.0,
        lambda_latent: float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_temporal: float = 0.5,
        use_learned_weights: bool = True,
        use_vicreg: bool = True,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        lambda_inv: float = 0.5,
        lambda_anchor: float = 1e-4,
    ):
        super().__init__()

        self.use_learned_weights = use_learned_weights
        self.use_vicreg = use_vicreg
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_inv = lambda_inv
        self.lambda_anchor = lambda_anchor

        self.recon_loss = ReconstructionLoss()
        self.latent_loss = LatentPredictionLoss()
        self.perceptual_loss = PerceptualLoss(encoder)
        self.temporal_loss = TemporalConsistencyLoss()

        if use_learned_weights:
            # Learned log-variance parameters (s = log(sigma^2))
            # Initialized such that exp(-s) approximately matches the initial hand-tuned lambdas
            # lambda = 1.0 -> s = -log(1.0) = 0.0
            # lambda = 0.1 -> s = -log(0.1) = 2.3025
            # lambda = 0.5 -> s = -log(0.5) = 0.6931
            self.s_recon = nn.Parameter(torch.tensor(0.0))
            self.s_latent = nn.Parameter(torch.tensor(0.0))
            self.s_perceptual = nn.Parameter(torch.tensor(2.3025))
            self.s_temporal = nn.Parameter(torch.tensor(0.6931))
        else:
            self.lambda_recon = lambda_recon
            self.lambda_latent = lambda_latent
            self.lambda_perceptual = lambda_perceptual
            self.lambda_temporal = lambda_temporal

    def forward(
        self,
        predicted_frames: torch.Tensor,
        target_frames: torch.Tensor,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
        context_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            total_loss: Scalar loss tensor for backpropagation.
            components: Dict of individual loss values and weights for logging.
        """
        l_recon = self.recon_loss(predicted_frames, target_frames)
        l_latent = self.latent_loss(predicted_latents, target_latents)
        l_perceptual = self.perceptual_loss(predicted_frames, target_frames)
        l_temporal = self.temporal_loss(predicted_frames, target_frames)

        if self.use_learned_weights:
            # Kendall et al. formula: Loss = exp(-s) * L + 0.5 * s
            total = (
                torch.exp(-self.s_recon) * l_recon + 0.5 * self.s_recon
                + torch.exp(-self.s_latent) * l_latent + 0.5 * self.s_latent
                + torch.exp(-self.s_perceptual) * l_perceptual + 0.5 * self.s_perceptual
                + torch.exp(-self.s_temporal) * l_temporal + 0.5 * self.s_temporal
            )
            components = {
                "loss/total": total.item(),
                "loss/recon": l_recon.item(),
                "loss/latent": l_latent.item(),
                "loss/perceptual": l_perceptual.item(),
                "loss/temporal": l_temporal.item(),
                "weight/recon": torch.exp(-self.s_recon).item(),
                "weight/latent": torch.exp(-self.s_latent).item(),
                "weight/perceptual": torch.exp(-self.s_perceptual).item(),
                "weight/temporal": torch.exp(-self.s_temporal).item(),
            }
        else:
            total = (
                self.lambda_recon * l_recon
                + self.lambda_latent * l_latent
                + self.lambda_perceptual * l_perceptual
                + self.lambda_temporal * l_temporal
            )
            components = {
                "loss/total": total.item(),
                "loss/recon": l_recon.item(),
                "loss/latent": l_latent.item(),
                "loss/perceptual": l_perceptual.item(),
                "loss/temporal": l_temporal.item(),
            }

        # Apply VICReg latent space regularization (Phase 2)
        if self.use_vicreg and context_latents is not None:
            # Flatten context_latents (B, T_in, D) -> (B * T_in, D)
            Z = context_latents.reshape(-1, context_latents.size(-1))
            D = Z.size(-1)
            eps = 1e-4
            gamma = 1.0

            # 1. Variance term (prevents dimension collapse)
            std = torch.sqrt(torch.var(Z, dim=0) + eps)
            l_var = torch.mean(torch.clamp(gamma - std, min=0.0))

            # 2. Covariance term (off-diagonal decorrelation)
            Z_mean = torch.mean(Z, dim=0, keepdim=True)
            Z_centered = Z - Z_mean
            N = Z.size(0)
            cov = (Z_centered.T @ Z_centered) / max(N - 1, 1)
            cov_squared = cov ** 2
            off_diag_mask = torch.ones_like(cov_squared) - torch.eye(D, device=cov.device)
            l_cov = (cov_squared * off_diag_mask).sum() / D

            # 3. Temporal Invariance term (smoothness between consecutive frames)
            l_inv_vic = F.mse_loss(context_latents[:, 1:], context_latents[:, :-1])

            # 4. L2 Magnitude Anchor
            l_anchor = torch.mean(context_latents ** 2)

            # Add to total loss
            total = (
                total
                + self.lambda_var * l_var
                + self.lambda_cov * l_cov
                + self.lambda_inv * l_inv_vic
                + self.lambda_anchor * l_anchor
            )

            components.update({
                "loss/vicreg_var": l_var.item(),
                "loss/vicreg_cov": l_cov.item(),
                "loss/vicreg_inv": l_inv_vic.item(),
                "loss/vicreg_anchor": l_anchor.item(),
                "loss/total": total.item(),  # update logged total loss
            })

        return total, components


