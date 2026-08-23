"""
PyTorch neural network architecture and loss function invariant tests.
"""

import pytest
import torch

from model.encoder import CNNEncoder
from model.decoder import CNNDecoder
from model.transformer import CausalLatentTransformer
from model.full_model import LatentVideoPredictor
from model.losses import (
    ReconstructionLoss,
    LatentPredictionLoss,
    PerceptualLoss,
    TemporalConsistencyLoss,
    CombinedLoss,
)



@pytest.fixture
def sample_frames():
    """Batch of frames: (B=2, T=10, C=1, H=64, W=64)."""
    return torch.randn(2, 10, 1, 64, 64)


@pytest.fixture
def sample_latents():
    """Batch of latents: (B=2, T=10, D=256)."""
    return torch.randn(2, 10, 256)


def test_cnn_encoder():
    """Verify CNN Encoder compresses (B*T, C, H, W) to (B*T, d_model)."""
    encoder = CNNEncoder(in_channels=1, channel_progression=(32, 64, 128, 256), d_model=256)
    x = torch.randn(4, 1, 64, 64)
    out = encoder(x)
    assert out.shape == (4, 256)
    assert not torch.isnan(out).any()


def test_cnn_decoder():
    """Verify CNN Decoder reconstructs (B*T, d_model) back to (B*T, C, H, W)."""
    decoder = CNNDecoder(out_channels=1, channel_progression=(256, 128, 64, 32), d_model=256)
    z = torch.randn(4, 256)
    out = decoder(z)
    assert out.shape == (4, 1, 64, 64)
    # Output should be bounded in [-1, 1] due to Tanh activation
    assert (out >= -1.05).all() and (out <= 1.05).all()


def test_causal_transformer():
    """Verify Transformer generates predicted latents and optional attention weights."""
    transformer = CausalLatentTransformer(
        d_model=256, n_layers=2, n_heads=4, d_ff=512, dropout=0.1, t_in=10, t_out=10
    )
    context_latents = torch.randn(2, 10, 256)
    target_latents = torch.randn(2, 10, 256)

    out = transformer(context_latents=context_latents, target_latents=target_latents, need_weights=True)
    assert "predicted_latents" in out
    assert out["predicted_latents"].shape == (2, 10, 256)

    assert "attention_weights" in out
    assert len(out["attention_weights"]) == 1  # Last layer cross-attention weights



def test_full_model_forward(sample_frames):
    """Verify end-to-end forward pass through LatentVideoPredictor."""
    model = LatentVideoPredictor(
        in_channels=1,
        d_model=128,
        encoder_channels=(16, 32, 64, 128),
        n_layers=2,
        n_heads=4,
        d_ff=256,
        t_in=10,
        t_out=10,
    )
    context_frames = sample_frames
    target_frames = torch.randn(2, 10, 1, 64, 64)

    output = model(context_frames=context_frames, target_frames=target_frames, need_weights=True)

    assert output["predicted_frames"].shape == (2, 10, 1, 64, 64)
    assert output["predicted_latents"].shape == (2, 10, 128)
    assert output["context_latents"].shape == (2, 10, 128)
    assert output["target_latents"].shape == (2, 10, 128)
    assert output["attention_weights"] is not None


def test_full_model_autoregressive(sample_frames):
    """Verify inference-time autoregressive prediction mode."""
    model = LatentVideoPredictor(
        in_channels=1,
        d_model=128,
        encoder_channels=(16, 32, 64, 128),
        n_layers=2,
        n_heads=4,
        d_ff=256,
        t_in=10,
        t_out=10,
    )
    output = model.predict_autoregressive(sample_frames)
    assert output["predicted_frames"].shape == (2, 10, 1, 64, 64)
    assert output["target_latents"] is None


def test_parameter_counting():
    """Verify parameter counter returns positive counts per component."""
    model = LatentVideoPredictor(
        in_channels=1,
        d_model=128,
        encoder_channels=(16, 32, 64, 128),
        n_layers=2,
        n_heads=4,
        d_ff=256,
    )
    counts = model.count_parameters()
    assert counts["encoder"] > 0
    assert counts["transformer"] > 0
    assert counts["decoder"] > 0
    assert counts["total"] == counts["encoder"] + counts["transformer"] + counts["decoder"]


def test_loss_functions():
    """Verify numerical sanity and non-negativity of composite loss functions."""
    pred_frames = torch.randn(2, 10, 1, 64, 64)
    target_frames = torch.randn(2, 10, 1, 64, 64)
    pred_latents = torch.randn(2, 10, 128)
    target_latents = torch.randn(2, 10, 128)

    l1_loss = ReconstructionLoss()
    mse_loss = LatentPredictionLoss()
    temp_loss = TemporalConsistencyLoss()

    l1_val = l1_loss(pred_frames, target_frames)
    mse_val = mse_loss(pred_latents, target_latents)
    temp_val = temp_loss(pred_frames, target_frames)

    assert l1_val.item() >= 0.0
    assert mse_val.item() >= 0.0
    assert temp_val.item() >= 0.0

    # Test Combined Loss (Fixed Weights)
    encoder = CNNEncoder(in_channels=1, channel_progression=(16, 32, 64, 128), d_model=128)
    combined_fixed = CombinedLoss(
        encoder=encoder,
        lambda_recon=1.0,
        lambda_latent=1.0,
        lambda_perceptual=0.1,
        lambda_temporal=0.5,
        use_learned_weights=False,
    )

    total_fixed, components_fixed = combined_fixed(
        predicted_frames=pred_frames,
        target_frames=target_frames,
        predicted_latents=pred_latents,
        target_latents=target_latents,
    )

    assert total_fixed.item() > 0.0
    assert "loss/total" in components_fixed
    assert "loss/recon" in components_fixed
    assert "loss/latent" in components_fixed
    assert "loss/perceptual" in components_fixed
    assert "loss/temporal" in components_fixed

    # Test Combined Loss (Learned Weights with backward gradients check)
    combined_learned = CombinedLoss(
        encoder=encoder,
        use_learned_weights=True,
    )

    total_learned, components_learned = combined_learned(
        predicted_frames=pred_frames,
        target_frames=target_frames,
        predicted_latents=pred_latents,
        target_latents=target_latents,
    )

    assert total_learned.item() > 0.0
    assert "loss/total" in components_learned
    assert "weight/recon" in components_learned

    # Backward check: optimize learned weights
    total_learned.backward()
    assert combined_learned.s_recon.grad is not None
    assert combined_learned.s_latent.grad is not None
    assert combined_learned.s_perceptual.grad is not None
    assert combined_learned.s_temporal.grad is not None



def test_onnx_parity():
    """Verify that ONNX Runtime predictions match PyTorch predictions within 1e-3 tolerance."""
    import os
    import numpy as np
    import onnxruntime as ort

    onnx_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "model.onnx"
    )
    if not os.path.exists(onnx_path):
        pytest.skip("model.onnx not found in checkpoints/")

    from model.full_model import LatentVideoPredictor

    # Load PyTorch model
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "best_model.pt"
    )
    if not os.path.exists(checkpoint_path):
        pytest.skip("best_model.pt not found in checkpoints/")

    # Read config state from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})

    model = LatentVideoPredictor(
        in_channels=cfg.get("channels", 1),
        d_model=cfg.get("d_model", 256),
        encoder_channels=tuple(cfg.get("encoder_channels", (32, 64, 128, 256))),
        n_layers=cfg.get("n_layers", 6),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 1024),
        t_in=cfg.get("t_in", 10),
        t_out=cfg.get("t_out", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy input (1, 10, 1, 64, 64)
    dummy_input = torch.randn(1, 10, 1, 64, 64)

    # PyTorch inference
    with torch.no_grad():
        pt_out = model.predict_autoregressive(dummy_input)
        pt_frames = pt_out["predicted_frames"].numpy()

    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_input = {input_name: dummy_input.numpy()}
    ort_out = session.run(None, ort_input)
    ort_frames = ort_out[0]

    # Max absolute difference
    max_diff = np.max(np.abs(pt_frames - ort_frames))
    print(f"ONNX vs PyTorch max absolute difference: {max_diff}")
    assert max_diff < 1e-3, f"ONNX and PyTorch outputs do not match within tolerance, max diff: {max_diff}"


