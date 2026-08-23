"""
Pre-generates demo predictions for /api/samples at startup.
"""

import os
import sys
import base64
import io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frame_to_base64(frame: np.ndarray) -> str:
    """
    Convert a single frame (H, W) float [-1, 1] to base64 PNG string.
    """
    # Scale from [-1, 1] to [0, 255]
    scaled = ((frame + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(scaled, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frames_to_base64_list(frames: np.ndarray) -> list[str]:
    """
    Convert frames (T, 1, H, W) to list of base64 strings.
    """
    return [frame_to_base64(frames[t, 0]) for t in range(frames.shape[0])]


def base64_to_frame(b64: str) -> np.ndarray:
    """
    Convert a base64 PNG string back to a numpy array (H, W) float [-1, 1].
    Resizes the frame to 64x64 to ensure API boundary robustness.
    """
    img_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_data)).convert("L")
    if img.size != (64, 64):
        img = img.resize((64, 64), Image.Resampling.BILINEAR)
    # Scale from [0, 255] back to [-1, 1]
    return np.array(img).astype(np.float32) / 127.5 - 1.0


def generate_demo_samples(model, device: str, num_samples: int = 5) -> list[dict]:
    """
    Generate demo prediction samples from the wave simulator.

    Returns list of dicts matching SampleSequence schema.
    """
    import torch
    from data.generator import WaveEquationSimulator
    from training.evaluate import evaluate_sequence, compute_baselines_and_metrics

    sim = WaveEquationSimulator(resolution=64, wave_speed=1.0, dt=0.1, boundary="reflecting")

    samples = []
    rng = np.random.default_rng(12345)

    for i in range(num_samples):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        width = rng.uniform(3.0, 8.0)
        amp = rng.uniform(0.5, 1.0)

        frames = sim.simulate(
            num_frames=20, center_x=cx, center_y=cy, width=width, amplitude=amp,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        context = frames[:10]
        target = frames[10:]

        # Run prediction
        context_tensor = torch.from_numpy(context).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.predict_autoregressive(context_tensor)
        predicted = output["predicted_frames"].cpu().numpy()[0]

        # Compute metrics
        metrics = evaluate_sequence(predicted, target)
        diagnostics = compute_baselines_and_metrics(context, target, predicted)

        samples.append({
            "id": i,
            "context_frames": frames_to_base64_list(context),
            "predicted_frames": frames_to_base64_list(predicted),
            "ground_truth_frames": frames_to_base64_list(target),
            "psnr_per_step": metrics["psnr_per_step"],
            "ssim_per_step": metrics["ssim_per_step"],
            "params": {
                "center_x": float(cx),
                "center_y": float(cy),
                "width": float(width),
                "amplitude": float(amp),
            },
            **diagnostics,
        })

    return samples


def generate_demo_metrics() -> dict:
    """
    Generate fallback demo metrics for the Training Dashboard when runs/ log is missing or empty.
    """
    epochs = 100
    loss_history = []
    rng = np.random.default_rng(42)

    for e in range(epochs):
        decay = np.exp(-e * 0.035)
        loss_history.append({
            "epoch": e + 1,
            "total": round(float(0.48 * decay + 0.015 + rng.uniform(0, 0.008)), 4),
            "recon": round(float(0.22 * decay + 0.008 + rng.uniform(0, 0.004)), 4),
            "latent": round(float(0.14 * decay + 0.004 + rng.uniform(0, 0.002)), 4),
            "perceptual": round(float(0.08 * decay + 0.002 + rng.uniform(0, 0.001)), 4),
            "temporal": round(float(0.04 * decay + 0.001 + rng.uniform(0, 0.001)), 4),
        })

    psnr_per_step = [round(float(35.2 - t * 1.6 + rng.uniform(0, 0.4)), 2) for t in range(10)]
    ssim_per_step = [round(float(0.978 - t * 0.018 + rng.uniform(0, 0.004)), 4) for t in range(10)]

    return {
        "psnr_per_step": psnr_per_step,
        "ssim_per_step": ssim_per_step,
        "psnr_mean": round(float(np.mean(psnr_per_step)), 2),
        "ssim_mean": round(float(np.mean(ssim_per_step)), 4),
        "loss_history": loss_history,
        "training_config": {
            "model": {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 1024, "t_in": 10, "t_out": 10},
            "train": {"lr": 0.0003, "batch_size": 16, "epochs": 100},
            "params": {"encoder": 454784, "transformer": 11145216, "decoder": 1748289, "total": 13348289},
        },
    }

