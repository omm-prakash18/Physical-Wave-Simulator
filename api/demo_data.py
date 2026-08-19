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
    """
    img_data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_data)).convert("L")
    # Scale from [0, 255] back to [-1, 1]
    return np.array(img).astype(np.float32) / 127.5 - 1.0


def generate_demo_samples(model, device: str, num_samples: int = 5) -> list[dict]:
    """
    Generate demo prediction samples from the wave simulator.

    Returns list of dicts matching SampleSequence schema.
    """
    import torch
    from data.generator import WaveEquationSimulator
    from training.evaluate import evaluate_sequence

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
        })

    return samples
