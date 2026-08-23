"""
Evaluation utilities: PSNR, SSIM, GIF generation, energy conservation.
"""

import os
import json
import numpy as np
import torch
import imageio.v3 as iio
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn


def compute_psnr(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Compute PSNR between two images.

    Args:
        predicted: (H, W) or (C, H, W), values in [0, 1]
        target: same shape and range
    """
    return psnr_fn(target, predicted, data_range=1.0)


def compute_ssim(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Compute SSIM between two images.

    Args:
        predicted: (H, W), values in [0, 1]
        target: same shape and range
    """
    return ssim_fn(target, predicted, data_range=1.0)


def evaluate_sequence(
    predicted_frames: np.ndarray,
    target_frames: np.ndarray,
) -> dict:
    """
    Evaluate a single predicted sequence against ground truth.

    Args:
        predicted_frames: (T, 1, H, W), values in [-1, 1]
        target_frames: (T, 1, H, W), values in [-1, 1]

    Returns:
        dict with per-step PSNR, SSIM, and summary statistics.
    """
    # Scale from [-1, 1] to [0, 1] for metrics calculation
    predicted_frames = (predicted_frames + 1.0) / 2.0
    target_frames = (target_frames + 1.0) / 2.0

    T = predicted_frames.shape[0]

    psnr_per_step = []
    ssim_per_step = []

    for t in range(T):
        pred = predicted_frames[t, 0]  # (H, W)
        tgt = target_frames[t, 0]

        psnr_per_step.append(compute_psnr(pred, tgt))
        ssim_per_step.append(compute_ssim(pred, tgt))

    return {
        "psnr_per_step": psnr_per_step,
        "ssim_per_step": ssim_per_step,
        "psnr_mean": float(np.mean(psnr_per_step)),
        "ssim_mean": float(np.mean(ssim_per_step)),
        "psnr_first": psnr_per_step[0],
        "psnr_last": psnr_per_step[-1],
        "ssim_first": ssim_per_step[0],
        "ssim_last": ssim_per_step[-1],
    }


def compute_wave_energy_physical(
    frames: np.ndarray,
    c: float = 1.0,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Compute actual physical wave energy functional (kinetic + potential energy).
    Formula: E = 0.5 * sum(v^2 + c^2 * (grad u)^2) * dx^2
    """
    T, C, H, W = frames.shape
    dx = 1.0 / H

    # Compute temporal velocity v = du/dt.
    # At t=0, we assume velocity v=0 (zero initial velocity initialization).
    vel = np.zeros_like(frames)
    vel[1:] = (frames[1:] - frames[:-1]) / dt

    energies = []
    for t in range(T):
        u = frames[t, 0]  # (H, W)
        v = vel[t, 0]  # (H, W)

        # Spatial gradients via central differences
        gy, gx = np.gradient(u, dx)

        # Kinetic energy density: 0.5 * v^2
        ek = 0.5 * (v ** 2)
        # Potential energy density: 0.5 * c^2 * (grad u)^2
        ep = 0.5 * (c ** 2) * (gx ** 2 + gy ** 2)

        # Total energy integrated over spatial grid cells
        total_energy = float(np.sum(ek + ep) * (dx ** 2))
        energies.append(total_energy)

    return np.array(energies)


def compute_energy_error(
    predicted_frames: np.ndarray,
    target_frames: np.ndarray,
) -> dict:
    """
    Compute energy conservation error for physics validation using actual physics.

    Args:
        predicted_frames: (T, 1, H, W)
        target_frames: (T, 1, H, W)

    Returns:
        dict with energy curves and relative error.
    """
    pred_energy = compute_wave_energy_physical(predicted_frames)
    tgt_energy = compute_wave_energy_physical(target_frames)

    # Relative energy error per step
    rel_error = np.abs(pred_energy - tgt_energy) / (tgt_energy + 1e-8)

    return {
        "pred_energy": pred_energy.tolist(),
        "target_energy": tgt_energy.tolist(),
        "relative_error": rel_error.tolist(),
        "mean_relative_error": float(rel_error.mean()),
    }



def create_comparison_gif(
    context_frames: np.ndarray,
    predicted_frames: np.ndarray,
    target_frames: np.ndarray,
    output_path: str,
    fps: int = 8,
) -> str:
    """
    Create a side-by-side GIF: Context → GT | Prediction | Error.

    Args:
        context_frames: (T_in, 1, H, W)
        predicted_frames: (T_out, 1, H, W)
        target_frames: (T_out, 1, H, W)
        output_path: Path to save the GIF.
        fps: Frames per second.

    Returns:
        Path to saved GIF.
    """
    T_out = predicted_frames.shape[0]
    H, W = predicted_frames.shape[2], predicted_frames.shape[3]

    frames_for_gif = []

    for t in range(T_out):
        pred = predicted_frames[t, 0]   # (H, W)
        tgt = target_frames[t, 0]

        # Error heatmap (absolute difference)
        error = np.abs(pred - tgt)

        # Create side-by-side: GT | Prediction | Error
        # Add labels area at top (16 pixels)
        label_h = 16
        panel = np.zeros((H + label_h, W * 3 + 4, 3), dtype=np.uint8)

        # Convert grayscale to RGB
        tgt_rgb = np.stack([tgt] * 3, axis=-1)
        pred_rgb = np.stack([pred] * 3, axis=-1)

        # Error heatmap: blue → red
        err_rgb = np.zeros((H, W, 3))
        err_rgb[:, :, 0] = error          # Red channel
        err_rgb[:, :, 2] = 1.0 - error    # Blue channel

        # Place panels
        panel[label_h:, :W, :] = (tgt_rgb * 255).astype(np.uint8)
        panel[label_h:, W + 2: 2 * W + 2, :] = (pred_rgb * 255).astype(np.uint8)
        panel[label_h:, 2 * W + 4: 3 * W + 4, :] = (err_rgb * 255).astype(np.uint8)

        # Simple label markers (small colored bars)
        panel[:label_h, :W, 1] = 200            # Green bar = GT
        panel[:label_h, W + 2:2 * W + 2, 2] = 200   # Blue bar = Pred
        panel[:label_h, 2 * W + 4:3 * W + 4, 0] = 200  # Red bar = Error

        frames_for_gif.append(panel)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    iio.imwrite(output_path, frames_for_gif, duration=int(1000 / fps), loop=0)

    return output_path


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: str = "cuda",
    num_samples: int = 8,
    output_dir: str = "eval_output",
) -> dict:
    """
    Full evaluation: PSNR/SSIM curves, energy error, sample GIFs.

    Args:
        model: LatentVideoPredictor (set to eval mode).
        dataloader: Test DataLoader.
        device: Device to run on.
        num_samples: Number of GIFs to generate.
        output_dir: Directory for saving outputs.

    Returns:
        dict with aggregated metrics.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []
    all_energy = []

    gif_count = 0

    for batch_idx, batch in enumerate(dataloader):
        context = batch["context"].to(device)
        target = batch["target"].to(device)

        # Full autoregressive prediction
        output = model.predict_autoregressive(context)
        predicted = output["predicted_frames"]

        # Move to numpy
        pred_np = predicted.cpu().numpy()
        tgt_np = target.cpu().numpy()
        ctx_np = context.cpu().numpy()

        B = pred_np.shape[0]

        for i in range(B):
            # Compute metrics
            metrics = evaluate_sequence(pred_np[i], tgt_np[i])
            all_metrics.append(metrics)

            energy = compute_energy_error(pred_np[i], tgt_np[i])
            all_energy.append(energy)

            # Generate GIFs for first N samples
            if gif_count < num_samples:
                gif_path = os.path.join(output_dir, f"sample_{gif_count:03d}.gif")
                create_comparison_gif(ctx_np[i], pred_np[i], tgt_np[i], gif_path)
                gif_count += 1

    # Aggregate metrics
    T = len(all_metrics[0]["psnr_per_step"])
    agg_psnr = [float(np.mean([m["psnr_per_step"][t] for m in all_metrics])) for t in range(T)]
    agg_ssim = [float(np.mean([m["ssim_per_step"][t] for m in all_metrics])) for t in range(T)]

    results = {
        "num_sequences": len(all_metrics),
        "psnr_per_step": agg_psnr,
        "ssim_per_step": agg_ssim,
        "psnr_mean": float(np.mean([m["psnr_mean"] for m in all_metrics])),
        "ssim_mean": float(np.mean([m["ssim_mean"] for m in all_metrics])),
        "psnr_step1": agg_psnr[0],
        "psnr_step10": agg_psnr[-1] if len(agg_psnr) >= 10 else agg_psnr[-1],
        "energy_mean_relative_error": float(
            np.mean([e["mean_relative_error"] for e in all_energy])
        ),
    }

    # Save results
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({results['num_sequences']} sequences)")
    print(f"{'='*50}")
    print(f"  PSNR (mean): {results['psnr_mean']:.2f} dB")
    print(f"  SSIM (mean): {results['ssim_mean']:.4f}")
    print(f"  PSNR @ step 1: {results['psnr_step1']:.2f} dB")
    print(f"  PSNR @ step 10: {results['psnr_step10']:.2f} dB")
    print(f"  Energy error (mean): {results['energy_mean_relative_error']:.4f}")
    print(f"  GIFs saved: {gif_count}")
    print(f"  Results saved to: {output_dir}")

    return results


def compute_baselines_and_metrics(
    context_frames: np.ndarray,
    target_frames: np.ndarray,
    predicted_frames: np.ndarray,
) -> dict:
    """
    Computes persistence and linear extrapolation baselines, along with physical wave
    energies for ground truth, predictions, and baselines.

    Args:
        context_frames: (T_in, 1, H, W) in [-1, 1]
        target_frames: (T_out, 1, H, W) in [-1, 1]
        predicted_frames: (T_out, 1, H, W) in [-1, 1]
    """
    T_out = target_frames.shape[0]

    # 1. Persistence baseline: repeat final context frame
    pred_persistence = np.repeat(context_frames[-1:], T_out, axis=0)

    # 2. Linear extrapolation: u_t = u_last + t * velocity
    # velocity v = u_last - u_prev_to_last
    v = context_frames[-1] - context_frames[-2]
    steps = np.arange(1, T_out + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    pred_linear = np.clip(context_frames[-1] + steps * v, -1.0, 1.0)

    # 3. Evaluate baseline metrics
    metrics_persist = evaluate_sequence(pred_persistence, target_frames)
    metrics_linear = evaluate_sequence(pred_linear, target_frames)

    # 4. Compute physical wave energies
    energy_gt = compute_wave_energy_physical(target_frames)
    energy_pred = compute_wave_energy_physical(predicted_frames)
    energy_persist = compute_wave_energy_physical(pred_persistence)
    energy_linear = compute_wave_energy_physical(pred_linear)

    return {
        "energy_gt": [float(val) for val in energy_gt],
        "energy_pred": [float(val) for val in energy_pred],

        "psnr_persistence": [float(val) for val in metrics_persist["psnr_per_step"]],
        "ssim_persistence": [float(val) for val in metrics_persist["ssim_per_step"]],
        "energy_persistence": [float(val) for val in energy_persist],

        "psnr_linear": [float(val) for val in metrics_linear["psnr_per_step"]],
        "ssim_linear": [float(val) for val in metrics_linear["ssim_per_step"]],
        "energy_linear": [float(val) for val in energy_linear],
    }

