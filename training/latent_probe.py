"""
Latent Probing for Physical Disentanglement.

Extracts latent representations from the trained CNN Encoder and trains
linear regression probes to predict physical simulation parameters
(center_x, center_y, width, amplitude).
Saves the validation R² scores to checkpoints/latent_probe_metrics.json.
"""

import os
import sys
import json
import numpy as np
import torch

# Add root directory to python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.full_model import LatentVideoPredictor
from data.generator import WaveEquationSimulator


def solve_linear_regression(X_train, y_train, X_val, y_val):
    """
    Fits a linear regression model using ordinary least squares (OLS) in NumPy.
    Returns the validation R² score.
    """
    # Add bias term: shape (N, D + 1)
    X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_val_bias = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

    # Closed-form OLS solution: beta = (X^T * X)^-1 * X^T * y
    try:
        beta, _, _, _ = np.linalg.lstsq(X_train_bias, y_train, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback if matrix is singular
        return 0.0

    # Predict on validation set
    y_pred = X_val_bias @ beta

    # Calculate R² score
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)

    if ss_tot < 1e-8:
        return 0.0
    r2 = 1.0 - (ss_res / ss_tot)
    return float(np.clip(r2, 0.0, 1.0))


def main():
    print("=" * 60)
    print("Latent Space Probing Diagnostics")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "best_model.pt"
    )

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint best_model.pt not found at {checkpoint_path}")
        print("Cannot run latent probe without trained model weights.")
        sys.exit(1)

    print(f"Loading trained model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("  Model weights successfully loaded!")

    # 1. Generate wave sequences with known simulation parameters
    num_samples = 400
    print(f"Generating {num_samples} physical validation sequences...")
    sim = WaveEquationSimulator(resolution=64, wave_speed=1.0, dt=0.1, boundary="reflecting")
    rng = np.random.default_rng(1337) # deterministic seed for probe reproducibility

    X_list = [] # latents (N, 256)
    y_x = []    # center_x (N,)
    y_y = []    # center_y (N,)
    y_w = []    # width (N,)
    y_a = []    # amplitude (N,)

    with torch.no_grad():
        for i in range(num_samples):
            cx = rng.uniform(0.25, 0.75)
            cy = rng.uniform(0.25, 0.75)
            width = rng.uniform(3.0, 7.5)
            amp = rng.uniform(0.5, 1.0)

            # Generate only the first frame (t=0) to probe structural representation
            frame = sim.simulate(
                num_frames=1,
                center_x=cx,
                center_y=cy,
                width=width,
                amplitude=amp,
                rng=rng
            ) # shape: (1, 1, H, W), normalized to [-1, 1]

            # CNN Encoder pass to get latent vector z_0
            frame_tensor = torch.from_numpy(frame).to(device) # (1, 1, H, W)
            latent = model.encoder(frame_tensor) # (1, 256)
            latent_np = latent.cpu().numpy()[0]

            X_list.append(latent_np)
            y_x.append(cx)
            y_y.append(cy)
            y_w.append(width)
            y_a.append(amp)

    X = np.stack(X_list, axis=0) # (N, 256)
    y_x = np.array(y_x)
    y_y = np.array(y_y)
    y_w = np.array(y_w)
    y_a = np.array(y_a)

    # 2. Train/Val Split (80% / 20%)
    split_idx = int(num_samples * 0.8)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    X_train, X_val = X[train_indices], X[val_indices]

    # 3. Solve and calculate R² validation scores per parameter
    r2_x = solve_linear_regression(X_train, y_x[train_indices], X_val, y_x[val_indices])
    r2_y = solve_linear_regression(X_train, y_y[train_indices], X_val, y_y[val_indices])
    r2_w = solve_linear_regression(X_train, y_w[train_indices], X_val, y_w[val_indices])
    r2_a = solve_linear_regression(X_train, y_a[train_indices], X_val, y_a[val_indices])

    # 4. Calculate latent representation statistics (spread & decorrelation)
    std_dims = np.std(X, axis=0)
    min_std = float(np.min(std_dims))
    mean_std = float(np.mean(std_dims))
    max_std = float(np.max(std_dims))

    corr_matrix = np.corrcoef(X, rowvar=False)
    # Fill diagonal with 0 to ignore self-correlation
    np.fill_diagonal(corr_matrix, 0.0)
    max_off_diag_corr = float(np.max(np.abs(corr_matrix)))

    print("\nProbing Results (Validation R2):")
    print(f"  - Pulse Center X:   {r2_x:.4f}")
    print(f"  - Pulse Center Y:   {r2_y:.4f}")
    print(f"  - Pulse Width (sigma): {r2_w:.4f}")
    print(f"  - Wave Amplitude:   {r2_a:.4f}")

    print("\nLatent Space Representation Statistics:")
    print(f"  - Per-dimension std:  min={min_std:.4f}, mean={mean_std:.4f}, max={max_std:.4f}")
    print(f"  - Max off-diagonal correlation: {max_off_diag_corr:.4f}")

    # Save to json file
    results = {
        "r2_center_x": round(r2_x, 4),
        "r2_center_y": round(r2_y, 4),
        "r2_width": round(r2_w, 4),
        "r2_amplitude": round(r2_a, 4),
        "min_std": round(min_std, 4),
        "mean_std": round(mean_std, 4),
        "max_std": round(max_std, 4),
        "max_off_diagonal_correlation": round(max_off_diag_corr, 4),
        "num_samples": num_samples,
    }

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "latent_probe_metrics.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved probing diagnostics to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
