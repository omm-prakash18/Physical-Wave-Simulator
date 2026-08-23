"""
Uncertainty Calibration Diagnostic.

Calculates the Pearson correlation coefficient between the MC-dropout
stochastic uncertainty standard deviation maps and the actual absolute
forecast errors to verify uncertainty calibration.
Saves results to checkpoints/uncertainty_calibration.json.
"""

import os
import sys
import json
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.full_model import LatentVideoPredictor
from data.dataset import create_dataloaders
from config import Config


def main():
    print("=" * 60)
    print("Uncertainty Calibration Verification (MC-Dropout)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "best_model.pt"
    )

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint best_model.pt not found at {checkpoint_path}")
        sys.exit(1)

    # 1. Load trained model checkpoint
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
    print("  Model weights successfully loaded!")

    # 2. Load test/validation data
    print("Loading validation dataset...")
    config = Config()
    config.data.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wave_64")
    
    try:
        loaders = create_dataloaders(
            data_dir=config.data.data_dir,
            t_in=config.model.t_in,
            t_out=config.model.t_out,
            batch_size=1, # process sequence-by-sequence
            num_workers=0
        )
        val_loader = loaders.get("val") or loaders.get("test")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)

    if not val_loader:
        print("ERROR: No validation loader found. Ensure data is generated in data/wave_64.")
        sys.exit(1)

    # 3. Evaluate uncertainty vs error correlation
    num_eval_seqs = min(15, len(val_loader.dataset))
    print(f"Evaluating calibration on {num_eval_seqs} sequences...")

    all_std_pixels = []
    all_err_pixels = []

    # Disable normal evaluation, force Dropout activation
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_eval_seqs:
                break
            
            context = batch["context"].to(device) # (1, 10, 1, 64, 64)
            target = batch["target"].to(device)   # (1, 10, 1, 64, 64)

            # MC-Dropout: run 20 passes to estimate epistemic uncertainty
            out = model.predict_uncertainty(context, num_samples=20)
            
            mean_pred = out["mean_predicted_frames"] # (1, 10, 1, 64, 64)
            std_pred = out["std_predicted_frames"]   # (1, 10, 1, 64, 64)

            # Compute absolute pixel-level error
            abs_error = torch.abs(mean_pred - target) # (1, 10, 1, 64, 64)

            # Flatten to gather correlation statistics
            std_np = std_pred.cpu().numpy().flatten()
            err_np = abs_error.cpu().numpy().flatten()

            all_std_pixels.append(std_np)
            all_err_pixels.append(err_np)

    # Reset model to full eval
    model.eval()

    # Concatenate all pixel evaluations
    std_flat = np.concatenate(all_std_pixels)
    err_flat = np.concatenate(all_err_pixels)

    # 4. Compute Pearson correlation coefficient
    corr = float(np.corrcoef(std_flat, err_flat)[0, 1])
    
    # Calculate R² of uncertainty predicting error
    try:
        slope, intercept = np.polyfit(std_flat, err_flat, 1)
        err_pred = slope * std_flat + intercept
        r2 = float(1.0 - (np.sum((err_flat - err_pred) ** 2) / np.sum((err_flat - np.mean(err_flat)) ** 2)))
    except Exception:
        r2 = 0.0

    print("\nCalibration Statistics:")
    print(f"  - Pearson Correlation: {corr:.4f}")
    print(f"  - Uncertainty R2 Fit:  {r2:.4f}")
    
    # Bucket calibration (group standard deviations and check mean error)
    buckets = np.linspace(0, max(0.01, std_flat.max()), 6)
    bucket_centers = []
    bucket_mean_errors = []
    
    for idx in range(len(buckets) - 1):
        mask = (std_flat >= buckets[idx]) & (std_flat < buckets[idx + 1])
        if np.any(mask):
            bucket_centers.append(float((buckets[idx] + buckets[idx + 1]) / 2.0))
            bucket_mean_errors.append(float(err_flat[mask].mean()))
            
    print("\nBucket Calibration (Uncertainty -> Mean Absolute Error):")
    for center, mean_err in zip(bucket_centers, bucket_mean_errors):
        print(f"  - Std Bucket ~ {center:.4f} -> Mean Error: {mean_err:.4f}")

    # Save to json file
    results = {
        "pearson_correlation": round(corr, 4),
        "r2_fit": round(r2, 4),
        "num_sequences": num_eval_seqs,
        "buckets": [round(b, 4) for b in buckets.tolist()],
        "bucket_centers": [round(c, 4) for c in bucket_centers],
        "bucket_mean_errors": [round(me, 4) for me in bucket_mean_errors],
    }

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "uncertainty_calibration.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved calibration results to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
