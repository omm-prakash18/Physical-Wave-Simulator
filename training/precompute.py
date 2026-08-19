"""
Precompute UMAP projection of latent vectors for the Latent Space Explorer.

Loads the best model, encodes a subset of sequences from the test dataset,
applies UMAP dimensionality reduction, and saves the 2D coordinates to a JSON file.
"""

import os
import sys
import json
import torch
import numpy as np
from umap import UMAP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.dataset import create_dataloaders
from model.full_model import LatentVideoPredictor


def main():
    print("Precomputing latent space UMAP projection...")
    config = Config()
    device = "cpu"
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = os.path.join(config.train.checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Run training first to save checkpoints.")
        sys.exit(1)

    print(f"Loading model from {checkpoint_path}...")
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

    # Load dataset loader
    print("Loading test data...")
    loaders = create_dataloaders(
        data_dir=config.data.data_dir,
        t_in=config.model.t_in,
        t_out=config.model.t_out,
        batch_size=8,
        num_workers=0,
    )
    test_loader = loaders.get("test")
    if not test_loader:
        print("Error: Test split loader not available.")
        sys.exit(1)

    # Gather latent vectors
    all_latents = []
    metadata = []
    max_sequences = 30  # Embed 30 sequences × 20 frames = 600 latents

    print(f"Encoding frames from {max_sequences} sequences...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i * 8 >= max_sequences:
                break
            context = batch["context"].to(device)  # (B, T_in, C, H, W)
            target = batch["target"].to(device)    # (B, T_out, C, H, W)
            full_seq = torch.cat([context, target], dim=1)  # (B, T_in + T_out, C, H, W)

            # Encode all 20 frames
            B, T, C, H, W = full_seq.shape
            flat_frames = full_seq.reshape(B * T, C, H, W)
            flat_latents = model.encoder(flat_frames).cpu().numpy()  # (B*T, d_model)
            latents_reshaped = flat_latents.reshape(B, T, -1)

            for b in range(B):
                seq_idx = i * 8 + b
                if seq_idx >= max_sequences:
                    break
                for t in range(T):
                    all_latents.append(latents_reshaped[b, t])
                    metadata.append({
                        "timestep": t,
                        "sequence": seq_idx,
                        "label": f"Seq {seq_idx}, t={t}",
                    })

    all_latents = np.stack(all_latents, axis=0)  # (N, d_model)
    print(f"Fitting UMAP projection on shape {all_latents.shape}...")

    # UMAP reduction
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(all_latents)  # (N, 2)

    # Save to json file
    umap_points = []
    for idx, (x, y) in enumerate(embedding):
        meta = metadata[idx]
        umap_points.append({
            "x": float(x),
            "y": float(y),
            "timestep": meta["timestep"],
            "sequence": meta["sequence"],
            "label": meta["label"],
        })

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "umap_projection.json")

    with open(output_path, "w") as f:
        json.dump(umap_points, f, indent=2)

    print(f"Precomputation complete! Saved coordinates to {output_path}")


if __name__ == "__main__":
    main()
