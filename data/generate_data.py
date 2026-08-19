"""
CLI script to generate the wave equation dataset.

Usage:
    python -m data.generate_data --num_sequences 10000 --output_dir data/wave_64
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import WaveEquationSimulator


def main():
    parser = argparse.ArgumentParser(description="Generate wave equation dataset")
    parser.add_argument("--num_sequences", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="data/wave_64")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--wave_speed", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--boundary", type=str, default="reflecting",
                        choices=["reflecting", "absorbing"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    sim = WaveEquationSimulator(
        resolution=args.resolution,
        wave_speed=args.wave_speed,
        dt=args.dt,
        boundary=args.boundary,
    )

    # Create output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)

    # Determine split indices
    n_train = int(args.num_sequences * args.train_ratio)
    n_val = int(args.num_sequences * args.val_ratio)

    # Track statistics
    all_params = []

    print(f"Generating {args.num_sequences} sequences at {args.resolution}x{args.resolution}...")
    print(f"  Splits: train={n_train}, val={n_val}, test={args.num_sequences - n_train - n_val}")
    print(f"  Frames per sequence: {args.num_frames}")
    print(f"  Boundary: {args.boundary}")
    print(f"  Output: {args.output_dir}")
    print()

    for i in tqdm(range(args.num_sequences), desc="Generating"):
        # Random simulation parameters
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        width = rng.uniform(3.0, 8.0)
        amplitude = rng.uniform(0.5, 1.0)

        # Generate sequence
        frames = sim.simulate(
            num_frames=args.num_frames,
            center_x=cx,
            center_y=cy,
            width=width,
            amplitude=amplitude,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        params = {
            "center_x": float(cx),
            "center_y": float(cy),
            "width": float(width),
            "amplitude": float(amplitude),
            "seed_offset": i,
        }
        all_params.append(params)

        # Determine split
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"

        # Save
        filepath = os.path.join(args.output_dir, split, f"seq_{i:06d}.npz")
        np.savez_compressed(filepath, frames=frames, params=params)

    # Save metadata
    metadata = {
        "dataset": "wave",
        "num_sequences": args.num_sequences,
        "resolution": args.resolution,
        "num_frames": args.num_frames,
        "wave_speed": args.wave_speed,
        "dt": sim.dt,
        "boundary": args.boundary,
        "seed": args.seed,
        "splits": {
            "train": n_train,
            "val": n_val,
            "test": args.num_sequences - n_train - n_val,
        },
        "courant_number": sim.courant,
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Dataset saved to {args.output_dir}")
    print(f"  Metadata: {os.path.join(args.output_dir, 'metadata.json')}")

    # Quick energy conservation check on last sequence
    energies = sim.compute_energy(frames)
    energy_drift = abs(energies[-1] - energies[0]) / (energies[0] + 1e-8)
    print(f"  Energy drift (last seq): {energy_drift:.4f}")


if __name__ == "__main__":
    main()
