"""
Synthetic physics 2D Wave PDE simulation generator tests.
"""

import numpy as np
import pytest

from data.generator import WaveEquationSimulator, generate_sequence


def test_wave_simulator_shape_and_bounds():
    """Verify wave simulator generates frames of shape (T, 1, H, W) in [-1, 1]."""
    sim = WaveEquationSimulator(resolution=32, wave_speed=1.0, dt=0.1, boundary="reflecting")
    frames = sim.simulate(num_frames=10, center_x=0.5, center_y=0.5, width=4.0, amplitude=1.0)

    assert frames.shape == (10, 1, 32, 32)
    assert frames.dtype == np.float32
    # Value range normalized to [-1, 1]
    assert frames.min() >= -1.05 and frames.max() <= 1.05


def test_wave_simulator_reproducibility():
    """Verify deterministic simulation generation given the same RNG seed."""
    seq1 = generate_sequence(resolution=32, num_frames=5, seed=42)
    seq2 = generate_sequence(resolution=32, num_frames=5, seed=42)

    np.testing.assert_allclose(seq1["frames"], seq2["frames"], rtol=1e-5, atol=1e-5)
    assert seq1["params"]["center_x"] == seq2["params"]["center_x"]


def test_wave_energy_computation():
    """Verify wave energy calculation returns positive scalar energy per frame."""
    sim = WaveEquationSimulator(resolution=32)
    frames = sim.simulate(num_frames=5)
    energies = sim.compute_energy(frames)

    assert len(energies) == 5
    assert (energies > 0).all()
