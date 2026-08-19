"""
2D Wave Equation Simulator.

Generates synthetic training data by simulating a Gaussian pulse
propagating on a 2D grid with reflecting or absorbing boundaries.
Fully deterministic given a seed.
"""

import numpy as np
from typing import Literal


class WaveEquationSimulator:
    """
    Finite-difference 2D wave equation solver.

    Solves:  ∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)

    Uses the standard second-order central difference scheme:
        u(t+1) = 2*u(t) - u(t-1) + (c*dt/dx)² * laplacian(u(t))
    """

    def __init__(
        self,
        resolution: int = 64,
        wave_speed: float = 1.0,
        dt: float = 0.1,
        boundary: Literal["reflecting", "absorbing"] = "reflecting",
    ):
        self.resolution = resolution
        self.c = wave_speed
        self.dt = dt
        self.dx = 1.0 / resolution  # spatial step
        self.boundary = boundary

        # CFL stability condition: c * dt / dx < 1/sqrt(2)
        self.courant = self.c * self.dt / self.dx
        if self.courant > 0.7:
            # Auto-adjust dt to satisfy CFL
            self.dt = 0.7 * self.dx / self.c
            self.courant = self.c * self.dt / self.dx

    def _init_gaussian_pulse(
        self,
        center_x: float,
        center_y: float,
        width: float,
        amplitude: float,
    ) -> np.ndarray:
        """Create a Gaussian pulse on the grid."""
        N = self.resolution
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)

        # Gaussian pulse
        sigma = width / N
        pulse = amplitude * np.exp(
            -((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma ** 2)
        )
        return pulse.astype(np.float32)

    def _apply_boundary(self, u: np.ndarray) -> np.ndarray:
        """Apply boundary conditions."""
        if self.boundary == "reflecting":
            # Neumann BC: ∂u/∂n = 0 at boundaries
            u[0, :] = u[1, :]
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
        elif self.boundary == "absorbing":
            # Simple absorbing: decay at boundary
            decay = 0.95
            u[0, :] *= decay
            u[-1, :] *= decay
            u[:, 0] *= decay
            u[:, -1] *= decay
        return u

    def _compute_laplacian(self, u: np.ndarray) -> np.ndarray:
        """Compute 2D discrete Laplacian using central differences."""
        laplacian = np.zeros_like(u)
        laplacian[1:-1, 1:-1] = (
            u[2:, 1:-1] + u[:-2, 1:-1]
            + u[1:-1, 2:] + u[1:-1, :-2]
            - 4.0 * u[1:-1, 1:-1]
        ) / (self.dx ** 2)
        return laplacian

    def simulate(
        self,
        num_frames: int,
        center_x: float | None = None,
        center_y: float | None = None,
        width: float | None = None,
        amplitude: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Run a wave simulation and return frames.

        Args:
            num_frames: Number of frames to generate.
            center_x: Pulse center x in [0, 1]. Random if None.
            center_y: Pulse center y in [0, 1]. Random if None.
            width: Pulse width in grid units. Random if None.
            amplitude: Pulse amplitude. Random if None.
            rng: Numpy random generator for reproducibility.

        Returns:
            frames: Array of shape (num_frames, 1, H, W), normalized to [0, 1].
        """
        if rng is None:
            rng = np.random.default_rng()

        # Randomize parameters if not specified
        if center_x is None:
            center_x = rng.uniform(0.2, 0.8)
        if center_y is None:
            center_y = rng.uniform(0.2, 0.8)
        if width is None:
            width = rng.uniform(3.0, 8.0)
        if amplitude is None:
            amplitude = rng.uniform(0.5, 1.0)

        N = self.resolution

        # Initialize: u(t=0) = pulse, u(t=-1) = pulse (zero initial velocity)
        u_curr = self._init_gaussian_pulse(center_x, center_y, width, amplitude)
        u_prev = u_curr.copy()

        frames = []
        courant_sq = self.courant ** 2

        # Run multiple sub-steps per frame for smoother dynamics
        sub_steps = 5

        for frame_idx in range(num_frames):
            frames.append(u_curr.copy())

            for _ in range(sub_steps):
                laplacian = self._compute_laplacian(u_curr)

                # Leapfrog integration
                u_next = 2.0 * u_curr - u_prev + courant_sq * laplacian * (self.dx ** 2)

                # Apply boundary conditions
                u_next = self._apply_boundary(u_next)

                u_prev = u_curr
                u_curr = u_next

        # Stack frames: (T, H, W) -> (T, 1, H, W)
        frames = np.stack(frames, axis=0)[:, np.newaxis, :, :]

        # Normalize to [-1, 1] per-sequence
        fmin = frames.min()
        fmax = frames.max()
        if fmax - fmin > 1e-8:
            frames = 2.0 * (frames - fmin) / (fmax - fmin) - 1.0
        else:
            frames = np.zeros_like(frames) * 2.0 - 1.0

        return frames.astype(np.float32)

    def compute_energy(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute total energy per frame for physics validation.

        Energy ∝ sum(u²) — a simplified proxy for the actual wave energy.

        Args:
            frames: Shape (T, 1, H, W).

        Returns:
            energies: Shape (T,).
        """
        return (frames ** 2).sum(axis=(1, 2, 3))


def generate_sequence(
    resolution: int = 64,
    num_frames: int = 20,
    wave_speed: float = 1.0,
    dt: float = 0.1,
    boundary: str = "reflecting",
    seed: int | None = None,
    **kwargs,
) -> dict:
    """
    Convenience function to generate a single sequence with metadata.

    Returns:
        dict with keys: 'frames' (T, 1, H, W), 'params' (dict of sim params).
    """
    rng = np.random.default_rng(seed)

    sim = WaveEquationSimulator(
        resolution=resolution,
        wave_speed=wave_speed,
        dt=dt,
        boundary=boundary,
    )

    # Random params
    cx = kwargs.get("center_x", rng.uniform(0.2, 0.8))
    cy = kwargs.get("center_y", rng.uniform(0.2, 0.8))
    w = kwargs.get("width", rng.uniform(3.0, 8.0))
    a = kwargs.get("amplitude", rng.uniform(0.5, 1.0))

    frames = sim.simulate(
        num_frames=num_frames,
        center_x=cx,
        center_y=cy,
        width=w,
        amplitude=a,
        rng=rng,
    )

    return {
        "frames": frames,
        "params": {
            "center_x": float(cx),
            "center_y": float(cy),
            "width": float(w),
            "amplitude": float(a),
            "wave_speed": wave_speed,
            "dt": dt,
            "boundary": boundary,
            "resolution": resolution,
            "num_frames": num_frames,
        },
    }
