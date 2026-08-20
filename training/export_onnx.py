"""
MLOps: ONNX Model Exporter & Latency Benchmarking Utility.

Traces the PyTorch LatentVideoPredictor model and exports it to ONNX format.
Validates exported computational graph with `onnx.checker` and runs a performance
benchmark comparing PyTorch vs ONNX Runtime CPU/CUDA latency.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.full_model import LatentVideoPredictor
from config import DEFAULT_CONFIG


class AutoregressiveWrapper(torch.nn.Module):
    """
    Wrapper module exposing a pure Tensor -> Tensor signature for ONNX export.
    """

    def __init__(self, model: LatentVideoPredictor):
        super().__init__()
        self.model = model

    def forward(self, context_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context_frames: (B, T_in, C, H, W)
        Returns:
            predicted_frames: (B, T_out, C, H, W)
        """
        output = self.model.predict_autoregressive(context_frames)
        return output["predicted_frames"]


def export_onnx(
    output_path: str = "checkpoints/model.onnx",
    checkpoint_path: str | None = None,
    opset_version: int = 17,
):
    """Export LatentVideoPredictor to ONNX format."""
    print("=" * 60)
    print("MLOps Engine — Exporting PyTorch Model to ONNX...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize model
    model = LatentVideoPredictor(
        in_channels=DEFAULT_CONFIG.data.channels,
        d_model=DEFAULT_CONFIG.model.d_model,
        encoder_channels=DEFAULT_CONFIG.model.encoder_channels,
        n_layers=DEFAULT_CONFIG.model.n_layers,
        n_heads=DEFAULT_CONFIG.model.n_heads,
        d_ff=DEFAULT_CONFIG.model.d_ff,
        t_in=DEFAULT_CONFIG.model.t_in,
        t_out=DEFAULT_CONFIG.model.t_out,
    ).to(device)

    # Load weights if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("No checkpoint specified or found; exporting structural model architecture.")

    model.eval()
    wrapper = AutoregressiveWrapper(model).to(device)

    # Dummy input for tracing: (B=1, T_in=10, C=1, H=64, W=64)
    dummy_input = torch.randn(1, DEFAULT_CONFIG.model.t_in, 1, 64, 64, device=device)

    print(f"Exporting computational graph to: {output_path}")
    
    try:
        # PyTorch 2.x Dynamo ONNX exporter
        onnx_program = torch.onnx.export(
            wrapper,
            (dummy_input,),
            output_path,
            export_params=True,
            dynamo=True,
        )
    except Exception as e:
        print(f"Dynamo exporter notice ({e}), falling back to TorchScript traced export...")
        traced_model = torch.jit.trace(wrapper, dummy_input)
        torch.onnx.export(
            traced_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            input_names=["context_frames"],
            output_names=["predicted_frames"],
            dynamic_axes={
                "context_frames": {0: "batch_size"},
                "predicted_frames": {0: "batch_size"},
            },
        )




    print("Checking exported ONNX model graph validity...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX Model Verification PASSED!")

    # Benchmark ONNX Runtime vs PyTorch
    benchmark_runtimes(wrapper, output_path, dummy_input, device)

    return output_path


def benchmark_runtimes(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    device: str,
    num_runs: int = 50,
) -> dict:
    """Run latency benchmarks comparing PyTorch vs ONNX Runtime."""
    print("-" * 60)
    print("Running Latency Benchmark (50 iterations)...")
    import onnxruntime as ort

    # Warmup PyTorch
    with torch.no_grad():
        for _ in range(5):
            _ = pytorch_model(dummy_input)

    # Measure PyTorch Latency
    pt_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = pytorch_model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    pt_avg_ms = np.mean(pt_times)
    pt_p95_ms = np.percentile(pt_times, 95)

    # Initialize ONNX Runtime Session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    ort_input = {input_name: dummy_input.cpu().numpy()}

    # Warmup ONNX
    for _ in range(5):
        _ = session.run(None, ort_input)

    # Measure ONNX Runtime Latency
    ort_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, ort_input)
        ort_times.append((time.perf_counter() - t0) * 1000.0)

    ort_avg_ms = np.mean(ort_times)
    ort_p95_ms = np.percentile(ort_times, 95)

    speedup = pt_avg_ms / max(ort_avg_ms, 1e-6)

    print(f"  PyTorch Latency : Avg = {pt_avg_ms:.2f} ms | P95 = {pt_p95_ms:.2f} ms")
    print(f"  ONNX Runtime    : Avg = {ort_avg_ms:.2f} ms | P95 = {ort_p95_ms:.2f} ms")
    print(f"  Latency Speedup : {speedup:.2f}x faster with ONNX Runtime")
    print("=" * 60)

    return {
        "pytorch_avg_ms": float(pt_avg_ms),
        "pytorch_p95_ms": float(pt_p95_ms),
        "onnx_avg_ms": float(ort_avg_ms),
        "onnx_p95_ms": float(ort_p95_ms),
        "speedup": float(speedup),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch Latent Video Model to ONNX.")
    parser.add_argument("--output", type=str, default="checkpoints/model.onnx", help="Output path for ONNX file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Input PyTorch checkpoint")
    args = parser.parse_args()

    export_onnx(output_path=args.output, checkpoint_path=args.checkpoint)
