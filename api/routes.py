"""
FastAPI route handlers for the Latent Video Prediction API.
"""

import os
import sys
import json
import numpy as np
import torch
from fastapi import APIRouter, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    PredictRequest, PredictResponse,
    GenerateRequest, GenerateResponse,
    SamplesResponse, SampleSequence,
    MetricsResponse,
    AttentionResponse,
)
from api.demo_data import (
    base64_to_frame, frames_to_base64_list, frame_to_base64,
    generate_demo_samples,
)
from data.generator import WaveEquationSimulator
from training.evaluate import evaluate_sequence

router = APIRouter()

# These will be set by main.py at startup
_model = None
_device = "cpu"
_demo_samples = []
_metrics_cache = None


def set_model(model, device: str):
    """Set the model singleton (called from main.py startup)."""
    global _model, _device
    _model = model
    _device = device


def set_demo_samples(samples: list[dict]):
    """Set pre-generated demo samples."""
    global _demo_samples
    _demo_samples = samples


def load_metrics_cache():
    """Load training metrics from the latest run directory."""
    global _metrics_cache
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")

    if not os.path.exists(runs_dir):
        return

    # Find latest run with metrics
    runs = sorted([d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))])
    if not runs:
        return

    latest_run = runs[-1]

    # Look for eval metrics
    for subdir in sorted(os.listdir(os.path.join(runs_dir, latest_run)), reverse=True):
        metrics_path = os.path.join(runs_dir, latest_run, subdir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                _metrics_cache = json.load(f)

            # Also load config if available
            config_path = os.path.join(runs_dir, latest_run, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    _metrics_cache["training_config"] = json.load(f)
            break


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict future frames from context frames."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 frames
        frames_np = []
        for b64 in request.frames:
            frame = base64_to_frame(b64)
            frames_np.append(frame)

        # Stack: (T_in, H, W) -> (1, T_in, 1, H, W)
        frames_arr = np.stack(frames_np, axis=0)
        frames_tensor = torch.from_numpy(frames_arr).unsqueeze(0).unsqueeze(2).to(_device)

        # Predict
        with torch.no_grad():
            output = _model.predict_autoregressive(frames_tensor)

        predicted = output["predicted_frames"].cpu().numpy()[0]  # (T_out, 1, H, W)

        result = {
            "predicted_frames": frames_to_base64_list(predicted),
        }

        if request.return_latents:
            latents = output["predicted_latents"].cpu().numpy()[0]
            result["latent_vectors"] = latents.tolist()

        return PredictResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate a fresh wave sequence and predict."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        sim = WaveEquationSimulator(resolution=64, wave_speed=1.0, dt=0.1, boundary="reflecting")

        frames = sim.simulate(
            num_frames=20,
            center_x=request.center_x,
            center_y=request.center_y,
            width=request.width,
            amplitude=request.amplitude,
            rng=np.random.default_rng(42),
        )

        context = frames[:10]
        target = frames[10:]

        # Predict
        context_tensor = torch.from_numpy(context).unsqueeze(0).to(_device)
        with torch.no_grad():
            output = _model.predict_autoregressive(context_tensor)
        predicted = output["predicted_frames"].cpu().numpy()[0]

        # Metrics
        metrics = evaluate_sequence(predicted, target)

        return GenerateResponse(
            context_frames=frames_to_base64_list(context),
            predicted_frames=frames_to_base64_list(predicted),
            ground_truth_frames=frames_to_base64_list(target),
            psnr_per_step=metrics["psnr_per_step"],
            ssim_per_step=metrics["ssim_per_step"],
            params={
                "center_x": request.center_x,
                "center_y": request.center_y,
                "width": request.width,
                "amplitude": request.amplitude,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/samples", response_model=SamplesResponse)
async def get_samples():
    """Return pre-generated demo samples."""
    return SamplesResponse(
        samples=[SampleSequence(**s) for s in _demo_samples]
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Return training metrics."""
    if _metrics_cache is None:
        load_metrics_cache()

    if _metrics_cache is None:
        # Return empty/placeholder metrics
        return MetricsResponse(
            psnr_per_step=[],
            ssim_per_step=[],
            psnr_mean=0.0,
            ssim_mean=0.0,
        )

    return MetricsResponse(**_metrics_cache)


@router.get("/attention", response_model=AttentionResponse)
async def get_attention(sample_index: int = 0, layer: int = -1):
    """Return attention weights for visualization."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if sample_index >= len(_demo_samples):
        raise HTTPException(status_code=404, detail="Sample not found")

    try:
        sample = _demo_samples[sample_index]

        # Decode context frames
        ctx_frames = [base64_to_frame(b64) for b64 in sample["context_frames"]]
        ctx_arr = np.stack(ctx_frames, axis=0)
        ctx_tensor = torch.from_numpy(ctx_arr).unsqueeze(0).unsqueeze(2).to(_device)

        # Run with attention weights
        with torch.no_grad():
            output = _model(
                context_frames=ctx_tensor,
                target_frames=None,
                teacher_forcing_ratio=0.0,
                need_weights=True,
            )

        attn_weights = output["attention_weights"]
        if not attn_weights:
            raise HTTPException(status_code=500, detail="No attention weights available")

        # Select layer
        layer_idx = layer if layer >= 0 else len(attn_weights) + layer
        layer_idx = max(0, min(layer_idx, len(attn_weights) - 1))

        # Average over heads: (B, n_heads, T, T) -> (T, T)
        attn = attn_weights[layer_idx][0].mean(dim=0).cpu().numpy()

        predicted = output["predicted_frames"].cpu().numpy()[0]

        return AttentionResponse(
            attention_weights=attn.tolist(),
            context_frames=sample["context_frames"],
            predicted_frames=frames_to_base64_list(predicted),
            layer=layer_idx,
            num_heads=_model.transformer.decoder_layers[0].cross_attn.num_heads,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
