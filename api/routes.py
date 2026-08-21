"""
FastAPI route handlers for the Latent Video Prediction API.
"""

import os
import sys
import json
import numpy as np
import torch
import io
from PIL import Image
from fastapi import APIRouter, HTTPException, File, UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    PredictRequest, PredictResponse,
    GenerateRequest, GenerateResponse,
    SamplesResponse, SampleSequence,
    MetricsResponse,
    AttentionResponse,
    BenchmarkResponse,
    UploadAnalysisResponse,
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


@router.get("/benchmark", response_model=BenchmarkResponse)
async def get_benchmark(num_runs: int = 10):
    """Benchmark PyTorch vs ONNX Runtime inference latency."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import time
    onnx_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "model.onnx"
    )
    onnx_exists = os.path.exists(onnx_path)

    # Dummy input (B=1, T_in=10, C=1, H=64, W=64)
    dummy_input = torch.randn(1, 10, 1, 64, 64, device=_device)

    # Benchmark PyTorch
    pt_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = _model.predict_autoregressive(dummy_input)
            if _device == "cuda":
                torch.cuda.synchronize()
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    pt_avg = float(np.mean(pt_times))
    pt_p95 = float(np.percentile(pt_times, 95))

    ort_avg = pt_avg
    ort_p95 = pt_p95
    speedup = 1.0

    if onnx_exists:
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _device == "cuda" else ["CPUExecutionProvider"]
            session = ort.InferenceSession(onnx_path, providers=providers)
            input_name = session.get_inputs()[0].name
            ort_input = {input_name: dummy_input.cpu().numpy()}

            ort_times = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = session.run(None, ort_input)
                ort_times.append((time.perf_counter() - t0) * 1000.0)

            ort_avg = float(np.mean(ort_times))
            ort_p95 = float(np.percentile(ort_times, 95))
            speedup = float(pt_avg / max(ort_avg, 1e-6))
        except Exception:
            onnx_exists = False

    return BenchmarkResponse(
        pytorch_avg_ms=round(pt_avg, 2),
        pytorch_p95_ms=round(pt_p95, 2),
        onnx_avg_ms=round(ort_avg, 2),
        onnx_p95_ms=round(ort_p95, 2),
        speedup=round(speedup, 2),
        onnx_available=onnx_exists,
    )


@router.post("/upload-analyze", response_model=UploadAnalysisResponse)
async def upload_analyze(file: UploadFile = File(...)):
    """
    Accept an uploaded image file, preprocess it into a 64x64 grayscale wave frame,
    extract its CNN latent embedding, analyze spatial features, and predict future evolution.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("L")
        img_resized = img.resize((64, 64), Image.Resampling.BILINEAR)

        # Scale pixel values to [-1, 1]
        frame_np = np.array(img_resized).astype(np.float32) / 127.5 - 1.0

        # Physical feature analysis
        pos_frame = (frame_np + 1.0) / 2.0  # [0, 1]
        total_energy = float(np.sum(pos_frame ** 2))
        total_intensity = pos_frame.sum()

        if total_intensity > 1e-5:
            cy = float((np.sum(pos_frame, axis=1) * np.arange(64)).sum() / (total_intensity * 64.0))
            cx = float((np.sum(pos_frame, axis=0) * np.arange(64)).sum() / (total_intensity * 64.0))
        else:
            cx, cy = 0.5, 0.5

        center_x = max(0.1, min(0.9, cx))
        center_y = max(0.1, min(0.9, cy))
        amplitude = float((frame_np.max() - frame_np.min()) / 2.0)
        width = float(np.clip(np.std(pos_frame) * 15.0, 2.0, 10.0))

        # Radial spatial spectrum (16 bins)
        y_grid, x_grid = np.ogrid[:64, :64]
        r_grid = np.sqrt((x_grid - center_x * 64) ** 2 + (y_grid - center_y * 64) ** 2)
        r_bins = np.linspace(0, 45, 16)
        spectrum = []
        for idx in range(len(r_bins) - 1):
            mask = (r_grid >= r_bins[idx]) & (r_grid < r_bins[idx + 1])
            val = float(pos_frame[mask].mean()) if np.any(mask) else 0.0
            spectrum.append(round(val, 4))

        # Calculate 2D Spatial Energy Gradient Vector Field (8x8 grid)
        gy, gx = np.gradient(pos_frame)
        grid_size = 8
        cell_h = 64 // grid_size
        cell_w = 64 // grid_size
        spatial_vectors = []

        for i in range(grid_size):
            for j in range(grid_size):
                r_start, r_end = i * cell_h, (i + 1) * cell_h
                c_start, c_end = j * cell_w, (j + 1) * cell_w
                cell_gx = float(gx[r_start:r_end, c_start:c_end].mean())
                cell_gy = float(gy[r_start:r_end, c_start:c_end].mean())
                mag = float(np.sqrt(cell_gx**2 + cell_gy**2))
                angle = float(np.arctan2(cell_gy, cell_gx))
                spatial_vectors.append({
                    "x": round((j + 0.5) / grid_size, 3),
                    "y": round((i + 0.5) / grid_size, 3),
                    "dx": round(cell_gx, 4),
                    "dy": round(cell_gy, 4),
                    "magnitude": round(mag, 4),
                    "angle": round(angle, 3),
                })

        # Form context tensor for model (1, 10, 1, 64, 64) by repeating input frame
        context_arr = np.repeat(frame_np[np.newaxis, np.newaxis, ...], 10, axis=0)  # (10, 1, 64, 64)
        context_tensor = torch.from_numpy(context_arr).unsqueeze(0).to(_device)    # (1, 10, 1, 64, 64)

        b64_feat_map = None
        with torch.no_grad():
            single_tensor = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0).to(_device) # (1, 1, 64, 64)
            feat_stages = _model.encoder.extract_features(single_tensor)
            if feat_stages and len(feat_stages) > 1:
                feat_map = feat_stages[1][0].mean(dim=0).cpu().numpy()  # (16, 16) average feature map
                f_min, f_max = feat_map.min(), feat_map.max()
                feat_scaled = ((feat_map - f_min) / max(f_max - f_min, 1e-5) * 255.0).astype(np.uint8)
                feat_img = Image.fromarray(feat_scaled).resize((64, 64), Image.Resampling.BILINEAR)
                feat_buf = io.BytesIO()
                feat_img.save(feat_buf, format="PNG")
                b64_feat_map = base64.b64encode(feat_buf.getvalue()).decode("utf-8")

            latents_all = _model.encode_frames(context_tensor)  # (1, 10, d_model)
            latent_vec = latents_all[0, 0].cpu().numpy()
            latent_norm = float(np.linalg.norm(latent_vec))

            output = _model.predict_autoregressive(context_tensor)
            predicted_np = output["predicted_frames"].cpu().numpy()[0]  # (10, 1, 64, 64)

        b64_processed = frame_to_base64(frame_np)
        b64_predictions = frames_to_base64_list(predicted_np)

        return UploadAnalysisResponse(
            original_filename=file.filename or "uploaded_image.png",
            processed_frame=b64_processed,
            feature_map_frame=b64_feat_map,
            latent_vector=[round(float(v), 5) for v in latent_vec],
            latent_norm=round(latent_norm, 4),
            predicted_frames=b64_predictions,
            estimated_params={
                "center_x": round(center_x, 3),
                "center_y": round(center_y, 3),
                "amplitude": round(amplitude, 3),
                "width": round(width, 3),
                "total_energy": round(total_energy, 2),
            },
            spatial_spectrum=spectrum,
            spatial_vectors=spatial_vectors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")


