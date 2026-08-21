"""
FastAPI REST API integration test suite using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.demo_data import frame_to_base64
import numpy as np


@pytest.fixture
def client():
    """Test client with application lifespan management."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify GET /health returns status 200 and model metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "model_loaded" in json_data
    assert "device" in json_data
    assert "param_count" in json_data


def test_samples_endpoint(client):
    """Verify GET /api/samples returns demo evaluation sequences."""
    response = client.get("/api/samples")
    assert response.status_code == 200
    json_data = response.json()
    assert "samples" in json_data
    assert isinstance(json_data["samples"], list)


def test_metrics_endpoint(client):
    """Verify GET /api/metrics returns metric structure."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    json_data = response.json()
    assert "psnr_per_step" in json_data
    assert "ssim_per_step" in json_data
    assert "psnr_mean" in json_data
    assert "ssim_mean" in json_data


def test_predict_endpoint(client):
    """Verify POST /api/predict accepts 10 base64 frames and returns forecast."""
    # Generate 10 dummy base64 frames
    dummy_frame = np.zeros((64, 64), dtype=np.float32)
    b64_str = frame_to_base64(dummy_frame)
    payload = {
        "frames": [b64_str] * 10,
        "return_latents": True,
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_frames" in json_data
    assert len(json_data["predicted_frames"]) == 10
    assert "latent_vectors" in json_data


def test_generate_endpoint(client):
    """Verify POST /api/generate creates a wave simulation and returns predictions."""
    payload = {
        "center_x": 0.5,
        "center_y": 0.5,
        "width": 5.0,
        "amplitude": 1.0,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "context_frames" in json_data
    assert "predicted_frames" in json_data
    assert "ground_truth_frames" in json_data
    assert "psnr_per_step" in json_data
    assert len(json_data["context_frames"]) == 10
    assert len(json_data["predicted_frames"]) == 10


def test_attention_endpoint(client):
    """Verify GET /api/attention returns cross-attention weight maps when samples exist."""
    response = client.get("/api/attention?sample_index=0&layer=0")
    # If samples exist returns 200, if not 404
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        json_data = response.json()
        assert "attention_weights" in json_data
        assert "num_heads" in json_data


def test_benchmark_endpoint(client):
    """Verify GET /api/benchmark computes PyTorch vs ONNX latency comparison."""
    response = client.get("/api/benchmark?num_runs=2")
    assert response.status_code == 200
    json_data = response.json()
    assert "pytorch_avg_ms" in json_data
    assert "onnx_avg_ms" in json_data
    assert "speedup" in json_data
    assert "onnx_available" in json_data


def test_upload_analyze_endpoint(client):
    """Verify POST /api/upload-analyze accepts an image file and returns wave analysis."""
    import io
    from PIL import Image

    # Create dummy PNG image bytes
    img = Image.new("L", (100, 100), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test_wave.png", buf, "image/png")}
    response = client.post("/api/upload-analyze", files=files)
    if response.status_code != 200:
        print("UPLOAD ERROR:", response.json())
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["original_filename"] == "test_wave.png"
    assert "processed_frame" in json_data
    assert "latent_vector" in json_data
    assert len(json_data["latent_vector"]) == 256
    assert "predicted_frames" in json_data
    assert len(json_data["predicted_frames"]) == 10
    assert "estimated_params" in json_data
    assert "center_x" in json_data["estimated_params"]
    assert "spatial_spectrum" in json_data
    assert "spatial_vectors" in json_data
    assert len(json_data["spatial_vectors"]) == 64
    assert "dx" in json_data["spatial_vectors"][0]
    assert "dy" in json_data["spatial_vectors"][0]
    assert "feature_map_frame" in json_data


def test_predict_uncertainty_endpoint(client):
    """Verify POST /api/predict_uncertainty executes MC-Dropout forward passes and returns uncertainty heatmaps."""
    dummy_frame = np.zeros((64, 64), dtype=np.float32)
    b64_str = frame_to_base64(dummy_frame)
    payload = {
        "frames": [b64_str] * 10,
        "num_samples": 5,
    }

    response = client.post("/api/predict_uncertainty", json=payload)
    if response.status_code != 200:
        print("UNCERTAINTY ERROR:", response.json())
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_frames" in json_data
    assert len(json_data["predicted_frames"]) == 10
    assert "uncertainty_maps" in json_data
    assert len(json_data["uncertainty_maps"]) == 10
    assert "per_frame_uncertainty" in json_data
    assert len(json_data["per_frame_uncertainty"]) == 10
    assert "mean_uncertainty" in json_data
    assert json_data["num_samples"] == 5




