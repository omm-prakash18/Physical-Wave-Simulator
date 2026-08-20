"""
Pydantic request/response schemas for the API.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
    param_count: int = 0


class PredictRequest(BaseModel):
    """Request for /api/predict."""
    frames: list[str] = Field(
        ...,
        description="List of base64-encoded grayscale frame images (T_in frames).",
        min_length=1,
    )
    return_latents: bool = Field(default=False, description="Include latent vectors in response.")


class PredictResponse(BaseModel):
    """Response from /api/predict."""
    predicted_frames: list[str]  # base64-encoded frames
    latent_vectors: list[list[float]] | None = None  # (T_out, d_model)
    psnr_per_step: list[float] | None = None
    ssim_per_step: list[float] | None = None


class GenerateRequest(BaseModel):
    """Request for /api/generate."""
    center_x: float = Field(default=0.5, ge=0.1, le=0.9)
    center_y: float = Field(default=0.5, ge=0.1, le=0.9)
    width: float = Field(default=5.0, ge=2.0, le=10.0)
    amplitude: float = Field(default=0.8, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    """Response from /api/generate."""
    context_frames: list[str]     # base64 context
    predicted_frames: list[str]   # base64 predictions
    ground_truth_frames: list[str]  # base64 GT
    psnr_per_step: list[float]
    ssim_per_step: list[float]
    params: dict


class SampleSequence(BaseModel):
    """A single demo sample."""
    id: int
    context_frames: list[str]
    predicted_frames: list[str]
    ground_truth_frames: list[str]
    psnr_per_step: list[float]
    ssim_per_step: list[float]
    params: dict


class SamplesResponse(BaseModel):
    """Response from /api/samples."""
    samples: list[SampleSequence]


class MetricsResponse(BaseModel):
    """Response from /api/metrics."""
    psnr_per_step: list[float]
    ssim_per_step: list[float]
    psnr_mean: float
    ssim_mean: float
    loss_history: dict | None = None
    training_config: dict | None = None
    param_counts: dict | None = None


class AttentionRequest(BaseModel):
    """Query params for /api/attention."""
    sample_index: int = Field(default=0, ge=0)
    layer: int = Field(default=-1, description="Layer index, -1 for last layer.")


class AttentionResponse(BaseModel):
    """Response from /api/attention."""
    attention_weights: list[list[float]]  # (T_total, T_total) — averaged over heads
    context_frames: list[str]  # base64 thumbnails
    predicted_frames: list[str]
    layer: int
    num_heads: int


class BenchmarkResponse(BaseModel):
    """Response from /api/benchmark latency comparison."""
    pytorch_avg_ms: float
    pytorch_p95_ms: float
    onnx_avg_ms: float
    onnx_p95_ms: float
    speedup: float
    onnx_available: bool

