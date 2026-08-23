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
    energy_gt: list[float] | None = None
    energy_pred: list[float] | None = None
    psnr_persistence: list[float] | None = None
    ssim_persistence: list[float] | None = None
    energy_persistence: list[float] | None = None
    psnr_linear: list[float] | None = None
    ssim_linear: list[float] | None = None
    energy_linear: list[float] | None = None


class SampleSequence(BaseModel):
    """A single demo sample."""
    id: int
    context_frames: list[str]
    predicted_frames: list[str]
    ground_truth_frames: list[str]
    psnr_per_step: list[float]
    ssim_per_step: list[float]
    params: dict
    energy_gt: list[float] | None = None
    energy_pred: list[float] | None = None
    psnr_persistence: list[float] | None = None
    ssim_persistence: list[float] | None = None
    energy_persistence: list[float] | None = None
    psnr_linear: list[float] | None = None
    ssim_linear: list[float] | None = None
    energy_linear: list[float] | None = None


class SamplesResponse(BaseModel):
    """Response from /api/samples."""
    samples: list[SampleSequence]


class LatentProbeResponse(BaseModel):
    """Response from /api/latent-probe."""
    r2_center_x: float
    r2_center_y: float
    r2_width: float
    r2_amplitude: float
    num_samples: int



class MetricsResponse(BaseModel):
    """Response from /api/metrics."""
    psnr_per_step: list[float]
    ssim_per_step: list[float]
    psnr_mean: float
    ssim_mean: float
    loss_history: list[dict] | dict | None = None
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


class UploadAnalysisResponse(BaseModel):
    """Response from /api/upload-analyze image upload endpoint."""
    original_filename: str
    processed_frame: str             # Base64 grayscale 64x64 frame
    feature_map_frame: str | None = None # Base64 2D spatial feature activation map from CNN
    latent_vector: list[float]       # 256-dim CNN encoder output
    latent_norm: float               # L2 norm of latent vector
    predicted_frames: list[str]      # Base64 predicted 10 frames
    estimated_params: dict           # center_x, center_y, amplitude, width, total_energy
    spatial_spectrum: list[float]    # Spatial energy distribution
    spatial_vectors: list[dict]      # List of {x, y, dx, dy, magnitude, angle} gradient vectors


class PredictUncertaintyRequest(BaseModel):
    """Request for /api/predict_uncertainty."""
    frames: list[str] = Field(..., description="List of base64-encoded grayscale context frame images.", min_length=1)
    num_samples: int = Field(default=20, ge=2, le=50, description="Number of stochastic MC-Dropout forward passes.")


class PredictUncertaintyResponse(BaseModel):
    """Response from /api/predict_uncertainty."""
    predicted_frames: list[str]      # Base64 mean forecast frames
    uncertainty_maps: list[str]      # Base64 std deviation heatmaps per frame
    per_frame_uncertainty: list[float] # Average std deviation per predicted step
    mean_uncertainty: float           # Overall sequence epistemic uncertainty
    num_samples: int                 # Number of stochastic samples run




