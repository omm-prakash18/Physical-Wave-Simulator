"""
FastAPI application for the Latent Video Prediction Engine.

Loads the trained model at startup and serves predictions via REST API.
"""

import os
import sys
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import HealthResponse
from api.routes import router, set_model, set_demo_samples, load_metrics_cache
from model.full_model import LatentVideoPredictor


# ─── Model loading ──────────────────────────────────────
def load_model(checkpoint_path: str | None = None) -> tuple:
    """Load the trained model from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints", "best_model.pt"
        )

    if not os.path.exists(checkpoint_path):
        print(f"WARNING: No checkpoint found at {checkpoint_path}")
        print("  Starting with untrained model for demo purposes.")
        # Create default model
        model = LatentVideoPredictor().to(device)
        model.eval()
        return model, device

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
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

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"  Model loaded: epoch={epoch}, val_loss={val_loss}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, device


# ─── App lifecycle ──────────────────────────────────────
_model_ref = {"model": None, "device": "cpu"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and demo data at startup."""
    print("=" * 60)
    print("Latent Video Prediction Engine — Starting up...")
    print("=" * 60)

    model, device = load_model()
    _model_ref["model"] = model
    _model_ref["device"] = device

    # Set model in routes
    set_model(model, device)

    # Generate demo samples
    print("Generating demo samples...")
    try:
        from api.demo_data import generate_demo_samples
        samples = generate_demo_samples(model, device, num_samples=5)
        set_demo_samples(samples)
        print(f"  Generated {len(samples)} demo samples")
    except Exception as e:
        print(f"  Warning: Could not generate demo samples: {e}")
        set_demo_samples([])

    # Load training metrics
    load_metrics_cache()

    print("=" * 60)
    print("Server ready!")
    print("=" * 60)

    yield

    print("Shutting down...")


# ─── App creation ───────────────────────────────────────
app = FastAPI(
    title="Latent Video Prediction Engine",
    description="CNN-Transformer hybrid model for predicting future frames of physical simulations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health endpoint ────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    model = _model_ref["model"]
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=_model_ref["device"],
        param_count=sum(p.numel() for p in model.parameters()) if model else 0,
    )


# ─── Mount API routes ──────────────────────────────────
app.include_router(router, prefix="/api")

# ─── Serve frontend static files (production) ──────────
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
