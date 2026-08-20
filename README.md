# Hybrid CNN-Transformer Latent Video Prediction Engine

A full-stack ML system that predicts future frames of a **2D wave equation** simulation by:
1. **Encoding** frames into a latent space with a CNN
2. **Forecasting** the latent sequence with a causal Transformer
3. **Decoding** predicted latents back to pixel space

**Dataset**: 2D Wave Equation (Gaussian pulse propagation with reflecting boundaries)
**Configurable via**: `config.py` — swap to `"balls"` or `"fluid"` datasets.

---

## Architecture

```
Input Frames (B, T_in, 1, 64, 64)
         │
    ┌────┴────┐
    │   CNN   │  4-stage Conv-BN-SiLU, stride 2
    │ Encoder │  64→32→16→8→4, GAP, Linear
    └────┬────┘
         │
  Latent Tokens (B, T_in, 256)
         │
    ┌────┴─────────┐
    │    Causal     │  6 layers, 8 heads, d=256
    │  Transformer  │  Learned query tokens
    └────┬─────────┘
         │
  Predicted Latents (B, T_out, 256)
         │
    ┌────┴────┐
    │   CNN   │  ConvTranspose ×4, Tanh
    │ Decoder │
    └────┬────┘
         │
  Predicted Frames (B, T_out, 1, 64, 64)
```

**Parameters**: ~8.5M total (Encoder ~178K, Transformer ~8.1M, Decoder ~188K)

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA (recommended)

### 1. Install dependencies

```bash
# Python (install CUDA PyTorch first if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

### 2. Generate training data

```bash
python -m data.generate_data --num_sequences 10000 --output_dir data/wave_64
```

This generates 10,000 wave equation sequences (20 frames each, 64×64) split into train/val/test.

### 3. Train the model

```bash
python -m training.train
```

Training logs are saved to `runs/` (TensorBoard) and checkpoints to `checkpoints/`.

Monitor training:
```bash
tensorboard --logdir runs
```

**Expected training time**: ~2–4 hours on RTX 3050 (6GB).

### 4. Evaluate

```bash
python -m training.evaluate
```

Produces PSNR/SSIM curves and sample GIFs in `eval_output/`.

### 5. Run the full stack

```bash
# Terminal 1: API server
python -m api.main

# Terminal 2: Frontend dev server
cd frontend && npm run dev
```

Open http://localhost:5173 to access the frontend.

---

## Loss Function

```
L_total = L_recon + L_latent + 0.1 × L_perceptual + 0.5 × L_temporal
```

| Component | Description |
|-----------|-------------|
| `L_recon` | L1 pixel loss between predicted and GT frames |
| `L_latent` | MSE in latent space (primary Transformer signal) |
| `L_perceptual` | Feature-space L1 using frozen encoder — prevents blurring |
| `L_temporal` | Motion consistency — penalizes flicker between frames |

All components are logged separately to TensorBoard.

---

## Training Details

- **Optimizer**: AdamW, lr=3e-4, cosine decay + 500-step linear warmup
- **Mixed precision**: FP16 with gradient scaling
- **Batch size**: 16 with gradient accumulation of 2 (effective 32)
- **Scheduled sampling**: Teacher forcing ratio decays 1.0 → 0.0 over training
- **Joint training**: Encoder, Transformer, and Decoder trained together (not pretrained separately)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health + model status |
| `/api/predict` | POST | Predict future frames from context |
| `/api/generate` | POST | Generate fresh simulation + predict |
| `/api/samples` | GET | Pre-generated demo sequences |
| `/api/metrics` | GET | Training metrics (PSNR/SSIM, loss) |
| `/api/attention` | GET | Transformer attention weights |

---

## Frontend Pages

1. **Overview** — Hero, live demo, key stats
2. **Playground** — Interactive parameter controls + live prediction
3. **Latent Explorer** — UMAP projection of latent tokens
4. **Attention Viz** — Transformer attention heatmap
5. **Training Dashboard** — Loss curves, PSNR/SSIM charts
6. **Model Card** — Architecture, parameters, config

The frontend gracefully degrades to demo mode when the backend is offline.

---

## Project Structure

```
cnn/
├── config.py                   # All hyperparameters
├── requirements.txt
├── data/
│   ├── generator.py            # Wave equation simulator
│   ├── dataset.py              # PyTorch Dataset
│   └── generate_data.py        # CLI data generation
├── model/
│   ├── encoder.py              # CNN Encoder
│   ├── decoder.py              # CNN Decoder
│   ├── transformer.py          # Causal Transformer
│   ├── full_model.py           # Composed model
│   └── losses.py               # Loss components
├── training/
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Evaluation + GIFs
│   └── scheduler.py            # LR + teacher forcing
├── api/
│   ├── main.py                 # FastAPI app
│   ├── routes.py               # Endpoints
│   ├── schemas.py              # Pydantic models
│   └── demo_data.py            # Demo sequence generation
├── frontend/                   # React + Vite + Tailwind
│   └── src/
│       ├── pages/              # 6 page components
│       ├── components/         # Shared UI components
│       └── api/                # Backend client
├── checkpoints/                # Model saves
└── runs/                       # TensorBoard logs
```

---

## Evaluation Metrics

- **PSNR / SSIM** per rollout step (expect degradation — this is standard)
- **Latent MSE** as primary training health metric
- **Energy conservation error** (physics validation)
- Qualitative side-by-side GIFs
