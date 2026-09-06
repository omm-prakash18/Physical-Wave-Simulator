# 🌊 Hybrid CNN-Transformer Latent Video Prediction Studio

A full-stack machine learning system that forecasts future frames of physical simulations (2D wave equation propagation) by:
1. **Compressing** high-dimensional spatial frames into latent tokens using a 4-stage Residual CNN Encoder
2. **Forecasting** future latent sequence trajectories with a 6-layer Encoder-Decoder Causal Transformer
3. **Reconstructing** predicted latent vectors back to pixel space using a 4-stage Residual CNN Decoder

Built with **PyTorch**, **FastAPI**, **React 18**, **Tailwind CSS**, and an elegant **warm professional studio theme** (Outfit + Source Sans 3 + Fira Code).

----

## 🎨 Visual Identity & UI/UX Design

The application features a warm, production-grade professional aesthetic:
- **Color Palette**: Deep warm navy canvas (`#1a1a2e`), cream-tinted glass panels, muted gold (`#d4a853`) and soft rose (`#c97b7b`) accents, sage green (`#7eb09b`) success states.
- **Typography**:
  - Headings: **Outfit** (clean geometric display sans)
  - Body: **Source Sans 3** (humanist sans-serif)
  - Monospace & Metrics: **Fira Code** (ligature-enabled monospace)
- **Micro-Interactions**: Ambient glowing backdrop fields, custom range scrubbers, synchronized frame players, and interactive UMAP/Attention heatmaps.

---

## 🏛️ Architecture Overview

```
Input Frames (B, T_in, 1, 64, 64)
         │
    ┌────┴────┐
    │ Residual│  4-stage Conv2d + ResNet shortcuts + Dropout2d
    │   CNN   │  64→32→16→8→4, Adaptive Pooling, Linear
    │ Encoder │  ~455K parameters
    └────┬────┘
         │
  Latent Tokens (B, T_in, 256)
         │
    ┌────┴─────────┐
    │  Encoder-    │  6 Encoder + 6 Decoder layers, 8 Attention Heads
    │  Decoder     │  Learned query tokens for T_out prediction horizon
    │ Transformer  │  Pre-norm LayerNorm, GELU Feed-Forward Networks
    │              │  ~11.1M parameters
    └────┬─────────┘
         │
  Predicted Latents (B, T_out, 256)
         │
    ┌────┴────┐
    │ Residual│  4-stage ConvTranspose2d + ResNet shortcuts
    │   CNN   │  Linear projection → 4→8→16→32→64, Conv + Tanh
    │ Decoder │  ~1.75M parameters
    └────┬────┘
         │
  Predicted Frames (B, T_out, 1, 64, 64)
```

**Total Parameters**: **~13.3M**

---

## ⚡ Key Highlights & Optimizations

- **Residual Connections**: 1×1 convolutional shortcut projections in encoder and decoder blocks maintain spatial fidelity and prevent vanishing gradients.
- **Normalized Data Pipeline**: Dynamic frame scaling to `[-1, 1]` aligned with the decoder's Tanh activation layer.
- **Zero-Initialized Transformer Projection**: Output projection bias initialized to zero to stabilize initial training steps.
- **Multi-Component Loss Function**:
  - $\mathcal{L}_{\text{recon}}$: L1 pixel reconstruction loss
  - $\mathcal{L}_{\text{latent}}$: Latent space MSE
  - $\mathcal{L}_{\text{perceptual}}$: Feature-space L1 using frozen encoder stages
  - $\mathcal{L}_{\text{temporal}}$: Frame-to-frame motion consistency
- **Automated Testing Suite (`pytest`)**: 17 comprehensive unit & integration tests covering tensor shape invariants, loss function calculations, wave PDE simulation bounds, and FastAPI endpoint routes.
- **MLOps & ONNX Export**: Graph compilation to ONNX format via `training/export_onnx.py` with ONNX Runtime backend serving & live latency benchmarking (`/api/benchmark`).
- **Production Containerization**: Multi-stage `Dockerfile` (React build + FastAPI Python runtime) and single-command orchestration via `docker-compose.yml`.
- **GitHub Actions CI/CD Pipeline**: Continuous integration workflow (`.github/workflows/ci.yml`) enforcing automated test suites and frontend production build checks.


---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA (recommended)

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/omm-prakash18/Physical-Wave-Simulator.git
cd Physical-Wave-Simulator

# Install Python dependencies (CUDA 12.6 PyTorch example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 3. Generate Simulation Data

```bash
python -m data.generate_data --num_sequences 10000 --output_dir data/wave_64
```

### 4. Train the Model

```bash
python -m training.train
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir runs
```

### 5. Launch Full Stack

```bash
# Terminal 1: FastAPI Backend Server
python -u api/main.py

# Terminal 2: React Vite Dev Server
cd frontend
npm run dev
```

Open **http://localhost:5173** to view the application.

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server status, GPU device, and model parameter count |
| `/api/predict` | POST | Forecast future frames from input context sequence |
| `/api/predict_uncertainty` | POST | MC-Dropout stochastic inference for epistemic uncertainty maps |
| `/api/upload-analyze` | POST | Custom picture upload analysis, 256-d latent extraction, and 2D spatial feature mapping |
| `/api/generate` | POST | Generate custom simulation and model prediction |
| `/api/samples` | GET | Pre-generated evaluation sequences |
| `/api/metrics` | GET | Training metrics, loss components, and PSNR curves |
| `/api/attention` | GET | Multi-head cross-attention weight maps |
| `/api/benchmark` | GET | PyTorch vs ONNX Runtime latency comparison |

---

## 🖥️ Studio Pages

1. **Overview**: Key metrics (PSNR, SSIM, parameter count), live context vs prediction comparison, MC-Dropout epistemic uncertainty heatmaps, and interactive per-step PSNR bar chart.
2. **Playground**: Real-time simulation parameter sliders (Pulse X/Y, Width, Amplitude) with instant model inference.
3. **Image Upload**: Custom picture upload & 3-Stage Model Architecture Pipeline visualizer.
4. **Latent Space**: 2D UMAP projection of frame tokens colored along temporal trajectories (gold → rose).
5. **Attention Viz**: Layer-by-layer cross-attention heatmaps mapping context frames to predicted timesteps.
6. **Training Dashboard**: Loss component breakdowns ($\mathcal{L}_{\text{recon}}$, $\mathcal{L}_{\text{latent}}$, $\mathcal{L}_{\text{perceptual}}$, $\mathcal{L}_{\text{temporal}}$) and rollout quality degradation plots.
7. **Model Card**: Full technical specification, loss formulas, dataset specs, and tech stack details.

---

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, TorchVision, NumPy, SciPy
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend App**: React 18, Vite, Tailwind CSS, Recharts
- **Design Tokens**: Outfit, Source Sans 3, Fira Code
- **Logging & Viz**: TensorBoard, PIL, Matplotlib

---

## 📄 License

MIT License — free for open-source research and educational development.
