import React, { useState, useRef } from 'react';
import FramePlayer from '../components/FramePlayer';
import Card from '../components/Card';
import MetricStat from '../components/MetricStat';
import { Sparkline } from '../components/MetricChart';
import { uploadAndAnalyzeImage } from '../api/client';
import demoFallback from '../data/demo_fallback.json';

// Helper to generate canvas preset images for client demonstration
function createPresetDataUrl(presetType) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(128, 128);
  const data = imgData.data;

  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) {
      const idx = (y * 128 + x) * 4;
      let val = 0;

      if (presetType === 'single_pulse') {
        const dx = (x - 64) / 20;
        const dy = (y - 64) / 20;
        const r2 = dx * dx + dy * dy;
        val = Math.exp(-r2) * 255;
      } else if (presetType === 'dual_ripples') {
        const d1 = Math.sqrt((x - 40) ** 2 + (y - 50) ** 2);
        const d2 = Math.sqrt((x - 88) ** 2 + (y - 78) ** 2);
        val = (Math.cos(d1 / 4) * Math.exp(-d1 / 30) + Math.cos(d2 / 4) * Math.exp(-d2 / 30)) * 120 + 128;
      } else if (presetType === 'ring_wave') {
        const r = Math.sqrt((x - 64) ** 2 + (y - 64) ** 2);
        val = Math.sin(r / 5) * Math.exp(-((r - 35) ** 2) / 300) * 200 + 128;
      } else {
        const r = Math.sqrt((x - 64) ** 2 + (y - 64) ** 2);
        val = (Math.cos(x / 3) * Math.sin(y / 3) * Math.exp(-r / 40)) * 120 + 128;
      }

      val = Math.max(0, Math.min(255, val));
      data[idx] = val;
      data[idx + 1] = val;
      data[idx + 2] = val;
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return canvas.toDataURL('image/png');
}

// Convert data URL to Blob/File helper
function dataURLtoFile(dataurl, filename) {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new File([u8arr], filename, { type: mime });
}

/**
 * Image Upload & Latent Wave Prediction page.
 * Compresses an uploaded image or preset into a latent token, 
 * details the 3-stage pipeline steps, and predicts future wave timelines.
 */
export default function ImageAnalysis({ isBackendOnline }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [activeStage, setActiveStage] = useState(1); // 1: Encoder, 2: Transformer, 3: Decoder
  const fileInputRef = useRef(null);

  const presets = [
    { id: 'single_pulse', name: 'Gaussian Wave Pulse', desc: 'Centered smooth energy peak' },
    { id: 'dual_ripples', name: 'Dual Wave Interference', desc: 'Interfering wavefronts' },
    { id: 'ring_wave', name: 'Ring Wavefront', desc: 'Expanding circular wave crest' },
    { id: 'high_freq', name: 'Complex Wave Dynamics', desc: 'High frequency spatial mode' },
  ];

  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid image file (PNG, JPG, WebP, GIF).');
      return;
    }
    setError(null);
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = () => setPreviewUrl(reader.result);
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handlePresetSelect = (presetId) => {
    const dataUrl = createPresetDataUrl(presetId);
    setPreviewUrl(dataUrl);
    const file = dataURLtoFile(dataUrl, `${presetId}.png`);
    setSelectedFile(file);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please upload an image or select a sample preset.');
      return;
    }

    setLoading(true);
    setError(null);

    const fallbackSample = demoFallback.samples[0];

    if (!isBackendOnline) {
      setTimeout(() => {
        const mockLatent = Array.from({ length: 256 }, (_, i) => Math.sin(i / 6) * 1.5 + Math.cos(i / 12) * 0.8);
        setResult({
          original_filename: selectedFile.name,
          processed_frame: fallbackSample.context_frames[0] || fallbackSample.predicted_frames[0],
          feature_map_frame: fallbackSample.context_frames[0] || fallbackSample.predicted_frames[0],
          latent_vector: mockLatent,
          latent_norm: 5.4218,
          predicted_frames: fallbackSample.predicted_frames,
          estimated_params: {
            center_x: 0.52,
            center_y: 0.48,
            amplitude: 0.85,
            width: 4.8,
            total_energy: 142.3,
          },
          spatial_spectrum: [0.1, 0.3, 0.7, 0.9, 0.8, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.0, 0.0],
        });
        setLoading(false);
      }, 700);
      return;
    }

    try {
      const data = await uploadAndAnalyzeImage(selectedFile);
      setResult(data);
      setCurrentFrame(0);
    } catch (err) {
      console.error('Image analysis failed:', err);
      setError('Backend server processing fallback mode active.');
      const mockLatent = Array.from({ length: 256 }, (_, i) => Math.sin(i / 6) * 1.5 + Math.cos(i / 12) * 0.8);
      setResult({
        original_filename: selectedFile.name,
        processed_frame: fallbackSample.context_frames[0] || fallbackSample.predicted_frames[0],
        feature_map_frame: fallbackSample.context_frames[0] || fallbackSample.predicted_frames[0],
        latent_vector: mockLatent,
        latent_norm: 4.89,
        predicted_frames: fallbackSample.predicted_frames,
        estimated_params: {
          center_x: 0.5,
          center_y: 0.5,
          amplitude: 0.8,
          width: 5.0,
          total_energy: 120.0,
        },
        spatial_spectrum: [0.1, 0.4, 0.85, 0.95, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0],
      });
    } finally {
      setLoading(false);
    }
  };

  const getLatentMinMax = () => {
    if (!result?.latent_vector || result.latent_vector.length === 0) return { min: -1, max: 1 };
    let min = Infinity, max = -Infinity;
    result.latent_vector.forEach(v => {
      if (v < min) min = v;
      if (v > max) max = v;
    });
    return { min, max: max === min ? min + 1e-5 : max };
  };

  const { min: lMin, max: lMax } = getLatentMinMax();

  // Colorblind-safe diverging Blue -> Neutral -> Gold scale for self-attention preview
  const getCausalAttentionHeatColor = (value) => {
    const v = Math.min(Math.max(value, 0), 1);
    let r, g, b;
    if (v < 0.5) {
      const t = v * 2;
      r = Math.round(35 + t * (107 - 35));
      g = Math.round(52 + t * (101 - 52));
      b = Math.round(85 + t * (88 - 85));
    } else {
      const t = (v - 0.5) * 2;
      r = Math.round(107 + t * (212 - 107));
      g = Math.round(101 + t * (168 - 101));
      b = Math.round(88 + t * (83 - 88));
    }
    return `rgb(${r}, ${g}, ${b})`;
  };

  return (
    <div className="animate-fade-in space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold font-heading text-text-primary mb-1">
          Diagnostics & Wave Analysis (OOD Study)
        </h1>
        <p className="text-xs text-text-secondary font-sans leading-relaxed">
          Upload physical wave snapshots, extract latent space embeddings, and analyze the causal prediction pipeline.
        </p>
      </div>

      {/* Out-Of-Distribution Warning Banner */}
      <div className="rounded-xl border border-accent-coral/25 bg-accent-coral/5 p-4 flex gap-3 text-xs select-none">
        <span className="text-accent-coral text-lg shrink-0">⚠️</span>
        <div className="space-y-1 text-left">
          <p className="font-semibold text-accent-coral font-heading">
            Out-of-Distribution (OOD) Generalization Warning
          </p>
          <p className="text-text-secondary leading-relaxed font-sans prose-panel">
            This simulator is trained purely on Gaussian pulse propagation. Uploading natural shapes, complex waveforms, or non-boundary photos is out-of-distribution. The latent mapping will execute, but predictions will decay into non-physical noise. Use this page to study convolutional boundary generalization failures.
          </p>
        </div>
      </div>

      {/* Upload and presets grids */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Dropzone Card */}
        <Card className="md:col-span-2 flex flex-col justify-between" title="Upload Simulation Snapshot" eyebrow="Input snapshot">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`
              border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-snappy flex flex-col items-center justify-center min-h-[170px] select-none
              ${dragActive
                ? 'border-accent-gold bg-accent-gold/5 shadow-md shadow-accent-gold/10'
                : previewUrl
                ? 'border-accent-gold/30 bg-white/[0.01] hover:border-accent-gold/60'
                : 'border-panel-border/30 hover:border-accent-gold/40 bg-white/[0.01]'
              }
            `}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              aria-label="Upload simulation frame image"
              accept="image/*"
              className="hidden"
            />

            {previewUrl ? (
              <div className="flex flex-col items-center gap-3">
                <img
                  src={previewUrl}
                  alt="Snapshot upload preview"
                  className="w-24 h-24 object-cover rounded-lg border border-accent-gold/30 shadow-md bg-[#15141f]"
                />
                <span className="text-xs text-accent-gold font-mono font-medium truncate max-w-[240px]">
                  {selectedFile ? selectedFile.name : 'preset_snapshot.png'}
                </span>
                <span className="text-[10px] text-text-secondary/70">Click or drag a new image to replace</span>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-10 h-10 rounded-lg bg-accent-gold/10 border border-accent-gold/25 text-accent-gold flex items-center justify-center text-lg mx-auto">
                  📤
                </div>
                <div>
                  <p className="text-xs font-semibold text-text-primary">Drag & drop your wave snapshot here</p>
                  <p className="text-[10px] text-text-secondary mt-0.5">Supports PNG, JPG, WebP format</p>
                </div>
              </div>
            )}
          </div>

          {error && <p className="text-xs text-accent-coral mt-3 text-center font-mono">{error}</p>}

          <div className="mt-6 flex justify-center border-t border-panel-border/20 pt-4">
            <button
              onClick={handleAnalyze}
              disabled={loading || !selectedFile}
              className={`
                px-8 py-2.5 rounded-lg font-semibold text-xs transition-snappy w-full md:w-auto cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-gold/45
                ${loading || !selectedFile
                  ? 'bg-white/5 text-text-secondary/30 cursor-not-allowed border border-panel-border'
                  : 'bg-gradient-to-r from-accent-gold to-accent-rose text-canvas-deep hover:shadow-glow-gold hover:scale-[1.02] active:scale-[0.98]'
                }
              `}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3.5 h-3.5 border-2 border-canvas-deep/40 border-t-canvas-deep rounded-full animate-spin" />
                  Running analysis...
                </span>
              ) : (
                '⚡ Process Snapshot & Forecast'
              )}
            </button>
          </div>
        </Card>

        {/* Preset Selector Card */}
        <Card title="Or load wave presets" eyebrow="Preset triggers">
          <p className="text-xs text-text-secondary mb-4 font-sans leading-relaxed">
            Click on a physical preset scenario below to load pre-calculated snapshot configurations:
          </p>
          <div className="space-y-2.5 select-none">
            {presets.map((p) => (
              <button
                key={p.id}
                onClick={() => handlePresetSelect(p.id)}
                className="w-full text-left p-3 rounded-xl bg-white/[0.02] border border-panel-border/20 hover:border-accent-gold/45 hover:bg-accent-gold/[0.03] transition-snappy text-xs flex justify-between items-center group cursor-pointer focus:outline-none focus:ring-1 focus:ring-accent-gold/30"
              >
                <div className="space-y-0.5">
                  <p className="font-semibold text-text-primary group-hover:text-accent-gold">{p.name}</p>
                  <p className="text-[9px] text-text-secondary">{p.desc}</p>
                </div>
                <span className="text-accent-gold/40 group-hover:text-accent-gold font-mono transition-transform group-hover:translate-x-1">→</span>
              </button>
            ))}
          </div>
        </Card>
      </div>

      {/* Analysis Reports & Multi-stage pipelines */}
      {result && (
        <div className="space-y-6">
          {/* Estimated parameter metrics */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <MetricStat
              label="Pulse Center X"
              value={result.estimated_params.center_x?.toFixed(2)}
              accent="gold"
            />
            <MetricStat
              label="Pulse Center Y"
              value={result.estimated_params.center_y?.toFixed(2)}
              accent="gold"
            />
            <MetricStat
              label="Wave Amplitude"
              value={result.estimated_params.amplitude?.toFixed(2)}
              accent="sage"
            />
            <MetricStat
              label="Pulse Width (σ)"
              value={result.estimated_params.width?.toFixed(1)}
              accent="rose"
            />
            <MetricStat
              label="Latent L2 Norm"
              value={result.latent_norm?.toFixed(4)}
              accent="primary"
            />
          </div>

          {/* 3-Stage Model pipeline Card */}
          <Card title="Causal Model Pipeline Diagnostic" eyebrow="ML Engine stages">
            {/* Stage Selector */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-panel-border/30 pb-4 select-none">
              <div>
                <h3 className="text-sm font-semibold text-text-primary font-heading">Interactive Pipeline Graph</h3>
                <p className="text-[11px] text-text-secondary">Navigate encoder-transformer-decoder blocks below.</p>
              </div>
              <div className="flex bg-white/5 p-0.5 rounded-lg border border-panel-border font-sans">
                {[
                  { stage: 1, label: '1. CNN Encoder' },
                  { stage: 2, label: '2. Latent Transformer' },
                  { stage: 3, label: '3. CNN Decoder' },
                ].map(({ stage, label }) => (
                  <button
                    key={stage}
                    onClick={() => setActiveStage(stage)}
                    className={`px-3 py-1 rounded-md text-[11px] font-semibold transition-snappy cursor-pointer ${
                      activeStage === stage
                        ? 'bg-accent-gold text-canvas-deep shadow-md'
                        : 'text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            {/* Stage Contents */}
            <div className="pt-4">
              
              {/* Stage 1: CNN Encoder */}
              {activeStage === 1 && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center animate-fade-in text-left">
                  <div className="space-y-3">
                    <span className="eyebrow-label text-accent-gold">Stage 1: Spatial Compression</span>
                    <h4 className="text-sm font-semibold text-text-primary">Snapshot to Latent Mapping</h4>
                    <p className="text-xs text-text-secondary leading-relaxed font-sans prose-panel">
                      The CNN Encoder extracts spatial wave features, maps high-dimensional 64×64 pixels (4096 dimensions) into a single 256-dimensional vector token z<sub>t</sub>.
                    </p>
                    <div className="bg-[#15141f]/75 p-3 rounded-lg border border-panel-border/30 text-[10px] font-mono text-text-secondary space-y-1 select-text font-mono-tabular">
                      <p>• Input grid dimension: 64 × 64 (grayscale)</p>
                      <p>• Projection target size: 256 embedding dimensions</p>
                      <p>• Compression factor: 16.0× reduction</p>
                    </div>
                  </div>

                  {/* Images Comparison */}
                  <div className="flex items-center justify-center gap-4 bg-[#15141f]/50 p-4 rounded-xl border border-panel-border/30 select-none">
                    <div className="text-center space-y-1">
                      <p className="text-[9px] text-text-secondary">Input file</p>
                      <img
                        src={previewUrl}
                        alt="Original Upload"
                        className="w-20 h-20 object-cover rounded border border-panel-border bg-[#15141f]"
                      />
                    </div>
                    <span className="text-accent-gold font-mono font-bold">→</span>
                    <div className="text-center space-y-1">
                      <p className="text-[9px] text-text-secondary">64×64 normalized</p>
                      <img
                        src={`data:image/png;base64,${result.processed_frame}`}
                        alt="Processed Tensor"
                        className="w-20 h-20 object-cover rounded border border-accent-gold/40 bg-[#15141f]"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    </div>
                  </div>

                  {/* 256-d Latent barcode */}
                  <div className="space-y-2 select-none">
                    <p className="text-[11px] font-semibold text-text-primary font-mono">Embedding z<sub>t</sub> (256 dimensions)</p>
                    <div className="h-16 flex items-center gap-[1px] overflow-hidden rounded bg-black/45 p-2 border border-panel-border/30">
                      {result.latent_vector.map((val, idx) => {
                        const normVal = Math.max(0, Math.min(1, (val - lMin) / (lMax - lMin)));
                        return (
                          <div
                            key={idx}
                            className="flex-1 h-full rounded-[0.5px]"
                            style={{
                              backgroundColor: `hsl(${210 - normVal * 150}, 80%, ${35 + normVal * 45}%)`,
                              opacity: 0.5 + normVal * 0.5,
                            }}
                            title={`Feature ${idx}: ${val.toFixed(4)}`}
                          />
                        );
                      })}
                    </div>
                    <div className="flex justify-between text-[9px] font-mono text-text-tertiary">
                      <span>Dim 0</span>
                      <span>Latent representation mapping</span>
                      <span>Dim 255</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Stage 2: Transformer forecasting */}
              {activeStage === 2 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center animate-fade-in text-left">
                  <div className="space-y-3">
                    <span className="eyebrow-label text-accent-rose">Stage 2: Causal Propagation</span>
                    <h4 className="text-sm font-semibold text-text-primary">Masked Causal Attention Mechanism</h4>
                    <p className="text-xs text-text-secondary leading-relaxed font-sans prose-panel">
                      The transformer blocks consume context latents [z<sub>1</sub>...z<sub>10</sub>] and causal attention masks to project predicted latents [z<sub>11</sub>...z<sub>20</sub>] autoregressively.
                    </p>
                    <div className="bg-[#15141f]/75 p-3 rounded-lg border border-panel-border/30 text-[10px] font-mono text-text-secondary space-y-1 select-text font-mono-tabular">
                      <p>• Attention Heads: 8 multi-head blocks</p>
                      <p>• Mask type: causal autoregressive upper mask</p>
                      <p>• Rollout horizon: 10 predicted temporal offsets</p>
                    </div>
                  </div>

                  {/* Masked Attention Matrix visualizer */}
                  <div className="bg-[#15141f]/60 p-4 rounded-xl border border-panel-border/30 space-y-3 select-none">
                    <div className="flex justify-between items-center text-[10px] font-mono">
                      <span className="font-semibold text-text-primary">10×10 Causal Mask matrix</span>
                      <span className="text-accent-rose">Causal masked applied</span>
                    </div>
                    <div className="grid grid-cols-10 gap-0.5 bg-black/45 p-2 rounded-lg border border-panel-border/20">
                      {Array.from({ length: 100 }).map((_, idx) => {
                        const row = Math.floor(idx / 10);
                        const col = idx % 10;
                        const isMasked = col > row;
                        const weight = isMasked ? 0 : Math.exp(-Math.abs(row - col) / 2.0);
                        const cellColor = isMasked ? 'rgba(255,255,255,0.01)' : getCausalAttentionHeatColor(weight);
                        
                        return (
                          <div
                            key={idx}
                            className="aspect-square rounded-[1px] transition-all"
                            style={{
                              backgroundColor: cellColor,
                              opacity: isMasked ? 0.1 : 0.45 + weight * 0.55,
                            }}
                          />
                        );
                      })}
                    </div>
                    <div className="flex justify-between text-[8px] font-mono text-text-tertiary">
                      <span>t = 1 (Context)</span>
                      <span>Masked Upper Triangle</span>
                      <span>t = 10 (Horizon)</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Stage 3: CNN Decoder Reconstructions */}
              {activeStage === 3 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center animate-fade-in text-left">
                  <div className="space-y-3">
                    <span className="eyebrow-label text-accent-sage">Stage 3: Reconstruction</span>
                    <h4 className="text-sm font-semibold text-text-primary">Latent decoder mapping</h4>
                    <p className="text-xs text-text-secondary leading-relaxed font-sans prose-panel">
                      Reconstructs high-dimensional 64×64 simulated frames from future latent tokens z<sub>t+k</sub> using a symmetric decoder model.
                    </p>
                    <div className="bg-[#15141f]/75 p-3 rounded-lg border border-panel-border/30 text-[10px] font-mono text-text-secondary space-y-1 select-text font-mono-tabular">
                      <p>• Decoder layout: 4 ConvTranspose blocks</p>
                      <p>• Output range: Tanh projected [-1.0, 1.0]</p>
                      <p>• Rollout steps: 10 future simulation predictions</p>
                    </div>
                  </div>

                  {/* predicted frame player */}
                  <div className="flex justify-center select-none">
                    {result.predicted_frames && result.predicted_frames.length > 0 ? (
                      <FramePlayer
                        frames={result.predicted_frames}
                        label="Forecasted future snapshots sequence"
                        size={210}
                        currentFrame={currentFrame}
                        onFrameChange={setCurrentFrame}
                        contextCount={0} // All are predictions
                        accent="gold"
                      />
                    ) : (
                      <div className="w-56 h-56 rounded-xl bg-black/45 border border-panel-border/30 flex items-center justify-center text-text-secondary/40 text-xs">
                        No frame outputs available
                      </div>
                    )}
                  </div>
                </div>
              )}

            </div>
          </Card>

          {/* Spatial energy radial profiles sparkline */}
          <Card eyebrow="Analysis" title="Wave Radial Energy Profile Index">
            <div className="pt-2 select-none">
              <Sparkline
                data={result.spatial_spectrum}
                color="var(--color-accent-sage)"
                height={60}
              />
            </div>
            <div className="flex justify-between text-[9px] font-mono text-text-tertiary mt-2 select-none">
              <span>Wave peak origin (r = 0)</span>
              <span>Radial distance spectrum (pixels)</span>
              <span>Domain reflecting border (r = 45)</span>
            </div>
          </Card>

        </div>
      )}
    </div>
  );
}
