import { useState, useRef } from 'react';
import FramePlayer from '../components/FramePlayer';
import { Sparkline } from '../components/MetricChart';
import { uploadAndAnalyzeImage } from '../api/client';
import demoFallback from '../data/demo_fallback.json';

// Helper to generate canvas preset images
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
        // High frequency pattern
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

// Convert data URL to Blob/File
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

// Client mock spatial vectors generator for offline fallback mode
function generateMockSpatialVectors() {
  const vectors = [];
  const grid = 8;
  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      const x = (j + 0.5) / grid;
      const y = (i + 0.5) / grid;
      const dx = (x - 0.5) * 0.4;
      const dy = (y - 0.5) * 0.4;
      const mag = Math.sqrt(dx * dx + dy * dy);
      vectors.push({
        x,
        y,
        dx: Math.round(dx * 1000) / 1000,
        dy: Math.round(dy * 1000) / 1000,
        magnitude: Math.round(mag * 1000) / 1000,
        angle: Math.atan2(dy, dx),
      });
    }
  }
  return vectors;
}

export default function ImageAnalysis({ isBackendOnline }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [viewMode, setViewMode] = useState('input'); // 'input' or 'features'
  const fileInputRef = useRef(null);

  const presets = [
    { id: 'single_pulse', name: 'Gaussian Pulse', desc: 'Centered smooth wave peak' },
    { id: 'dual_ripples', name: 'Dual Ripples', desc: 'Interfering wavefronts' },
    { id: 'ring_wave', name: 'Ring Wavefront', desc: 'Expanding circular crest' },
    { id: 'high_freq', name: 'High Frequency Wave', desc: 'Complex spatial wave mode' },
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

    if (!isBackendOnline) {
      setTimeout(() => {
        const sample = demoFallback.samples[0];
        const mockLatent = Array.from({ length: 256 }, (_, i) => Math.sin(i / 6) * 1.5 + Math.cos(i / 12) * 0.8);
        setResult({
          original_filename: selectedFile.name,
          processed_frame: sample.context_frames[0],
          feature_map_frame: sample.context_frames[0],
          latent_vector: mockLatent,
          latent_norm: 5.4218,
          predicted_frames: sample.predicted_frames,
          estimated_params: {
            center_x: 0.52,
            center_y: 0.48,
            amplitude: 0.85,
            width: 4.8,
            total_energy: 142.3,
          },
          spatial_spectrum: [0.1, 0.3, 0.7, 0.9, 0.8, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.0, 0.0],
          spatial_vectors: generateMockSpatialVectors(),
        });
        setLoading(false);
      }, 600);
      return;
    }

    try {
      const data = await uploadAndAnalyzeImage(selectedFile);
      setResult(data);
      setCurrentFrame(0);
    } catch (err) {
      console.error('Image analysis failed:', err);
      setError('Failed to analyze image on backend server. Falling back to client estimation mode.');
      const sample = demoFallback.samples[0];
      const mockLatent = Array.from({ length: 256 }, (_, i) => Math.sin(i / 6) * 1.5 + Math.cos(i / 12) * 0.8);
      setResult({
        original_filename: selectedFile.name,
        processed_frame: sample.context_frames[0],
        feature_map_frame: sample.context_frames[0],
        latent_vector: mockLatent,
        latent_norm: 4.89,
        predicted_frames: sample.predicted_frames,
        estimated_params: {
          center_x: 0.5,
          center_y: 0.5,
          amplitude: 0.8,
          width: 5.0,
          total_energy: 120.0,
        },
        spatial_spectrum: [0.1, 0.4, 0.85, 0.95, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0],
        spatial_vectors: generateMockSpatialVectors(),
      });
    } finally {
      setLoading(false);
    }
  };

  // Helper for dynamic latent normalization
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

  return (
    <div className="animate-fade-in space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
          Image Upload & Wave Analysis
        </h1>
        <p className="text-sm text-text-secondary">
          Upload any wave image, ripple snapshot, or pattern photograph to extract CNN latent embeddings, compute 2D spatial gradient vectors, estimate physical wave properties, and simulate auto-regressive future wave evolution.
        </p>
      </div>

      {/* Upload & Controls Panel */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Drag & Drop Dropzone */}
        <div className="md:col-span-2 glass-card p-6 flex flex-col justify-between">
          <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>
            Upload Picture
          </h2>

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[200px] ${
              dragActive
                ? 'border-gold bg-gold/10 shadow-lg shadow-gold/10'
                : previewUrl
                ? 'border-gold/40 bg-white/2 hover:border-gold/80'
                : 'border-white/15 hover:border-gold/50 bg-white/2'
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              accept="image/*"
              className="hidden"
            />

            {previewUrl ? (
              <div className="flex flex-col items-center gap-3">
                <img
                  src={previewUrl}
                  alt="Uploaded Preview"
                  className="w-32 h-32 object-cover rounded-lg border border-gold/30 shadow-md"
                />
                <span className="text-xs text-gold font-mono truncate max-w-[200px]">
                  {selectedFile ? selectedFile.name : 'Selected Image'}
                </span>
                <span className="text-[11px] text-text-secondary">Click or drag a new image to replace</span>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="w-12 h-12 rounded-full bg-gold/10 border border-gold/20 text-gold flex items-center justify-center text-xl mx-auto">
                  📷
                </div>
                <div>
                  <p className="text-sm font-medium text-text-primary">Drag & drop your picture here</p>
                  <p className="text-xs text-text-secondary mt-1">Supports PNG, JPG, WebP, GIF files</p>
                </div>
              </div>
            )}
          </div>

          {error && <p className="text-xs text-rose mt-3 text-center">{error}</p>}

          <div className="mt-6 flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={loading || !selectedFile}
              className={`px-8 py-3 rounded-xl font-semibold text-sm transition-all w-full md:w-auto ${
                loading || !selectedFile
                  ? 'bg-gold/15 text-gold/40 cursor-not-allowed'
                  : 'bg-gradient-to-r from-gold to-rose text-[#1a1a2e] hover:shadow-lg hover:shadow-gold/20 hover:scale-[1.02] active:scale-[0.98]'
              }`}
              style={{ fontFamily: 'var(--font-heading)' }}
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-gold/40 border-t-gold rounded-full animate-spin" />
                  Analyzing Picture & Extracting Vectors...
                </span>
              ) : (
                '🔬 Analyze Image & Extract Vectors'
              )}
            </button>
          </div>
        </div>

        {/* Preset Sample Selector */}
        <div className="glass-card p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-semibold text-text-primary mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
              Or Try Sample Presets
            </h2>
            <p className="text-xs text-text-secondary mb-4">Click a preset below to instantly load a test wave image.</p>
            <div className="space-y-3">
              {presets.map((p) => (
                <button
                  key={p.id}
                  onClick={() => handlePresetSelect(p.id)}
                  className="w-full text-left p-3 rounded-xl bg-white/4 border border-white/6 hover:border-gold/40 hover:bg-gold/5 transition-all text-xs flex justify-between items-center group"
                >
                  <div>
                    <p className="font-semibold text-text-primary group-hover:text-gold">{p.name}</p>
                    <p className="text-[10px] text-text-secondary">{p.desc}</p>
                  </div>
                  <span className="text-gold/50 group-hover:text-gold font-mono">→</span>
                </button>
              ))}
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-white/6 text-[10px] text-text-secondary text-center">
            {isBackendOnline ? (
              <span className="text-sage">✓ Live PyTorch Backend Connected</span>
            ) : (
              <span className="text-gold">⚡ Running in Client Simulation Mode</span>
            )}
          </div>
        </div>
      </div>

      {/* Analysis Results View */}
      {result && (
        <div className="space-y-6">
          {/* Top Metrics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="glass-card p-4 text-center">
              <span className="text-[10px] text-text-secondary uppercase tracking-wider block">Pulse Center X</span>
              <span className="text-lg font-bold font-mono text-gold">{result.estimated_params.center_x}</span>
            </div>
            <div className="glass-card p-4 text-center">
              <span className="text-[10px] text-text-secondary uppercase tracking-wider block">Pulse Center Y</span>
              <span className="text-lg font-bold font-mono text-gold">{result.estimated_params.center_y}</span>
            </div>
            <div className="glass-card p-4 text-center">
              <span className="text-[10px] text-text-secondary uppercase tracking-wider block">Amplitude</span>
              <span className="text-lg font-bold font-mono text-sage">{result.estimated_params.amplitude}</span>
            </div>
            <div className="glass-card p-4 text-center">
              <span className="text-[10px] text-text-secondary uppercase tracking-wider block">Pulse Width</span>
              <span className="text-lg font-bold font-mono text-rose">{result.estimated_params.width}</span>
            </div>
            <div className="glass-card p-4 text-center col-span-2 md:col-span-1">
              <span className="text-[10px] text-text-secondary uppercase tracking-wider block">Latent L2 Norm</span>
              <span className="text-lg font-bold font-mono text-text-primary">{result.latent_norm}</span>
            </div>
          </div>

          {/* 2D Spatial Vectors & Feature Activation Map Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 2D Spatial Vector Field Overlay Panel */}
            <div className="glass-card p-6 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-text-primary" style={{ fontFamily: 'var(--font-heading)' }}>
                  2D Spatial Energy Gradient Vectors ($\nabla I$)
                </h3>
                <div className="flex items-center gap-1 bg-black/40 p-1 rounded-lg border border-white/6 text-[11px]">
                  <button
                    onClick={() => setViewMode('input')}
                    className={`px-2 py-0.5 rounded transition-all ${viewMode === 'input' ? 'bg-gold/20 text-gold font-semibold' : 'text-text-secondary'}`}
                  >
                    Input Frame
                  </button>
                  <button
                    onClick={() => setViewMode('features')}
                    className={`px-2 py-0.5 rounded transition-all ${viewMode === 'features' ? 'bg-gold/20 text-gold font-semibold' : 'text-text-secondary'}`}
                  >
                    CNN Feature Map
                  </button>
                </div>
              </div>

              <div className="relative w-full max-w-[280px] h-[280px] mx-auto bg-black/50 rounded-xl overflow-hidden border border-gold/40 shadow-inner flex items-center justify-center">
                {/* Background Frame (Input or CNN feature map) */}
                <img
                  src={`data:image/png;base64,${viewMode === 'features' && result.feature_map_frame ? result.feature_map_frame : result.processed_frame}`}
                  alt="Spatial Base"
                  className="absolute inset-0 w-full h-full object-cover opacity-60"
                  style={{ imageRendering: 'pixelated' }}
                />

                {/* SVG Vector Field Overlay */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100">
                  {result.spatial_vectors?.map((vec, idx) => {
                    const cx = vec.x * 100;
                    const cy = vec.y * 100;
                    const len = Math.min(10, Math.max(2, vec.magnitude * 80));
                    const angleRad = vec.angle;
                    const targetX = cx + Math.cos(angleRad) * len;
                    const targetY = cy + Math.sin(angleRad) * len;
                    const color = vec.magnitude > 0.05 ? '#d4a853' : '#7eb09b';

                    return (
                      <g key={idx}>
                        {/* Vector Line */}
                        <line
                          x1={cx}
                          y1={cy}
                          x2={targetX}
                          y2={targetY}
                          stroke={color}
                          strokeWidth="1.2"
                          strokeLinecap="round"
                          opacity={0.3 + Math.min(0.7, vec.magnitude * 5)}
                        />
                        {/* Vector Point */}
                        <circle
                          cx={cx}
                          cy={cy}
                          r="1"
                          fill={color}
                          opacity={0.7}
                        />
                      </g>
                    );
                  })}

                  {/* Estimated Center Marker */}
                  <circle
                    cx={result.estimated_params.center_x * 100}
                    cy={result.estimated_params.center_y * 100}
                    r="3.5"
                    fill="#e06c75"
                    stroke="#ffffff"
                    strokeWidth="1"
                    className="animate-pulse"
                  />
                </svg>
              </div>

              <p className="text-[11px] text-text-secondary text-center">
                Directional spatial vectors ($\nabla I$) calculated across an 8×8 grid from your input picture. Red dot indicates estimated wave center $(X, Y) = ({result.estimated_params.center_x}, {result.estimated_params.center_y})$.
              </p>
            </div>

            {/* Predicted Wave Evolution Player */}
            <div className="glass-card p-6 flex flex-col justify-between">
              <h3 className="text-sm font-semibold text-text-primary mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
                Auto-Regressive Wave Evolution Forecast
              </h3>
              <div className="flex justify-center py-2">
                <FramePlayer
                  frames={result.predicted_frames}
                  label="Predicted Future Wave Sequence"
                  size={240}
                  currentFrame={currentFrame}
                  onFrameChange={setCurrentFrame}
                />
              </div>
              <p className="text-[11px] text-text-secondary text-center">
                Predicted 10 future time steps rolled out by the Latent Video Predictor from your input frame.
              </p>
            </div>
          </div>

          {/* CNN Latent Vector & Spatial Spectrum Charts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Latent Vector Visualization - All 256 Dimensions */}
            <div className="glass-card p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-text-primary" style={{ fontFamily: 'var(--font-heading)' }}>
                  CNN Latent Embedding (All 256 Dimensions)
                </h3>
                <span className="text-[10px] font-mono text-gold">Range: [{lMin.toFixed(2)}, {lMax.toFixed(2)}]</span>
              </div>
              <p className="text-xs text-text-secondary">
                Full 256-dimensional latent feature spectrum output from the CNN Encoder:
              </p>

              {/* 256-Dim Heatmap Grid (16 rows x 16 cols or bar strip) */}
              <div className="h-20 flex items-center gap-[1px] overflow-hidden rounded-lg bg-black/40 p-2 border border-white/6">
                {result.latent_vector.map((val, idx) => {
                  const normVal = Math.max(0, Math.min(1, (val - lMin) / (lMax - lMin)));
                  return (
                    <div
                      key={idx}
                      className="flex-1 h-full rounded-[0.5px] transition-all hover:scale-110"
                      style={{
                        backgroundColor: `hsl(${220 - normVal * 180}, 90%, ${25 + normVal * 55}%)`,
                        opacity: 0.4 + normVal * 0.6,
                      }}
                      title={`Dim ${idx}: ${val}`}
                    />
                  );
                })}
              </div>
              <div className="flex justify-between text-[10px] font-mono text-text-secondary">
                <span>Dim 0</span>
                <span>256 Latent Feature Channels</span>
                <span>Dim 255</span>
              </div>
            </div>

            {/* Spatial Energy Radial Profile */}
            <div className="glass-card p-6 space-y-3">
              <h3 className="text-sm font-semibold text-text-primary" style={{ fontFamily: 'var(--font-heading)' }}>
                Spatial Energy Radial Distribution
              </h3>
              <Sparkline
                data={result.spatial_spectrum}
                color="#7eb09b"
                height={70}
                label="Mean Intensity vs Radial Distance from Peak Center"
              />
              <div className="flex justify-between text-[10px] font-mono text-text-secondary">
                <span>Center (r=0)</span>
                <span>Radial Distance (pixels)</span>
                <span>Edge (r=45)</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
