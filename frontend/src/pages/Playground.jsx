import { useState } from 'react';
import FramePlayer from '../components/FramePlayer';
import { Sparkline } from '../components/MetricChart';
import { generatePrediction } from '../api/client';
import demoFallback from '../data/demo_fallback.json';

export default function Playground({ isBackendOnline }) {
  const [params, setParams] = useState({
    center_x: 0.5,
    center_y: 0.5,
    width: 5.0,
    amplitude: 0.8,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);

  const handleGenerate = async () => {
    if (!isBackendOnline) {
      if (demoFallback.samples?.length > 1) {
        setResult(demoFallback.samples[1]);
      }
      return;
    }

    setLoading(true);
    try {
      const data = await generatePrediction(params);
      setResult(data);
      setCurrentFrame(0);
    } catch (err) {
      console.error('Generation failed:', err);
      if (demoFallback.samples?.length > 1) {
        setResult(demoFallback.samples[1]);
      }
    } finally {
      setLoading(false);
    }
  };

  const sliders = [
    { key: 'center_x', label: 'Pulse X', min: 0.1, max: 0.9, step: 0.05 },
    { key: 'center_y', label: 'Pulse Y', min: 0.1, max: 0.9, step: 0.05 },
    { key: 'width', label: 'Width', min: 2.0, max: 10.0, step: 0.5 },
    { key: 'amplitude', label: 'Amplitude', min: 0.1, max: 1.0, step: 0.05 },
  ];

  return (
    <div className="animate-fade-in space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-2" style={{ fontFamily: 'var(--font-heading)' }}>Live Prediction Playground</h1>
        <p className="text-sm text-text-secondary">
          Adjust the wave simulation parameters, generate a new sequence, and watch the model predict future frames in real time.
        </p>
      </div>

      {/* Controls */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Simulation Parameters</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {sliders.map(({ key, label, min, max, step }) => (
            <div key={key} className="space-y-2">
              <div className="flex justify-between">
                <label className="text-xs text-text-secondary font-medium">{label}</label>
                <span className="text-xs font-mono text-gold">{params[key].toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={params[key]}
                onChange={(e) => setParams({ ...params, [key]: parseFloat(e.target.value) })}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
              />
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-center">
          <button
            onClick={handleGenerate}
            disabled={loading}
            className={`px-8 py-3 rounded-xl font-semibold text-sm transition-all ${
              loading
                ? 'bg-gold/15 text-gold/50 cursor-wait'
                : 'bg-gradient-to-r from-gold to-rose text-[#1a1a2e] hover:shadow-lg hover:shadow-gold/20 hover:scale-[1.02] active:scale-[0.98]'
            }`}
            style={{ fontFamily: 'var(--font-heading)' }}
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="w-4 h-4 border-2 border-gold/40 border-t-gold rounded-full animate-spin" />
                Generating...
              </span>
            ) : (
              '⚡ Generate & Predict'
            )}
          </button>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FramePlayer
              frames={result.ground_truth_frames}
              label="Ground Truth"
              size={280}
              currentFrame={currentFrame}
            />
            <FramePlayer
              frames={result.predicted_frames}
              label="Model Prediction"
              size={280}
              onFrameChange={setCurrentFrame}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="glass-card p-4">
              <Sparkline
                data={result.psnr_per_step}
                color="#7eb09b"
                height={60}
                label="PSNR per step (dB) — Higher is better"
              />
            </div>
            <div className="glass-card p-4">
              <Sparkline
                data={result.ssim_per_step}
                color="#d4a853"
                height={60}
                label="SSIM per step — Closer to 1.0 is better"
              />
            </div>
          </div>

          <div className="glass-card p-4">
            <p className="text-xs font-mono text-text-secondary">
              Params: center=({result.params?.center_x?.toFixed(2)}, {result.params?.center_y?.toFixed(2)}),
              width={result.params?.width?.toFixed(1)},
              amp={result.params?.amplitude?.toFixed(2)}
              {' | '}
              Mean PSNR: {(result.psnr_per_step?.reduce((a, b) => a + b, 0) / result.psnr_per_step?.length).toFixed(1)} dB
              {' | '}
              Mean SSIM: {(result.ssim_per_step?.reduce((a, b) => a + b, 0) / result.ssim_per_step?.length).toFixed(3)}
            </p>
          </div>
        </div>
      )}

      {!result && !loading && (
        <div className="text-center py-16 text-text-secondary/50">
          <p className="text-4xl mb-4">🌊</p>
          <p className="text-sm">Adjust the parameters above and click "Generate & Predict" to see the model in action.</p>
        </div>
      )}
    </div>
  );
}
