import { useState, useEffect } from 'react';
import FramePlayer from '../components/FramePlayer';
import { SkeletonCard } from '../components/SkeletonLoader';
import { fetchSamples } from '../api/client';
import demoFallback from '../data/demo_fallback.json';

export default function Overview({ isBackendOnline }) {
  const [sample, setSample] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        if (isBackendOnline) {
          const samples = await fetchSamples();
          if (samples.length > 0) setSample(samples[0]);
        } else if (demoFallback.samples?.length > 0) {
          setSample(demoFallback.samples[0]);
        }
      } catch {
        if (demoFallback.samples?.length > 0) {
          setSample(demoFallback.samples[0]);
        }
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [isBackendOnline]);

  const stats = sample ? [
    {
      label: 'Parameters',
      value: '~10M',
      detail: 'CNN Encoder + Transformer + CNN Decoder',
      color: 'text-electric',
    },
    {
      label: 'PSNR @ Step 1',
      value: sample.psnr_per_step?.[0]?.toFixed(1) || '—',
      detail: 'Peak signal-to-noise ratio, first predicted frame',
      color: 'text-emerald',
      unit: 'dB',
    },
    {
      label: 'PSNR @ Step 10',
      value: sample.psnr_per_step?.[9]?.toFixed(1) || sample.psnr_per_step?.slice(-1)[0]?.toFixed(1) || '—',
      detail: 'Expected degradation over longer rollouts',
      color: 'text-amber',
      unit: 'dB',
    },
  ] : [];

  const [currentFrame, setCurrentFrame] = useState(0);

  return (
    <div className="animate-fade-in space-y-8">
      {/* Hero */}
      <div className="text-center space-y-4 py-8">
        <h1 className="text-4xl md:text-5xl font-bold gradient-text">
          Latent Video Prediction
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          A hybrid CNN-Transformer model that compresses physical simulation frames into a
          learned latent space, then forecasts future dynamics using causal attention.
          Trained on 2D wave equation propagation.
        </p>
      </div>

      {/* Stats */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => <SkeletonCard key={i} />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {stats.map(({ label, value, detail, color, unit }) => (
            <div key={label} className="glass-card p-6 text-center glow-pulse">
              <p className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">{label}</p>
              <p className={`text-3xl font-bold font-mono ${color}`}>
                {value}{unit && <span className="text-lg ml-1">{unit}</span>}
              </p>
              <p className="text-xs text-slate-400/60 mt-2">{detail}</p>
            </div>
          ))}
        </div>
      )}

      {/* Live demo */}
      {sample && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-white">
            Live Prediction Example
          </h2>
          <p className="text-sm text-slate-400">
            Side-by-side comparison of ground truth (left) vs. model prediction (right).
            The model receives 10 context frames and predicts the next 10.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FramePlayer
              frames={sample.ground_truth_frames}
              label="Ground Truth"
              size={256}
              currentFrame={currentFrame}
            />
            <FramePlayer
              frames={sample.predicted_frames}
              label="Model Prediction"
              size={256}
              onFrameChange={setCurrentFrame}
            />
          </div>

          {/* Per-frame PSNR */}
          {sample.psnr_per_step && (
            <div className="glass-card p-4">
              <p className="text-xs font-mono text-slate-400 mb-2">Per-frame PSNR (dB)</p>
              <div className="flex items-end gap-1 h-16">
                {sample.psnr_per_step.map((v, i) => {
                  const max = Math.max(...sample.psnr_per_step);
                  const min = Math.min(...sample.psnr_per_step);
                  const h = ((v - min) / (max - min + 1)) * 100;
                  return (
                    <div
                      key={i}
                      className={`flex-1 rounded-t transition-all ${
                        i === currentFrame ? 'bg-electric' : 'bg-electric/30'
                      }`}
                      style={{ height: `${Math.max(h, 5)}%` }}
                      title={`Step ${i + 1}: ${v.toFixed(1)} dB`}
                    />
                  );
                })}
              </div>
              <p className="text-xs text-slate-400/60 mt-2 italic">
                Higher bars indicate better prediction quality. Quality typically degrades for later frames.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Architecture overview */}
      <div className="glass-card p-8 space-y-4">
        <h2 className="text-xl font-semibold text-white">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {[
            {
              step: '01',
              title: 'Encode',
              desc: 'Each frame passes through a 4-stage CNN encoder, compressing from 64×64 pixels to a 256-dimensional latent token.',
              color: 'from-electric to-blue-400',
            },
            {
              step: '02',
              title: 'Predict',
              desc: 'A 6-layer causal Transformer receives context latents and autoregressively generates future latent tokens using learned queries.',
              color: 'from-purple to-pink-400',
            },
            {
              step: '03',
              title: 'Decode',
              desc: 'A mirror-image CNN decoder reconstructs predicted frames from the Transformer\'s output latents back to 64×64 resolution.',
              color: 'from-amber to-orange-400',
            },
          ].map(({ step, title, desc, color }) => (
            <div key={step} className="space-y-3">
              <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center text-white font-mono font-bold text-sm`}>
                {step}
              </div>
              <h3 className="font-semibold text-white">{title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
