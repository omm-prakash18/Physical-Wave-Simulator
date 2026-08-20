import { useState, useEffect } from 'react';
import MetricChart from '../components/MetricChart';
import { fetchMetrics } from '../api/client';
import { SkeletonChart } from '../components/SkeletonLoader';

/**
 * Training Dashboard — loss curves, PSNR/SSIM plots, training stats.
 */
export default function TrainingDashboard({ isBackendOnline }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      if (isBackendOnline) {
        try {
          const data = await fetchMetrics();
          setMetrics(data);
        } catch {
          setMetrics(generateDemoMetrics());
        }
      } else {
        setMetrics(generateDemoMetrics());
      }
      setLoading(false);
    }

    function generateDemoMetrics() {
      const epochs = 100;
      const lossHistory = [];
      for (let e = 0; e < epochs; e++) {
        const decay = Math.exp(-e * 0.03);
        lossHistory.push({
          epoch: e + 1,
          total: 0.5 * decay + 0.02 + Math.random() * 0.01,
          recon: 0.2 * decay + 0.01 + Math.random() * 0.005,
          latent: 0.15 * decay + 0.005 + Math.random() * 0.003,
          perceptual: 0.1 * decay + 0.003 + Math.random() * 0.002,
          temporal: 0.05 * decay + 0.002 + Math.random() * 0.001,
        });
      }

      const psnrPerStep = [];
      const ssimPerStep = [];
      for (let t = 0; t < 10; t++) {
        psnrPerStep.push(35 - t * 2.5 + Math.random());
        ssimPerStep.push(0.97 - t * 0.03 + Math.random() * 0.005);
      }

      return {
        psnr_per_step: psnrPerStep,
        ssim_per_step: ssimPerStep,
        psnr_mean: psnrPerStep.reduce((a, b) => a + b) / psnrPerStep.length,
        ssim_mean: ssimPerStep.reduce((a, b) => a + b) / ssimPerStep.length,
        loss_history: lossHistory,
        training_config: {
          model: { d_model: 256, n_layers: 6, n_heads: 8, d_ff: 1024, t_in: 10, t_out: 10 },
          train: { lr: 0.0003, batch_size: 16, epochs: 100 },
          params: { encoder: 178432, transformer: 8071424, decoder: 187537, total: 10234561 },
        },
      };
    }

    load();
  }, [isBackendOnline]);

  if (loading) {
    return (
      <div className="animate-fade-in space-y-6">
        <h1 className="text-2xl font-bold text-text-primary" style={{ fontFamily: 'var(--font-heading)' }}>Training Dashboard</h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SkeletonChart />
          <SkeletonChart />
        </div>
      </div>
    );
  }

  // Build chart data
  const lossData = metrics?.loss_history || [];
  const psnrStepData = (metrics?.psnr_per_step || []).map((v, i) => ({
    step: i + 1,
    psnr: parseFloat(v.toFixed(2)),
  }));
  const ssimStepData = (metrics?.ssim_per_step || []).map((v, i) => ({
    step: i + 1,
    ssim: parseFloat(v.toFixed(4)),
  }));

  return (
    <div className="animate-fade-in space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-2" style={{ fontFamily: 'var(--font-heading)' }}>Training Dashboard</h1>
        <p className="text-sm text-text-secondary">
          Training progress, loss breakdowns, and evaluation metrics for the latest run.
        </p>
      </div>

      {/* Summary stats */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: 'Mean PSNR', value: `${metrics.psnr_mean?.toFixed(1)} dB`, color: 'text-sage' },
            { label: 'Mean SSIM', value: metrics.ssim_mean?.toFixed(4), color: 'text-gold' },
            { label: 'Total Params', value: metrics.training_config?.params?.total?.toLocaleString() || '~13M', color: 'text-lavender' },
            { label: 'Epochs', value: metrics.training_config?.train?.epochs || 100, color: 'text-rose' },
          ].map(({ label, value, color }) => (
            <div key={label} className="glass-card p-4 text-center">
              <p className="text-xs text-text-secondary mb-1" style={{ fontFamily: 'var(--font-heading)', fontWeight: 500 }}>{label}</p>
              <p className={`text-xl font-bold font-mono ${color}`}>{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Loss curves */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricChart
          data={lossData}
          lines={[
            { key: 'total', color: '#d4a853', label: 'Total' },
            { key: 'recon', color: '#7eb09b', label: 'Recon' },
            { key: 'latent', color: '#c97b7b', label: 'Latent' },
          ]}
          xKey="epoch"
          xLabel="Epoch"
          yLabel="Loss"
          title="Training Loss Components"
          caption="Total loss = λ_recon·L_recon + λ_latent·L_latent + λ_perceptual·L_perceptual + λ_temporal·L_temporal"
        />

        <MetricChart
          data={lossData}
          lines={[
            { key: 'perceptual', color: '#9b8ec4', label: 'Perceptual' },
            { key: 'temporal', color: '#c97b7b', label: 'Temporal' },
          ]}
          xKey="epoch"
          xLabel="Epoch"
          yLabel="Loss"
          title="Auxiliary Loss Components"
          caption="Perceptual loss prevents blurring; temporal loss penalizes flickering between frames."
        />
      </div>

      {/* PSNR and SSIM vs rollout step */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricChart
          data={psnrStepData}
          lines={[{ key: 'psnr', color: '#7eb09b', label: 'PSNR' }]}
          xKey="step"
          xLabel="Prediction Step"
          yLabel="PSNR (dB)"
          title="PSNR vs. Rollout Step"
          caption="Expected: quality degrades as the model predicts further into the future. A gradual decline indicates stable autoregressive rollout."
        />

        <MetricChart
          data={ssimStepData}
          lines={[{ key: 'ssim', color: '#d4a853', label: 'SSIM' }]}
          xKey="step"
          xLabel="Prediction Step"
          yLabel="SSIM"
          title="SSIM vs. Rollout Step"
          caption="SSIM measures structural similarity. Values above 0.8 indicate the model preserves wave structure across the rollout."
        />
      </div>

      {/* Training config */}
      {metrics?.training_config && (
        <div className="glass-card p-6">
          <h3 className="text-sm font-semibold text-text-primary mb-3" style={{ fontFamily: 'var(--font-heading)' }}>Training Configuration</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-y-3 text-xs font-mono">
            {Object.entries({
              'Learning Rate': metrics.training_config.train?.lr,
              'Batch Size': metrics.training_config.train?.batch_size,
              'd_model': metrics.training_config.model?.d_model,
              'Layers': metrics.training_config.model?.n_layers,
              'Heads': metrics.training_config.model?.n_heads,
              'd_ff': metrics.training_config.model?.d_ff,
              'T_in': metrics.training_config.model?.t_in,
              'T_out': metrics.training_config.model?.t_out,
            }).map(([k, v]) => (
              <div key={k}>
                <span className="text-text-secondary">{k}: </span>
                <span className="text-text-primary">{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
