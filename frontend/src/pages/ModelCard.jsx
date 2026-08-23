import React from 'react';
import Card from '../components/Card';

/**
 * Model Card Page - Complete technical specifications sheet.
 * Caps prose panels to 70ch, right-aligns parameter counts in tabular-nums,
 * and formats loss metrics and physical datasets schemas using unified Cards.
 */
export default function ModelCard() {
  const architecture = [
    {
      component: 'CNN spatial encoder',
      params: '455,200',
      details: '4-stage Conv-BN-SiLU downsampler (64→32→16→8→4), global average pool, projection to d=256',
    },
    {
      component: 'Causal Latent Transformer',
      params: '11,100,000',
      details: '6 causal attention encoder + 6 decoder layers, 8 heads, d_model=256, d_ff=1024, pre-LN, learned query vectors',
    },
    {
      component: 'CNN spatial decoder',
      params: '1,750,000',
      details: 'Linear projection, reshape, 4-stage ConvTranspose-BN-SiLU upsampler (4→8→16→32→64), ConvOut + Tanh projection',
    },
  ];

  const lossComponents = [
    { name: 'L_recon', weight: '1.0', desc: 'Mean Absolute Error (L1) pixel reconstruction loss on wave heights' },
    { name: 'L_latent', weight: '1.0', desc: 'Mean Squared Error (MSE) in latent space - primary transformer optimization target' },
    { name: 'L_perceptual', weight: '0.1', desc: 'Feature space L1 distance using frozen encoder outputs to retain high-frequency waves' },
    { name: 'L_temporal', weight: '0.5', desc: 'Motion consistency loss penalizing frame-to-frame pixel differences' },
  ];

  return (
    <div className="animate-fade-in space-y-6">
      {/* Title Header */}
      <div>
        <h1 className="text-2xl font-bold font-heading text-text-primary mb-1">
          Model Specifications Card
        </h1>
        <p className="text-xs text-text-secondary font-sans leading-relaxed">
          Technical specifications sheet, parameter logs, dataset parameters, and training loss criteria.
        </p>
      </div>

      {/* Grid Layout of Cards */}
      <div className="grid grid-cols-1 gap-6">
        
        {/* Architecture Layout Scheme */}
        <Card eyebrow="Pipeline Diagram" title="Encoder-Decoder Transformer Layout">
          <div className="mt-2 bg-[#15141f]/75 border border-panel-border/30 rounded-xl p-5 select-text overflow-x-auto">
            <pre className="text-[10px] font-mono text-accent-gold/80 leading-relaxed whitespace-pre font-mono-tabular">
{`   Input Sequence (Batch, T_in = 10, Channels = 1, Width = 64, Height = 64)
          │
     ┌────┴────┐
     │   CNN   │ ◄── 4-stage Conv-BN-SiLU downsampler, stride 2
     │ Encoder │     Features: 64→32→16→8→4, GAP, Linear projection
     └────┬────┘
          │
    Latent Space (Batch, T_in = 10, d_model = 256)
          │
     ┌────┴────────────────────────┐
     │ Causal Latent Transformer  │ ◄── 6 + 6 layer Encoder-Decoder layers, 8 heads
     │ (Causal Attention Masking) │     Learned query tokens for autoregressive prediction
     └────┬────────────────────────┘
          │
   Forecasted Latents (Batch, T_out = 10, d_model = 256)
          │
     ┌────┴────┐
     │   CNN   │ ◄── Symmetric ConvTranspose upsampler, stride 2
     │ Decoder │     Linear projection, Reshape, ConvT-BN-SiLU x4, Tanh
     └────┬────┘
          │
   Predicted Rollout (Batch, T_out = 10, Channels = 1, Width = 64, Height = 64)`}
            </pre>
          </div>
        </Card>

        {/* Parameter table breakdown */}
        <Card eyebrow="Model Weights" title="Parameter Budget Breakdown">
          <div className="overflow-x-auto pt-2">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-panel-border/30 text-text-secondary select-none">
                  <th className="py-2 pr-4 font-mono font-medium uppercase tracking-wider text-[10px]">Architecture Component</th>
                  <th className="py-2 pr-4 text-right font-mono font-medium uppercase tracking-wider text-[10px]">Weights Count</th>
                  <th className="py-2 font-mono font-medium uppercase tracking-wider text-[10px]">Layer Specifications</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-panel-border/10 font-sans">
                {architecture.map(({ component, params, details }) => (
                  <tr key={component} className="hover:bg-white/[0.01] transition-snappy">
                    <td className="py-3 pr-4 font-semibold text-text-primary text-[13px]">{component}</td>
                    <td className="py-3 pr-4 text-right font-mono font-mono-tabular text-accent-gold text-[13px]">{params}</td>
                    <td className="py-3 text-text-secondary leading-relaxed max-w-[400px] text-xs">{details}</td>
                  </tr>
                ))}
                {/* Total Row */}
                <tr className="bg-accent-gold/[0.03] font-mono">
                  <td className="py-3 pr-4 font-bold text-text-primary text-[13px]">Total Param Budget</td>
                  <td className="py-3 pr-4 text-right font-bold font-mono-tabular text-accent-rose text-[13px]">13,305,200</td>
                  <td className="py-3 text-text-secondary/70 leading-relaxed text-xs">Production-ready hybrid latent forecaster</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>

        {/* Loss criteria explanation */}
        <Card eyebrow="Objective Function" title="Combined Training Loss Criteria">
          <div className="space-y-4 pt-2 select-text">
            <div className="prose-panel text-xs text-text-secondary leading-relaxed">
              The model parameters are optimized end-to-end using a weighted combination of pixel accuracy, latent structure, visual fidelity, and frame consistency loss terms:
            </div>
            <div className="p-3 bg-[#15141f]/75 rounded-xl border border-panel-border/30 font-mono text-[11px] text-accent-gold max-w-xl">
              L_total = 1.0 · L_recon + 1.0 · L_latent + 0.1 · L_perceptual + 0.5 · L_temporal
            </div>
            <div className="space-y-3 pt-2">
              {lossComponents.map(({ name, weight, desc }) => (
                <div key={name} className="flex items-start gap-3 select-text font-sans">
                  <span className="text-[10px] font-mono font-bold text-accent-gold bg-accent-gold/10 px-2 py-0.5 rounded border border-accent-gold/20 shrink-0">
                    λ = {weight}
                  </span>
                  <div className="text-xs">
                    <span className="font-semibold text-text-primary block font-mono">{name}</span>
                    <p className="text-text-secondary mt-0.5 leading-relaxed prose-panel">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Physical Dataset specifications */}
        <Card eyebrow="Data Setup" title="2D Wave Solver Dataset Grid">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2 text-xs font-mono select-text font-mono-tabular">
            {[
              { label: 'Total Sequences', value: '10,000' },
              { label: 'Frames per Sequence', value: '20' },
              { label: 'Spatial Resolution', value: '64 × 64 pixels' },
              { label: 'Grid Channels', value: '1 (Grayscale)' },
              { label: 'Boundary Conditions', value: 'Neumann reflecting' },
              { label: 'Train/Val/Test Split', value: '80 / 10 / 10 %' },
              { label: 'Simulation Type', value: '2D Finite-Difference' },
              { label: 'Solver Seed', value: 'Deterministic (#1337)' },
            ].map(({ label, value }) => (
              <div key={label} className="border border-panel-border/20 p-3 rounded-lg bg-white/[0.01]">
                <span className="text-text-secondary text-[10px] uppercase font-sans tracking-wide block">{label}</span>
                <p className="text-text-primary font-bold mt-1 text-[13px]">{value}</p>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-text-secondary/50 font-sans italic mt-4 leading-relaxed prose-panel">
            Initial states are formed by a single Gaussian pulse with random centers (x, y), peak amplitudes (A), and widths (σ) propagating on a uniform grid with boundary reflections.
          </p>
        </Card>

        {/* Training Parameters Details */}
        <Card eyebrow="Optimizer" title="Training Optimization Details">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 pt-2 text-xs font-mono select-text font-mono-tabular">
            {[
              { label: 'Optimizer Algorithm', value: 'AdamW' },
              { label: 'Initial Learning Rate', value: '3e-4' },
              { label: 'Learning Scheduler', value: 'Cosine annealing' },
              { label: 'Batch Size', value: '16' },
              { label: 'Precision Format', value: 'FP16 mixed precision' },
              { label: 'Warmup Epochs', value: '5 epochs' },
              { label: 'Teacher Forcing Schedule', value: '1.0 → 0.0 linear decay' },
              { label: 'Gradient Clipping Limit', value: '1.0 max norm' },
              { label: 'L2 Weight Decay', value: '0.01' },
            ].map(({ label, value }) => (
              <div key={label} className="border border-panel-border/20 p-3 rounded-lg bg-white/[0.01]">
                <span className="text-text-secondary text-[10px] uppercase font-sans tracking-wide block">{label}</span>
                <p className="text-text-primary font-bold mt-1 text-[13px]">{value}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Verification & Physics Rigor diagnostics card */}
        <Card eyebrow="ML Rigor" title="Diagnostics & Physical Verification Results">
          <div className="space-y-4 pt-2 select-text font-sans text-xs">
            <div className="prose-panel text-text-secondary leading-relaxed">
              To validate whether the autoregressive model represents physically consistent wave equation dynamics (rather than simply minimizing short-term pixel losses), we run linear probes and conservation validations on the test split:
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div className="border border-panel-border/20 p-3.5 rounded-lg bg-white/[0.01]">
                <span className="text-accent-sage font-mono uppercase tracking-wide text-[10px] block font-semibold mb-1">Latent Probing (OLS R2)</span>
                <ul className="space-y-1 font-mono font-mono-tabular text-text-secondary select-text">
                  <li>- Center X position: <strong className="text-text-primary">0.9701</strong></li>
                  <li>- Center Y position: <strong className="text-text-primary">0.9651</strong></li>
                  <li>- Pulse Width (sigma): <strong className="text-text-primary">0.9924</strong></li>
                  <li>- Wave Amplitude: <strong className="text-accent-rose">0.0000</strong>*</li>
                </ul>
                <p className="text-[9px] text-text-tertiary mt-2 leading-tight">
                  *Amplitude maps to 0.0 due to per-sequence simulation normalization.
                </p>
              </div>

              <div className="border border-panel-border/20 p-3.5 rounded-lg bg-white/[0.01]">
                <span className="text-accent-gold font-mono uppercase tracking-wide text-[10px] block font-semibold mb-1">Energy drift (10 steps)</span>
                <ul className="space-y-1 font-mono font-mono-tabular text-text-secondary select-text">
                  <li>- Ground Truth: <strong className="text-text-primary">0.00%</strong> (Conservative)</li>
                  <li>- Predictor: <strong className="text-accent-sage">-12.42%</strong> (Diffusion)</li>
                  <li>- Persistence: <strong className="text-text-primary">0.00%</strong> (Static frame)</li>
                  <li>- Extrapolation: <strong className="text-accent-coral">+342.15%</strong> (Explodes)</li>
                </ul>
              </div>

              <div className="border border-panel-border/20 p-3.5 rounded-lg bg-white/[0.01]">
                <span className="text-[#c286d9] font-mono uppercase tracking-wide text-[10px] block font-semibold mb-1">Inference Latency</span>
                <ul className="space-y-1 font-mono font-mono-tabular text-text-secondary select-text">
                  <li>- PyTorch CPU: <strong className="text-text-primary">~22.4 ms</strong></li>
                  <li>- ONNX Runtime: <strong className="text-accent-gold">~5.1 ms</strong></li>
                  <li>- Speedup factor: <strong className="text-accent-sage">4.39x</strong></li>
                </ul>
              </div>
            </div>
          </div>
        </Card>

        {/* Technology stack tags */}
        <Card eyebrow="Technologies" title="Development Tech Stack">
          <div className="flex flex-wrap gap-2 pt-2 select-none">
            {['PyTorch', 'FastAPI', 'Vite', 'React 18', 'Tailwind CSS v4', 'Recharts', 'TensorBoard', 'NumPy', 'Docker'].map((tech) => (
              <span
                key={tech}
                className="px-3.5 py-1.5 rounded-full text-xs font-semibold bg-white/5 border border-panel-border text-accent-gold"
              >
                {tech}
              </span>
            ))}
          </div>
        </Card>

      </div>
    </div>
  );
}
