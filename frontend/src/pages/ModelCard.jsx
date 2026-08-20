/**
 * Model Card — architecture summary, parameter breakdown, and dataset description.
 */
export default function ModelCard() {
  const architecture = [
    {
      component: 'CNN Encoder',
      params: '~455K',
      details: '4-stage Conv-BN-SiLU downsampler (64→32→16→8→4), global avg pool, linear projection to d=256',
    },
    {
      component: 'Causal Transformer',
      params: '~11.1M',
      details: '6 encoder + 6 decoder layers, 8 heads, d_model=256, d_ff=1024, pre-norm, sinusoidal PE, learned query tokens',
    },
    {
      component: 'CNN Decoder',
      params: '~1.75M',
      details: 'Linear → reshape, 4-stage ConvTranspose-BN-SiLU upsampler (4→8→16→32→64), Conv + Tanh',
    },
  ];

  const lossComponents = [
    { name: 'L_recon', weight: '1.0', desc: 'L1 pixel reconstruction loss between predicted and GT frames' },
    { name: 'L_latent', weight: '1.0', desc: 'MSE in latent space — primary Transformer training signal' },
    { name: 'L_perceptual', weight: '0.1', desc: 'Feature-space L1 using frozen encoder layers — prevents blurring' },
    { name: 'L_temporal', weight: '0.5', desc: 'Motion consistency loss — penalizes frame-to-frame flicker' },
  ];

  return (
    <div className="animate-fade-in space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-2" style={{ fontFamily: 'var(--font-heading)' }}>Model Card</h1>
        <p className="text-sm text-text-secondary">
          Complete technical specification and architecture details.
        </p>
      </div>

      {/* Architecture diagram */}
      <div className="glass-card p-6 overflow-x-auto">
        <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Architecture Overview</h2>
        <pre className="text-xs font-mono text-gold/80 leading-relaxed whitespace-pre">
{`  Input Frames (B, T_in, 1, 64, 64)
         │
    ┌────┴────┐
    │  CNN    │ ← 4-stage Conv-BN-SiLU, stride 2
    │ Encoder │    64→32→16→8→4, GAP, Linear
    └────┬────┘
         │
  Latent Tokens (B, T_in, 256)
         │
    ┌────┴─────────┐
    │  Encoder-     │ ← 6+6 layers, 8 heads
    │  Decoder      │   Learned query tokens for T_out
    │  Transformer  │   Sinusoidal positional encoding
    │               │   Pre-norm, GELU FFN
    └────┬─────────┘
         │
  Predicted Latents (B, T_out, 256)
         │
    ┌────┴────┐
    │  CNN    │ ← Mirror of encoder
    │ Decoder │    Linear, ConvT-BN-SiLU ×4, Conv + Tanh
    └────┬────┘
         │
  Predicted Frames (B, T_out, 1, 64, 64)`}
        </pre>
      </div>

      {/* Parameter breakdown */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Parameter Breakdown</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gold/10">
                <th className="text-left text-xs font-mono text-text-secondary py-2 pr-4">Component</th>
                <th className="text-right text-xs font-mono text-text-secondary py-2 pr-4">Parameters</th>
                <th className="text-left text-xs font-mono text-text-secondary py-2">Details</th>
              </tr>
            </thead>
            <tbody>
              {architecture.map(({ component, params, details }) => (
                <tr key={component} className="border-b border-white/4">
                  <td className="py-3 pr-4 font-medium text-text-primary">{component}</td>
                  <td className="py-3 pr-4 text-right font-mono text-gold">{params}</td>
                  <td className="py-3 text-text-secondary text-xs">{details}</td>
                </tr>
              ))}
              <tr className="bg-gold/4">
                <td className="py-3 pr-4 font-bold text-text-primary">Total</td>
                <td className="py-3 pr-4 text-right font-mono font-bold text-rose">~13.3M</td>
                <td className="py-3 text-text-secondary text-xs">Production-ready architecture with full encoder-decoder Transformer</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Loss function */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Loss Function</h2>
        <p className="text-xs font-mono text-text-secondary mb-4 bg-white/3 rounded-xl p-3.5 border border-gold/8">
          L_total = λ₁·L_recon + λ₂·L_latent + λ₃·L_perceptual + λ₄·L_temporal
        </p>
        <div className="space-y-3">
          {lossComponents.map(({ name, weight, desc }) => (
            <div key={name} className="flex items-start gap-3">
              <span className="text-xs font-mono text-gold bg-gold/10 px-2 py-0.5 rounded-md shrink-0">
                λ={weight}
              </span>
              <div>
                <span className="text-sm font-mono text-text-primary">{name}</span>
                <p className="text-xs text-text-secondary mt-0.5">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Dataset: 2D Wave Equation</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
          {[
            { label: 'Sequences', value: '10,000' },
            { label: 'Frames/seq', value: '20' },
            { label: 'Resolution', value: '64×64' },
            { label: 'Channels', value: '1 (grayscale)' },
            { label: 'Boundary', value: 'Reflecting' },
            { label: 'Split', value: '80/10/10' },
            { label: 'Physics', value: '2D wave eq.' },
            { label: 'Deterministic', value: 'Yes (seeded)' },
          ].map(({ label, value }) => (
            <div key={label}>
              <span className="text-text-secondary">{label}</span>
              <p className="text-text-primary mt-0.5">{value}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-text-secondary/50 mt-4 italic">
          Gaussian pulse initialization with random center, width, and amplitude.
          Finite-difference solver on a uniform grid. CFL-stable time stepping.
        </p>
      </div>

      {/* Training details */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Training Details</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-xs font-mono">
          {[
            { label: 'Optimizer', value: 'AdamW' },
            { label: 'Learning Rate', value: '3e-4' },
            { label: 'Schedule', value: 'Cosine + warmup' },
            { label: 'Batch Size', value: '32' },
            { label: 'Precision', value: 'FP16 mixed' },
            { label: 'Teacher Forcing', value: '1.0→0.0 (linear)' },
            { label: 'Gradient Clip', value: '1.0 max norm' },
            { label: 'Weight Decay', value: '0.01' },
            { label: 'Activation', value: 'Tanh (decoder)' },
          ].map(({ label, value }) => (
            <div key={label}>
              <span className="text-text-secondary">{label}</span>
              <p className="text-text-primary mt-0.5">{value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Tech stack */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-semibold text-text-primary mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Technology Stack</h2>
        <div className="flex flex-wrap gap-2">
          {['PyTorch', 'FastAPI', 'React', 'Tailwind CSS', 'Recharts', 'TensorBoard', 'NumPy', 'SciPy'].map((tech) => (
            <span
              key={tech}
              className="px-3 py-1.5 rounded-full text-xs font-medium bg-gradient-to-r from-gold/10 to-rose/10 text-gold border border-gold/15"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
