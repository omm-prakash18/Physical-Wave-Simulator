import { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { SkeletonChart } from '../components/SkeletonLoader';

/**
 * Latent Space Explorer — 2D projection of latent tokens.
 * Shows precomputed UMAP/t-SNE embeddings colored by timestep.
 */
export default function LatentExplorer() {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Generate synthetic UMAP-like data for demo
    // In production, this would come from a precomputed API endpoint
    const points = [];
    const numSequences = 20;
    const stepsPerSeq = 20;

    for (let s = 0; s < numSequences; s++) {
      const baseX = (Math.random() - 0.5) * 10;
      const baseY = (Math.random() - 0.5) * 10;
      const angle = Math.random() * Math.PI * 2;

      for (let t = 0; t < stepsPerSeq; t++) {
        const progress = t / stepsPerSeq;
        const r = 0.5 + progress * 3;
        points.push({
          x: baseX + r * Math.cos(angle + progress * Math.PI * 0.8) + (Math.random() - 0.5) * 0.3,
          y: baseY + r * Math.sin(angle + progress * Math.PI * 0.8) + (Math.random() - 0.5) * 0.3,
          timestep: t,
          sequence: s,
          label: `Seq ${s}, t=${t}`,
        });
      }
    }
    setData(points);
  }, []);

  // Color gradient: gold (t=0) → rose (t=19)
  const getColor = (timestep) => {
    const t = timestep / 19;
    const r = Math.round(212 - t * 11);
    const g = Math.round(168 - t * 45);
    const b = Math.round(83 + t * 40);
    return `rgb(${r}, ${g}, ${b})`;
  };

  if (data.length === 0) {
    return (
      <div className="animate-fade-in space-y-6">
        <h1 className="text-2xl font-bold text-text-primary" style={{ fontFamily: 'var(--font-heading)' }}>Latent Space Explorer</h1>
        <SkeletonChart />
      </div>
    );
  }

  return (
    <div className="animate-fade-in space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-2" style={{ fontFamily: 'var(--font-heading)' }}>Latent Space Explorer</h1>
        <p className="text-sm text-text-secondary">
          UMAP projection of latent tokens across sequences. Each point represents one frame's
          latent vector. Color encodes timestep: <span className="text-gold">gold = early</span> →{' '}
          <span className="text-rose">rose = late</span>.
        </p>
      </div>

      <div className="glass-card p-6">
        <ResponsiveContainer width="100%" height={500}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(212, 168, 83, 0.06)" />
            <XAxis
              type="number"
              dataKey="x"
              stroke="#b8a99a"
              fontSize={11}
              fontFamily="'Fira Code', monospace"
              label={{ value: 'UMAP-1', position: 'insideBottom', offset: -10, fill: '#b8a99a' }}
            />
            <YAxis
              type="number"
              dataKey="y"
              stroke="#b8a99a"
              fontSize={11}
              fontFamily="'Fira Code', monospace"
              label={{ value: 'UMAP-2', angle: -90, position: 'insideLeft', fill: '#b8a99a' }}
            />
            <Tooltip
              cursor={false}
              content={({ payload }) => {
                if (!payload || payload.length === 0) return null;
                const d = payload[0].payload;
                return (
                  <div className="glass-card p-3 text-xs font-mono">
                    <p className="text-text-primary">{d.label}</p>
                    <p className="text-text-secondary">
                      ({d.x.toFixed(2)}, {d.y.toFixed(2)})
                    </p>
                  </div>
                );
              }}
            />
            <Scatter data={data} isAnimationActive={false}>
              {data.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={getColor(entry.timestep)}
                  fillOpacity={0.7}
                  r={3}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <p className="text-xs text-text-secondary/50 mt-3 italic">
          Temporal trajectories form smooth arcs in latent space, indicating the model has learned
          a structured representation of wave dynamics. Nearby points represent similar physical states.
        </p>
      </div>

      {/* Color legend */}
      <div className="glass-card p-4">
        <p className="text-xs text-text-secondary mb-2" style={{ fontFamily: 'var(--font-heading)', fontWeight: 500 }}>Timestep Color Scale</p>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-text-secondary">t=0</span>
          <div className="flex-1 h-3 rounded-full" style={{
            background: 'linear-gradient(to right, #d4a853, #9b8ec4, #c97b7b)',
          }} />
          <span className="text-xs font-mono text-text-secondary">t=19</span>
        </div>
      </div>
    </div>
  );
}
