import { useState, useEffect } from 'react';
import { fetchAttention } from '../api/client';
import { SkeletonCard } from '../components/SkeletonLoader';

/**
 * Attention Visualization — heatmap of Transformer cross-attention weights.
 */
export default function AttentionViz({ isBackendOnline }) {
  const [attentionData, setAttentionData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedLayer, setSelectedLayer] = useState(-1);

  useEffect(() => {
    async function load() {
      if (!isBackendOnline) {
        generateDemoAttention();
        return;
      }
      try {
        const data = await fetchAttention(0, selectedLayer);
        setAttentionData(data);
      } catch {
        generateDemoAttention();
      } finally {
        setLoading(false);
      }
    }

    function generateDemoAttention() {
      const T_out = 10;
      const T_in = 10;
      const weights = [];
      for (let i = 0; i < T_out; i++) {
        const row = [];
        for (let j = 0; j < T_in; j++) {
          // Cross attention pattern: query attends to relevant context frames (propagation shift)
          const dist = Math.abs(j - i);
          row.push(Math.exp(-dist * 0.4) * 0.8 + Math.random() * 0.08);
        }
        // Normalize row
        const sum = row.reduce((a, b) => a + b, 0);
        weights.push(row.map((v) => v / sum));
      }

      setAttentionData({
        attention_weights: weights,
        layer: 5,
        num_heads: 8,
      });
      setLoading(false);
    }

    load();
  }, [isBackendOnline, selectedLayer]);

  const getHeatColor = (value) => {
    // Dark blue → cyan → yellow → red
    const v = Math.min(Math.max(value, 0), 1);
    if (v < 0.33) {
      const t = v / 0.33;
      return `rgb(${Math.round(10 + t * 50)}, ${Math.round(15 + t * 100)}, ${Math.round(30 + t * 200)})`;
    } else if (v < 0.66) {
      const t = (v - 0.33) / 0.33;
      return `rgb(${Math.round(60 + t * 195)}, ${Math.round(115 + t * 130)}, ${Math.round(230 - t * 180)})`;
    } else {
      const t = (v - 0.66) / 0.34;
      return `rgb(${Math.round(255)}, ${Math.round(245 - t * 145)}, ${Math.round(50 - t * 50)})`;
    }
  };

  if (loading) {
    return (
      <div className="animate-fade-in space-y-6">
        <h1 className="text-2xl font-bold text-white">Attention Visualization</h1>
        <SkeletonCard />
      </div>
    );
  }

  const weights = attentionData?.attention_weights || [];
  const numRows = weights.length;
  const numCols = weights[0]?.length || 0;

  return (
    <div className="animate-fade-in space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-2">Attention Visualization</h1>
        <p className="text-sm text-slate-400">
          Heatmap of the Transformer's cross-attention weights (context frame → predicted frame) averaged over all heads.
          Rows represent the predicted target frames; columns represent the context input frames.
        </p>
      </div>

      {/* Layer selector */}
      <div className="glass-card p-4 flex items-center gap-4">
        <span className="text-xs font-mono text-slate-400">Layer:</span>
        {[0, 1, 2, 3, 4, 5].map((l) => (
          <button
            key={l}
            onClick={() => setSelectedLayer(l)}
            className={`px-3 py-1 rounded-md text-xs font-mono transition-all ${
              (selectedLayer === -1 && l === 5) || selectedLayer === l
                ? 'bg-electric/20 text-electric'
                : 'text-slate-400 hover:text-white hover:bg-white/5'
            }`}
          >
            Decoder L{l}
          </button>
        ))}
        <span className="text-xs text-slate-400/60 ml-auto">
          {attentionData?.num_heads || 8} heads (averaged)
        </span>
      </div>

      {/* Heatmap */}
      <div className="glass-card p-6 overflow-x-auto">
        <div className="inline-block">
          {/* Column labels */}
          <div className="flex ml-16">
            {Array.from({ length: numCols }).map((_, j) => (
              <div
                key={j}
                className="w-10 h-10 flex items-center justify-center text-[10px] font-mono text-electric/80"
              >
                ctx {j}
              </div>
            ))}
          </div>

          {/* Rows */}
          {weights.map((row, i) => (
            <div key={i} className="flex items-center">
              {/* Row label */}
              <div className="w-16 text-right pr-3 text-[10px] font-mono text-amber/80">
                pred {i}
              </div>

              {/* Cells */}
              {row.map((val, j) => (
                <div
                  key={j}
                  className="w-10 h-10 border border-navy-950/40 transition-all hover:scale-125 hover:z-10 cursor-crosshair"
                  style={{ backgroundColor: getHeatColor(val * 4) }}
                  title={`[pred ${i}] attends to [ctx ${j}]: ${(val * 100).toFixed(1)}%`}
                />
              ))}
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-2 mt-6">
          <span className="text-xs font-mono text-slate-400">Low</span>
          <div className="h-3 w-48 rounded-full" style={{
            background: 'linear-gradient(to right, rgb(10,15,30), rgb(60,115,230), rgb(255,245,50), rgb(255,100,0))',
          }} />
          <span className="text-xs font-mono text-slate-400">High</span>
        </div>

        <div className="flex gap-4 mt-4">
          <span className="text-xs font-mono">
            <span className="inline-block w-2.5 h-2.5 rounded bg-electric/60 mr-1.5" /> Context frames (0–9)
          </span>
          <span className="text-xs font-mono">
            <span className="inline-block w-2.5 h-2.5 rounded bg-amber/60 mr-1.5" /> Predicted frames (10–19)
          </span>
        </div>

        <p className="text-xs text-slate-400/60 mt-4 italic">
          In this Encoder-Decoder architecture, predicted frames cross-attend directly to the context input frames.
          Notice the diagonal alignment showing how future states correlate to spatial wave patterns from specific context time offsets.
        </p>
      </div>
    </div>
  );
}
