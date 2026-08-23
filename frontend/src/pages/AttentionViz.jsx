import React, { useState, useEffect } from 'react';
import Card from '../components/Card';
import { fetchAttention } from '../api/client';

/**
 * Attention Visualization - Renders a 2D cross-attention weight matrix.
 * Implements per-row vs global scale normalization, colorblind-friendly blue-gold scale,
 * interactive legend details, and active cell hover readouts.
 */
export default function AttentionViz({ isBackendOnline }) {
  const [attentionData, setAttentionData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedLayer, setSelectedLayer] = useState(5);
  
  // Normalization Scale state
  const [isRowNormalized, setIsRowNormalized] = useState(false);
  
  // Active Hovered Cell info
  const [hoveredCell, setHoveredCell] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
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
          // Simulated attention flow: queries attend to spatial wave shift propagation
          const dist = Math.abs(j - i);
          const weight = Math.exp(-dist * 0.45) * 0.75 + (i === j ? 0.2 : 0) + Math.random() * 0.04;
          row.push(weight);
        }
        // Row softmax normalization simulation
        const sum = row.reduce((a, b) => a + b, 0);
        weights.push(row.map((v) => v / sum));
      }

      setAttentionData({
        attention_weights: weights,
        layer: selectedLayer,
        num_heads: 8,
      });
      setLoading(false);
    }

    load();
  }, [isBackendOnline, selectedLayer]);

  // Colorblind-safe diverging Blue -> Neutral -> Gold scale
  // Dark Navy Blue (low attention) -> Warm Gray (neutral) -> Gold (high attention)
  const getHeatColor = (value, rowMax = 1) => {
    const denom = isRowNormalized ? rowMax : 0.25; // standard max expected weight is ~0.25 without normalization
    const v = Math.min(Math.max(value / denom, 0), 1);
    
    let r, g, b;
    if (v < 0.5) {
      const t = v * 2; // interpolate from Blue [35, 52, 85] to Muted Gray [107, 101, 88]
      r = Math.round(35 + t * (107 - 35));
      g = Math.round(52 + t * (101 - 52));
      b = Math.round(85 + t * (88 - 85));
    } else {
      const t = (v - 0.5) * 2; // interpolate from Muted Gray [107, 101, 88] to Gold [212, 168, 83]
      r = Math.round(107 + t * (212 - 107));
      g = Math.round(101 + t * (168 - 101));
      b = Math.round(88 + t * (83 - 88));
    }
    return `rgb(${r}, ${g}, ${b})`;
  };

  const weights = attentionData?.attention_weights || [];
  const numCols = weights[0]?.length || 0;
  const rowMaximums = weights.map(row => Math.max(...row, 1e-9));

  return (
    <div className="animate-fade-in space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold font-heading text-text-primary mb-1">
          Transformer Attention Field
        </h1>
        <p className="text-xs text-text-secondary font-sans leading-relaxed">
          Heatmap visualization of Cross-Attention weights (context frames inputs attend to predicted outputs) averaged over all 8 attention heads.
        </p>
      </div>

      {/* Layer Selection and Normalizer controls */}
      <Card eyebrow="Visualization Settings" title="Attention Layer & Scale Controls">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pt-2 select-none">
          {/* Layer tabs */}
          <div className="flex flex-wrap gap-1.5" role="tablist">
            {[0, 1, 2, 3, 4, 5].map((l) => (
              <button
                key={l}
                role="tab"
                aria-selected={selectedLayer === l}
                onClick={() => setSelectedLayer(l)}
                className={`px-3 py-1 rounded text-xs font-mono transition-snappy cursor-pointer focus:outline-none focus:ring-1 focus:ring-accent-gold/40 ${
                  selectedLayer === l
                    ? 'bg-accent-gold/15 text-accent-gold border border-accent-gold/30'
                    : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
                }`}
              >
                Decoder L{l}
              </button>
            ))}
          </div>

          {/* Normalization Toggle */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-text-secondary font-sans">Scale Normalization:</span>
            <div className="flex bg-white/5 rounded border border-panel-border p-0.5">
              <button
                onClick={() => setIsRowNormalized(false)}
                className={`px-2.5 py-0.5 rounded text-[11px] font-semibold transition-snappy cursor-pointer ${
                  !isRowNormalized
                    ? 'bg-accent-gold text-canvas-deep'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                Global (Max 0.25)
              </button>
              <button
                onClick={() => setIsRowNormalized(true)}
                className={`px-2.5 py-0.5 rounded text-[11px] font-semibold transition-snappy cursor-pointer ${
                  isRowNormalized
                    ? 'bg-accent-gold text-canvas-deep'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                Row-wise (Max 1.0)
              </button>
            </div>
          </div>
        </div>
      </Card>

      {/* Main Heatmap Visualization Card */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Heatmap Grid Panel */}
        <Card className="lg:col-span-3 overflow-x-auto" eyebrow="Weights Matrix" title={`Decoder Layer ${selectedLayer} cross-attention`}>
          {loading ? (
            <div className="h-96 flex flex-col items-center justify-center animate-pulse">
              <div className="w-80 h-80 bg-white/5 rounded-lg" />
            </div>
          ) : (
            <div className="inline-block min-w-full pb-4">
              {/* Column (Context) headers */}
              <div className="flex ml-16">
                {Array.from({ length: numCols }).map((_, j) => (
                  <div
                    key={j}
                    className="w-10 h-10 flex items-center justify-center text-[9px] font-mono text-accent-rose font-medium select-none"
                  >
                    ctx {j + 1}
                  </div>
                ))}
              </div>

              {/* Matrix rows */}
              {weights.map((row, i) => (
                <div key={i} className="flex items-center">
                  {/* Row (Prediction) labels */}
                  <div className="w-16 text-right pr-3 text-[9px] font-mono text-accent-gold font-medium select-none">
                    pred {i + 1}
                  </div>

                  {/* Weight cells */}
                  {row.map((val, j) => {
                    const color = getHeatColor(val, rowMaximums[i]);
                    const isCellHovered = hoveredCell && hoveredCell.row === i && hoveredCell.col === j;

                    return (
                      <div
                        key={j}
                        onMouseEnter={() => setHoveredCell({ row: i, col: j, val })}
                        onMouseLeave={() => setHoveredCell(null)}
                        className={`
                          w-10 h-10 border border-[#15141f]/80 transition-all duration-100 cursor-crosshair
                          ${isCellHovered ? 'scale-125 z-20 shadow-lg shadow-black/80 border-white/50' : 'hover:scale-110 hover:z-10'}
                        `}
                        style={{ backgroundColor: color }}
                        aria-label={`Attention weight pred step ${i + 1} to context step ${j + 1}: ${(val * 100).toFixed(1)}%`}
                      />
                    );
                  })}
                </div>
              ))}

              {/* Colorbar Interactive Legend */}
              <div className="flex items-center gap-3 mt-6 border-t border-panel-border/20 pt-4 select-none">
                <span className="text-[10px] font-mono text-text-secondary">Low (Blue)</span>
                <div
                  className="h-3 w-48 rounded-full border border-panel-border shadow-inner"
                  style={{
                    background: 'linear-gradient(to right, rgb(35, 52, 85) 0%, rgb(107, 101, 88) 50%, rgb(212, 168, 83) 100%)',
                  }}
                />
                <span className="text-[10px] font-mono text-text-secondary">
                  High ({isRowNormalized ? 'Row Max' : '0.25+'})
                </span>
                
                <div className="ml-auto flex gap-4 text-[10px] font-mono">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-accent-rose/20 border border-accent-rose/30" />
                    Context frames 1–10
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-accent-gold/20 border border-accent-gold/30" />
                    Predicted frames 11–20
                  </span>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Floating Cell Inspector Info Panel */}
        <Card className="lg:col-span-1" eyebrow="Inspector" title="Weight Details">
          {hoveredCell ? (
            <div className="space-y-4 animate-fade-in">
              <div className="space-y-1">
                <span className="eyebrow-label text-accent-gold">Source Query</span>
                <p className="text-sm font-semibold text-text-primary font-heading">
                  Future Prediction Step {hoveredCell.row + 1}
                </p>
              </div>

              <div className="space-y-1">
                <span className="eyebrow-label text-accent-rose">Target Context</span>
                <p className="text-sm font-semibold text-text-primary font-heading">
                  Context Frame Input {hoveredCell.col + 1}
                </p>
              </div>

              <div className="space-y-1 pt-2 border-t border-panel-border/30">
                <span className="eyebrow-label">Attention Weight</span>
                <div className="text-3xl font-bold font-mono-tabular text-accent-gold">
                  {(hoveredCell.val * 100).toFixed(2)}%
                </div>
                <p className="text-[10px] text-text-secondary/60 leading-relaxed font-sans mt-1">
                  Represents the percentage of activation energy the transformer query draws from this specific input.
                </p>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col justify-center items-center text-center py-10 text-text-secondary/40 select-none">
              <span className="text-2xl mb-2">🔍</span>
              <p className="text-xs font-sans">Hover over matrix cells to inspect causal connection weights.</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
