import React, { useState } from 'react';
import NavRail from './NavRail';
import StatusBar from './StatusBar';

const navItems = [
  { path: '/', label: 'Overview', icon: '◎' },
  { path: '/playground', label: 'Playground', icon: '▶' },
  { path: '/upload', label: 'Image Upload', icon: '📷' },
  { path: '/latent', label: 'Latent Space', icon: '◇' },
  { path: '/attention', label: 'Attention Viz', icon: '⊞' },
  { path: '/training', label: 'Training Log', icon: '📈' },
  { path: '/model', label: 'Model Card', icon: '⚙' },
];

/**
 * Main application Layout frame.
 * Orchestrates NavRail, top workspace header, scrollable workspace canvas,
 * responsive grids, Copilot side panel, and StatusBar.
 */
export default function Layout({ children, isBackendOnline }) {
  const [activeTab, setActiveTab] = useState('Pages');
  const [filterText, setFilterText] = useState('');
  const [zoom, setZoom] = useState(100);

  // Apply zoom factor inline style to content area
  const zoomStyle = {
    transform: `scale(${zoom / 100})`,
    transformOrigin: 'top center',
    width: `${10000 / zoom}%`, // adjust width to avoid scrollbar squishing
  };

  return (
    <div className="relative flex h-screen w-screen overflow-hidden bg-canvas-deep font-sans text-text-primary select-none">
      {/* ─── Ambient Drifting Background Glow Mesh ─── */}
      <div className="ambient-mesh" aria-hidden="true">
        <div className="ambient-glow-1" />
        <div className="ambient-glow-2" />
      </div>

      {/* ─── Persistent Left Navigation Rail ─── */}
      <NavRail
        navItems={navItems}
        isBackendOnline={isBackendOnline}
        filterText={filterText}
        setFilterText={setFilterText}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* ─── Main Content Area ─── */}
      <div className="flex-1 flex flex-col min-w-0 z-10">
        {/* Top Header */}
        <header
          className="h-16 border-b border-panel-border bg-[#15141f]/75 px-8 flex items-center justify-between shrink-0 backdrop-blur-md"
          role="banner"
        >
          <div className="flex items-center gap-2.5">
            <span className="eyebrow-label text-text-secondary/70">Workspace</span>
            <span className="text-[11px] font-semibold text-accent-gold bg-accent-gold/10 px-3 py-1 rounded border border-accent-gold/20 font-mono">
              wave-predictor-v2
            </span>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="text-xs font-mono text-text-secondary hover:text-accent-gold transition-colors focus-visible:ring-2 focus-visible:ring-accent-gold/45 focus:outline-none px-2.5 py-1 rounded hover:bg-white/5"
            >
              API Docs
            </a>
            <button className="bg-accent-gold hover:bg-accent-gold-hover text-canvas-deep text-xs font-semibold px-4 py-1.5 rounded-lg active:scale-95 transition-snappy shadow-md shadow-accent-gold/10 cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-gold/40">
              Launch API
            </button>
          </div>
        </header>

        {/* Scrollable Design Canvas Wrapper */}
        <main className="flex-1 overflow-auto design-canvas p-6 md:p-10 flex flex-col items-center justify-start">
          <div
            className="w-full max-w-6xl space-y-8 md:space-y-10 transition-smooth"
            style={zoomStyle}
          >
            {children}
          </div>
        </main>

        {/* ─── Bottom Status Bar ─── */}
        <StatusBar
          isBackendOnline={isBackendOnline}
          zoom={zoom}
          setZoom={setZoom}
          paramCount="13.3M"
        />
      </div>

      {/* ─── Right Copilot Panel (Desktop only) ─── */}
      <aside
        className="w-80 border-l border-panel-border bg-[#15141f]/50 flex flex-col justify-between shrink-0 hidden lg:flex z-10 backdrop-blur-2xl"
        aria-label="Copilot Panel"
      >
        <div className="p-6 space-y-6 flex-1 overflow-y-auto">
          {/* Panel header */}
          <div className="flex items-center justify-between pb-3 border-b border-panel-border/30">
            <span className="text-xs font-semibold text-text-primary uppercase tracking-wider font-heading">Copilot Panel</span>
            <span className="text-[10px] font-mono bg-accent-gold/10 text-accent-gold px-2 py-0.5 rounded border border-accent-gold/25">
              Active
            </span>
          </div>

          {/* Model info explanation */}
          <div className="space-y-4 text-xs">
            <div className="p-3.5 bg-panel-glass rounded-xl border border-panel-border space-y-1.5">
              <span className="eyebrow-label">Model Pipeline</span>
              <p className="text-text-secondary leading-relaxed font-sans">
                Hybrid CNN-Transformer forecasting future physical propagation in a learned 256-d latent space.
              </p>
            </div>

            {/* Chat notifications list */}
            <div className="space-y-3">
              {[
                {
                  sender: 'System Node',
                  text: 'Training normalized to [-1, 1]. Decoder using Tanh layer. Energy conservation checked.',
                  time: 'Active',
                },
                {
                  sender: 'Copilot',
                  text: 'Encoder-Decoder avoids error propagation. Cross-attention maps predictions to initial inputs.',
                  time: '1m ago',
                },
                {
                  sender: 'Physics solver',
                  text: 'CFL threshold check: courant=0.70. Wave speed=1.0. Boundary mode: Neumann reflecting.',
                  time: 'Just now',
                },
              ].map((msg, i) => (
                <div key={i} className="space-y-1.5">
                  <div className="flex items-center justify-between">
                     <span className="font-semibold text-text-primary text-[11px] font-heading">{msg.sender}</span>
                     <span className="text-[9px] text-text-secondary/50 font-mono">{msg.time}</span>
                  </div>
                  <p className="text-text-secondary leading-relaxed bg-panel-glass p-2.5 rounded-lg border border-panel-border/40 font-sans">
                    {msg.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Copilot Footer */}
        <div className="p-6 border-t border-panel-border/30 bg-white/[0.01] flex items-center justify-between text-xs font-mono text-text-secondary/50">
          <span>TensorBoard logs</span>
          <div className="flex gap-2.5">
            <span className="cursor-help hover:text-accent-gold transition-colors" title="Keyboard Shortcuts">⌨</span>
            <span className="cursor-help hover:text-accent-gold transition-colors" title="Settings">⚙</span>
          </div>
        </div>
      </aside>
    </div>
  );
}
