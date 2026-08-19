import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

const navItems = [
  { path: '/', label: '/overview', icon: '◎' },
  { path: '/playground', label: '/playground', icon: '▶' },
  { path: '/latent', label: '/latent-space', icon: '◇' },
  { path: '/attention', label: '/attention-viz', icon: '⊞' },
  { path: '/training', label: '/training-log', icon: '📈' },
  { path: '/model', label: '/model-card', icon: '⚙' },
];

export default function Layout({ children, isBackendOnline }) {
  const location = useLocation();
  const [activeTab, setActiveTab] = useState('Pages');
  const [filterText, setFilterText] = useState('');
  const [zoom, setZoom] = useState(100);

  const filteredNavItems = navItems.filter((item) =>
    item.label.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-canvas-bg font-sans">
      {/* ─── LEFT SIDEBAR ─────────────────────────────────── */}
      <aside className="w-60 border-r border-border-color bg-panel-bg flex flex-col justify-between shrink-0">
        <div className="p-4 space-y-4">
          {/* macOS controls */}
          <div className="flex items-center justify-between">
            <div className="mac-dots">
              <span className="mac-dot red" />
              <span className="mac-dot yellow" />
              <span className="mac-dot green" />
            </div>
            <span className="text-[10px] font-mono text-text-secondary">Forge IDE</span>
          </div>

          {/* Toggle Tab Switcher */}
          <div className="bg-[#1c1c1f] p-0.5 rounded-lg flex border border-border-color">
            {['Pages', 'Components'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 text-center py-1.5 rounded-md text-xs font-semibold transition-all ${
                  activeTab === tab
                    ? 'bg-[#27272a] text-white shadow-sm'
                    : 'text-text-secondary hover:text-white'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Search/Filter */}
          <input
            type="text"
            placeholder="Filter pages..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            className="w-full design-input"
          />

          {/* Page Links */}
          <nav className="space-y-1">
            {filteredNavItems.map(({ path, label, icon }) => (
              <NavLink
                key={path}
                to={path}
                className={({ isActive }) =>
                  `sidebar-link ${isActive ? 'active' : ''}`
                }
              >
                <span className="text-xs opacity-75">{icon}</span>
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Guest profile footer info */}
        <div className="p-4 border-t border-border-color bg-[#0e0e10] space-y-2">
          <p className="text-[11px] text-text-secondary">
            You're running as a <span className="text-accent-orange font-semibold">guest</span>.
          </p>
          <div className="text-[10px] font-mono text-text-secondary/70 flex items-center justify-between">
            <span>Model: cu126-RTX3050</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${isBackendOnline ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
              <span className={isBackendOnline ? 'text-emerald-400' : 'text-red-400'}>
                {isBackendOnline ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* ─── MAIN WORKSPACE CONTENT ─────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Workspace Bar */}
        <header className="h-12 border-b border-border-color bg-panel-bg px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-text-secondary uppercase tracking-wider">Workspace:</span>
            <span className="text-xs font-semibold text-white bg-[#1c1c1f] px-2.5 py-1 rounded-md border border-border-color font-mono">
              wave-latent-predictor-v1
            </span>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="text-xs font-mono text-text-secondary hover:text-white px-3 py-1.5 rounded-md hover:bg-white/5 transition-all"
            >
              /api-docs
            </a>
            <button className="bg-accent-orange hover:bg-accent-orange-hover text-white text-xs font-semibold px-4 py-1.5 rounded-md transition-all active:scale-95 shadow-lg shadow-accent-orange/20">
              Launch API
            </button>
          </div>
        </header>

        {/* Scrollable Design Canvas */}
        <div className="flex-1 overflow-auto design-canvas p-8 flex flex-col items-center justify-start">
          <div className="w-full max-w-5xl space-y-6">
            {/* Viewport viewport wrapper card */}
            <div className="canvas-card p-6 min-h-[500px]">
              {children}
            </div>
          </div>
        </div>

        {/* Bottom Status Canvas Bar */}
        <footer className="h-9 border-t border-border-color bg-[#0e0e10] px-4 flex items-center justify-between text-xs font-mono text-text-secondary shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-emerald-500">✓</span>
            <span>deployed! wave.u8.ai</span>
          </div>
          <div className="flex items-center gap-4">
            <span>Params: 13.3M</span>
            <div className="flex items-center gap-2 border-l border-border-color pl-4">
              <button onClick={() => setZoom(Math.max(50, zoom - 10))} className="hover:text-white">-</button>
              <span>{zoom}%</span>
              <button onClick={() => setZoom(Math.min(150, zoom + 10))} className="hover:text-white">+</button>
            </div>
          </div>
        </footer>
      </div>

      {/* ─── RIGHT COPILOT/ASSISTANT SIDEBAR ───────────────── */}
      <aside className="w-72 border-l border-border-color bg-panel-bg flex flex-col justify-between shrink-0 hidden lg:flex">
        <div className="p-4 space-y-4 flex-1 overflow-y-auto">
          <div className="flex items-center justify-between pb-3 border-b border-border-color">
            <span className="text-xs font-semibold text-white font-mono">← Copilot Panel</span>
            <span className="text-[10px] font-mono bg-accent-orange/10 text-accent-orange px-1.5 py-0.5 rounded">
              Active
            </span>
          </div>

          {/* Model cards explanation */}
          <div className="space-y-4 text-xs">
            <div className="p-3 bg-[#1c1c1f] rounded-lg border border-border-color space-y-1">
              <span className="text-[10px] font-mono text-text-secondary uppercase">Explorer Log</span>
              <p className="text-white leading-relaxed font-mono">
                The Hybrid CNN-Transformer model predicts future physical simulation states directly in the compressed 256-dimensional latent space.
              </p>
            </div>

            {/* Assistant simulated chat messages */}
            <div className="space-y-3">
              {[
                {
                  sender: 'System Log',
                  text: 'Training data successfully normalized to [-1, 1] range to match Tanh decoder activation. Checked energy conservation.',
                  time: 'Active',
                },
                {
                  sender: 'Copilot',
                  text: 'The Encoder-Decoder architecture prevents error compounding in long prediction sequences. Cross-attention maps predicted steps to specific context stages.',
                  time: '1m ago',
                },
                {
                  sender: 'Simulation Engine',
                  text: 'CFL condition checks: courant=0.70. Wave speed set to 1.0. Boundary conditions are Neumann reflecting.',
                  time: 'Just now',
                },
              ].map((msg, i) => (
                <div key={i} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white text-[11px] font-mono">{msg.sender}</span>
                    <span className="text-[9px] text-text-secondary/70">{msg.time}</span>
                  </div>
                  <p className="text-text-secondary leading-relaxed bg-[#1c1c1f]/40 p-2.5 rounded-md border border-white/3">
                    {msg.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom helper action buttons */}
        <div className="p-4 border-t border-border-color bg-[#0e0e10] flex items-center justify-between text-xs font-mono text-text-secondary/80">
          <span>Logs: TensorBoard</span>
          <div className="flex gap-2">
            <span className="cursor-help hover:text-white" title="View Keybinds">⌨</span>
            <span className="cursor-help hover:text-white" title="Settings">⚙</span>
          </div>
        </div>
      </aside>
    </div>
  );
}
