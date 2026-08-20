import { useState } from 'react';
import { NavLink } from 'react-router-dom';

const navItems = [
  { path: '/', label: 'Overview', icon: '◎' },
  { path: '/playground', label: 'Playground', icon: '▶' },
  { path: '/latent', label: 'Latent Space', icon: '◇' },
  { path: '/attention', label: 'Attention Viz', icon: '⊞' },
  { path: '/training', label: 'Training Log', icon: '📈' },
  { path: '/model', label: 'Model Card', icon: '⚙' },
];

export default function Layout({ children, isBackendOnline }) {
  const [activeTab, setActiveTab] = useState('Pages');
  const [filterText, setFilterText] = useState('');
  const [zoom, setZoom] = useState(100);

  const filteredNavItems = navItems.filter((item) =>
    item.label.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className="relative flex h-screen w-screen overflow-hidden bg-[#1a1a2e] font-sans">
      {/* ─── WARM BACKGROUND GRADIENTS ─────────────────────────── */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-8%] left-[-8%] w-[500px] h-[500px] rounded-full bg-gradient-to-br from-amber-700/8 to-transparent blur-[130px] animate-pulse" style={{ animationDuration: '9s' }} />
        <div className="absolute bottom-[5%] right-[-8%] w-[600px] h-[600px] rounded-full bg-gradient-to-tr from-rose-700/8 to-transparent blur-[150px] animate-pulse" style={{ animationDuration: '13s' }} />
        <div className="absolute top-[40%] left-[50%] w-[400px] h-[400px] rounded-full bg-gradient-to-bl from-violet-700/6 to-transparent blur-[120px] animate-pulse" style={{ animationDuration: '11s' }} />
      </div>

      {/* ─── LEFT SIDEBAR ─────────────────────────────────── */}
      <aside className="w-60 border-r border-border-color bg-panel-bg flex flex-col justify-between shrink-0 glass-panel-left z-10">
        <div className="p-4 space-y-4">
          {/* macOS controls */}
          <div className="flex items-center justify-between">
            <div className="mac-dots">
              <span className="mac-dot red" />
              <span className="mac-dot yellow" />
              <span className="mac-dot green" />
            </div>
            <span className="text-[10px] font-mono text-text-secondary tracking-wide">Wave Studio</span>
          </div>

          {/* Toggle Tab Switcher */}
          <div className="bg-white/4 p-0.5 rounded-lg flex border border-white/4">
            {['Pages', 'Components'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 text-center py-1.5 rounded-md text-xs font-semibold transition-all ${
                  activeTab === tab
                    ? 'bg-gold/15 text-gold shadow-sm border border-gold/15'
                    : 'text-text-secondary hover:text-text-primary'
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
                <span className="text-sm opacity-70">{icon}</span>
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Guest profile footer */}
        <div className="p-4 border-t border-border-color/50 bg-white/2 space-y-2">
          <p className="text-[11px] text-text-secondary">
            You're running as a <span className="text-gold font-semibold">guest</span>.
          </p>
          <div className="text-[10px] font-mono text-text-secondary/70 flex items-center justify-between">
            <span>RTX 3050 • CUDA</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${isBackendOnline ? 'bg-sage animate-pulse' : 'bg-rose'}`} />
              <span className={isBackendOnline ? 'text-sage' : 'text-rose'}>
                {isBackendOnline ? 'Live' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* ─── MAIN WORKSPACE CONTENT ─────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 z-10">
        {/* Top Header */}
        <header className="h-12 border-b border-border-color bg-panel-bg px-6 flex items-center justify-between shrink-0 glass-panel-header">
          <div className="flex items-center gap-2.5">
            <span className="text-xs text-text-secondary uppercase tracking-widest" style={{ fontFamily: 'var(--font-heading)' }}>Workspace</span>
            <span className="text-xs font-semibold text-text-primary bg-white/5 px-3 py-1 rounded-md border border-gold/10 font-mono">
              wave-predictor-v2
            </span>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="text-xs font-mono text-text-secondary hover:text-gold px-3 py-1.5 rounded-md hover:bg-gold/5 transition-all"
            >
              API Docs
            </a>
            <button className="bg-gold hover:bg-gold-hover text-[#1a1a2e] text-xs font-semibold px-4 py-1.5 rounded-lg transition-all active:scale-95 shadow-lg shadow-gold/15" style={{ fontFamily: 'var(--font-heading)' }}>
              Launch API
            </button>
          </div>
        </header>

        {/* Scrollable Design Canvas */}
        <div className="flex-1 overflow-auto design-canvas p-8 flex flex-col items-center justify-start">
          <div className="w-full max-w-5xl space-y-6">
            <div className="canvas-card p-6 min-h-[500px]">
              {children}
            </div>
          </div>
        </div>

        {/* Bottom Status Bar */}
        <footer className="h-9 border-t border-border-color bg-panel-bg px-4 flex items-center justify-between text-xs font-mono text-text-secondary shrink-0 glass-panel-footer">
          <div className="flex items-center gap-2">
            <span className="text-sage">✓</span>
            <span>Model loaded • 13.3M params</span>
          </div>
          <div className="flex items-center gap-4">
            <span>PyTorch + CUDA</span>
            <div className="flex items-center gap-2 border-l border-border-color pl-4">
              <button onClick={() => setZoom(Math.max(50, zoom - 10))} className="hover:text-gold transition-colors">−</button>
              <span>{zoom}%</span>
              <button onClick={() => setZoom(Math.min(150, zoom + 10))} className="hover:text-gold transition-colors">+</button>
            </div>
          </div>
        </footer>
      </div>

      {/* ─── RIGHT COPILOT SIDEBAR ───────────────── */}
      <aside className="w-72 border-l border-border-color bg-panel-bg flex flex-col justify-between shrink-0 hidden lg:flex glass-panel-right z-10">
        <div className="p-4 space-y-4 flex-1 overflow-y-auto">
          <div className="flex items-center justify-between pb-3 border-b border-border-color">
            <span className="text-xs font-semibold text-text-primary" style={{ fontFamily: 'var(--font-heading)' }}>Copilot Panel</span>
            <span className="text-[10px] font-mono bg-gold/10 text-gold px-2 py-0.5 rounded-md">
              Active
            </span>
          </div>

          {/* Model explanation */}
          <div className="space-y-4 text-xs">
            <div className="p-3.5 bg-white/4 rounded-xl border border-gold/8 space-y-1.5">
              <span className="text-[10px] font-mono text-text-secondary uppercase tracking-wider">Model Info</span>
              <p className="text-text-primary leading-relaxed">
                Hybrid CNN-Transformer predicting future physical simulation states in a compressed 256-d latent space.
              </p>
            </div>

            {/* Simulated chat messages */}
            <div className="space-y-3">
              {[
                {
                  sender: 'System',
                  text: 'Training data normalized to [-1, 1] range. Decoder uses Tanh activation. Energy conservation verified.',
                  time: 'Active',
                },
                {
                  sender: 'Copilot',
                  text: 'Encoder-Decoder architecture prevents error compounding. Cross-attention maps predicted steps to context stages.',
                  time: '1m ago',
                },
                {
                  sender: 'Physics Engine',
                  text: 'CFL condition: courant=0.70. Wave speed=1.0. Boundary conditions: Neumann reflecting.',
                  time: 'Just now',
                },
              ].map((msg, i) => (
                <div key={i} className="space-y-1.5">
                  <div className="flex items-center justify-between">
                     <span className="font-semibold text-text-primary text-[11px]" style={{ fontFamily: 'var(--font-heading)' }}>{msg.sender}</span>
                     <span className="text-[9px] text-text-secondary/60">{msg.time}</span>
                  </div>
                  <p className="text-text-secondary leading-relaxed bg-white/3 p-2.5 rounded-lg border border-gold/6">
                    {msg.text}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom helper */}
        <div className="p-4 border-t border-border-color/50 bg-white/2 flex items-center justify-between text-xs font-mono text-text-secondary/70">
          <span>TensorBoard Logs</span>
          <div className="flex gap-2.5">
            <span className="cursor-help hover:text-gold transition-colors" title="Keyboard Shortcuts">⌨</span>
            <span className="cursor-help hover:text-gold transition-colors" title="Settings">⚙</span>
          </div>
        </div>
      </aside>
    </div>
  );
}
