import React from 'react';

/**
 * Slim status bar for displaying live backend metrics, param size, GPU nodes,
 * and standard workspace zoom control options.
 */
export default function StatusBar({
  isBackendOnline = false,
  zoom = 100,
  setZoom,
  paramCount = '13.3M',
}) {
  return (
    <footer
      className="h-9 border-t border-panel-border bg-[#15141f]/75 px-4 flex items-center justify-between text-[11px] font-mono text-text-secondary shrink-0 backdrop-blur-md"
      role="contentinfo"
    >
      {/* Left section: model status */}
      <div className="flex items-center gap-2">
        <span className={isBackendOnline ? 'text-accent-sage' : 'text-accent-rose'}>
          {isBackendOnline ? '●' : '○'}
        </span>
        <span>
          Model state: <strong className="text-text-primary">Ready</strong> • {paramCount} params
        </span>
      </div>

      {/* Right section: node specs & zoom settings */}
      <div className="flex items-center gap-4">
        <span>FastAPI + PyTorch GPU</span>
        
        {/* Zoom controls */}
        <div className="flex items-center gap-2 border-l border-panel-border pl-4">
          <button
            onClick={() => setZoom(Math.max(50, zoom - 10))}
            className="hover:text-accent-gold transition-colors px-1 cursor-pointer focus:outline-none focus:text-accent-gold"
            aria-label="Zoom out"
            title="Zoom out"
          >
            −
          </button>
          <span className="w-10 text-center select-none text-text-primary/80">
            {zoom}%
          </span>
          <button
            onClick={() => setZoom(Math.min(150, zoom + 10))}
            className="hover:text-accent-gold transition-colors px-1 cursor-pointer focus:outline-none focus:text-accent-gold"
            aria-label="Zoom in"
            title="Zoom in"
          >
            +
          </button>
        </div>
      </div>
    </footer>
  );
}
