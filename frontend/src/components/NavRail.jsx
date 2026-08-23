import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';

/**
 * Navigation Rail - Collapsible left sidebar layout with keyboard navigation support.
 * Can be collapsed to icon-only mode to save space on scientific monitors.
 */
export default function NavRail({
  navItems = [],
  isBackendOnline = false,
  filterText = '',
  setFilterText,
  activeTab = 'Pages',
  setActiveTab,
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const filteredNavItems = navItems.filter((item) =>
    item.label.toLowerCase().includes(filterText.toLowerCase())
  );

  // Keyboard navigation within nav links
  const handleKeyDown = (e) => {
    const links = Array.from(e.currentTarget.querySelectorAll('.sidebar-link-btn'));
    const activeIndex = links.indexOf(document.activeElement);

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const nextIdx = (activeIndex + 1) % links.length;
      links[nextIdx]?.focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prevIdx = (activeIndex - 1 + links.length) % links.length;
      links[prevIdx]?.focus();
    }
  };

  return (
    <aside
      className={`
        border-r border-panel-border bg-panel-glass flex flex-col justify-between shrink-0 transition-all duration-300 z-20 backdrop-blur-2xl
        ${isCollapsed ? 'w-20' : 'w-64'}
      `}
      aria-label="Sidebar Navigation"
    >
      {/* Upper rail section */}
      <div className="p-4 space-y-4">
        {/* macOS Controls & Studio Brand */}
        <div className="flex items-center justify-between">
          {!isCollapsed && (
            <div className="flex items-center gap-1.5 select-none" aria-hidden="true">
              <span className="w-2.5 h-2.5 rounded-full bg-[#E4A499]/85" />
              <span className="w-2.5 h-2.5 rounded-full bg-accent-gold/85" />
              <span className="w-2.5 h-2.5 rounded-full bg-accent-sage/85" />
            </div>
          )}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="p-1 rounded bg-white/5 hover:bg-white/10 text-accent-gold text-xs transition-colors focus-visible:ring-2 focus-visible:ring-accent-gold/45 focus:outline-none"
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? '❯' : '❮'}
          </button>
        </div>

        {/* Brand label */}
        {!isCollapsed && (
          <div className="pt-1">
            <span className="eyebrow-label text-text-primary/75">Simulation Studio</span>
          </div>
        )}

        {/* Tab switch pages / components */}
        {!isCollapsed && (
          <div className="bg-white/5 p-0.5 rounded-lg flex border border-panel-border" role="tablist">
            {['Pages', 'Components'].map((tab) => (
              <button
                key={tab}
                role="tab"
                aria-selected={activeTab === tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 text-center py-1 rounded-md text-[11px] font-semibold transition-snappy ${
                  activeTab === tab
                    ? 'bg-accent-gold/15 text-accent-gold border border-accent-gold/25'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        )}

        {/* Filter input */}
        {!isCollapsed && (
          <input
            type="text"
            placeholder="Search panels..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            className="w-full design-input"
            aria-label="Filter navigation panels"
          />
        )}

        {/* Navigation list */}
        <nav
          className="space-y-1 pt-2"
          onKeyDown={handleKeyDown}
          aria-label="Primary"
        >
          {filteredNavItems.map(({ path, label, icon }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) => `
                sidebar-link-btn flex items-center gap-3 p-2.5 rounded-lg text-xs font-semibold select-none transition-snappy
                focus-visible:ring-2 focus-visible:ring-accent-gold/45 focus:outline-none
                ${isActive
                  ? 'bg-accent-gold text-canvas-deep shadow-glow-gold'
                  : 'text-text-secondary hover:text-text-primary hover:bg-white/5'
                }
              `}
              title={label}
            >
              <span className="text-sm shrink-0" aria-hidden="true">{icon}</span>
              {!isCollapsed && <span className="truncate">{label}</span>}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Rail Profile / System status Footer */}
      <div className="p-4 border-t border-panel-border/30 bg-white/[0.01]">
        {isCollapsed ? (
          <div className="flex flex-col items-center gap-3">
            <span
              className={`w-2 h-2 rounded-full ${isBackendOnline ? 'bg-accent-sage animate-pulse' : 'bg-accent-rose'}`}
              title={isBackendOnline ? 'Backend Online' : 'Backend Offline'}
            />
            <span className="text-[10px] font-mono text-text-secondary" title="RTX 3050">GPU</span>
          </div>
        ) : (
          <div className="space-y-2.5 select-none">
            <div className="text-[10px] font-mono text-text-secondary/60 flex items-center justify-between">
              <span>RTX 3050 • CUDA</span>
              <div className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${isBackendOnline ? 'bg-accent-sage animate-pulse' : 'bg-accent-rose'}`} />
                <span className={isBackendOnline ? 'text-accent-sage font-medium' : 'text-accent-rose'}>
                  {isBackendOnline ? 'Connected' : 'Offline'}
                </span>
              </div>
            </div>
            <p className="text-[10px] text-text-tertiary font-sans leading-tight">
              Hardware node: LOCAL-V2
            </p>
          </div>
        )}
      </div>
    </aside>
  );
}
