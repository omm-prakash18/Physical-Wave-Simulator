import React from 'react';
import Card from './Card';

/**
 * MetricStat component for displaying scientific model metrics, counts, and measurements.
 * Displays labels in uppercase monospace eyebrow labels, values in Fira Code tabular-nums
 * with right-alignment, and optional detail text.
 */
export default function MetricStat({
  label,
  value,
  unit = '',
  detail = '',
  accent = 'gold', // 'gold' | 'rose' | 'sage' | 'primary'
  loading = false,
  className = '',
}) {
  const accentTextClasses = {
    gold: 'text-accent-gold',
    rose: 'text-accent-rose',
    sage: 'text-accent-sage',
    primary: 'text-text-primary',
  };

  return (
    <Card className={`animate-fade-in ${className}`}>
      {loading ? (
        <div className="space-y-3 animate-pulse">
          <div className="h-3 w-1/2 bg-white/10 rounded" />
          <div className="h-8 w-2/3 bg-white/10 rounded" />
          <div className="h-3 w-3/4 bg-white/10 rounded" />
        </div>
      ) : (
        <div className="flex flex-col h-full justify-between">
          <div className="space-y-1">
            {/* Monospace Eyebrow Label */}
            <span className="eyebrow-label block">{label}</span>
            
            {/* Value in Fira Code Tabular Numbers */}
            <div className="flex items-baseline font-mono-tabular text-3xl font-semibold tracking-tight">
              <span className={accentTextClasses[accent] || 'text-text-primary'}>
                {value}
              </span>
              {unit && (
                <span className="ml-1 text-sm font-sans font-medium text-text-secondary/70">
                  {unit}
                </span>
              )}
            </div>
          </div>

          {/* Detailed caption */}
          {detail && (
            <p className="text-[11px] text-text-secondary/60 leading-relaxed mt-3 border-t border-panel-border/20 pt-2 font-sans">
              {detail}
            </p>
          )}
        </div>
      )}
    </Card>
  );
}
