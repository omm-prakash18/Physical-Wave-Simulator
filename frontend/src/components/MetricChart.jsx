import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

/**
 * Reusable chart component for metrics visualization.
 * Standardised to use design token variables for colors, borders, and monospaced text.
 */
export default function MetricChart({
  data = [],
  lines = [{ key: 'value', color: '#d4a853', label: 'Value' }],
  xKey = 'step',
  xLabel = 'Step',
  yLabel = '',
  title = '',
  caption = '',
  height = 280,
  className = '',
}) {
  if (data.length === 0) {
    return (
      <div className={`rounded-panel bg-panel-glass border border-panel-border p-6 ${className}`}>
        {title && <h3 className="text-sm font-semibold text-text-primary mb-3 font-heading">{title}</h3>}
        <div className="skeleton h-52 animate-pulse w-full bg-white/5" />
        {caption && <p className="text-xs text-text-secondary/50 mt-2 italic font-sans">{caption}</p>}
      </div>
    );
  }

  return (
    <div className={`rounded-panel bg-panel-glass border border-panel-border p-6 ${className}`}>
      {title && (
        <h3 className="text-sm font-semibold text-text-primary mb-4 font-heading tracking-wide">
          {title}
        </h3>
      )}

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(212, 168, 83, 0.04)" />
          <XAxis
            dataKey={xKey}
            stroke="#6b6558"
            fontSize={10}
            fontFamily="var(--font-mono)"
            tickLine={false}
            axisLine={false}
            label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -5, fill: '#6b6558', fontSize: 10, fontFamily: 'var(--font-mono)' } : undefined}
          />
          <YAxis
            stroke="#6b6558"
            fontSize={10}
            fontFamily="var(--font-mono)"
            tickLine={false}
            axisLine={false}
            label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: '#6b6558', fontSize: 10, fontFamily: 'var(--font-mono)' } : undefined}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(21, 20, 31, 0.95)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(212, 168, 83, 0.2)',
              borderRadius: '12px',
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.5)',
            }}
            labelStyle={{ color: '#a8a196', fontWeight: 650 }}
            itemStyle={{ color: '#f5f0e8' }}
          />
          {lines.length > 1 && (
            <Legend
              wrapperStyle={{ fontSize: '10px', fontFamily: 'var(--font-mono)', paddingTop: '10px' }}
            />
          )}
          {lines.map(({ key, color, label }) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={color}
              strokeWidth={2}
              dot={false}
              name={label}
              activeDot={{ r: 4, strokeWidth: 0, fill: color }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      {caption && (
        <p className="text-[10px] text-text-secondary/50 mt-3 italic font-sans leading-relaxed">
          {caption}
        </p>
      )}
    </div>
  );
}

/**
 * Sparkline — compact inline chart for parameter sequences.
 */
export function Sparkline({ data = [], color = '#d4a853', height = 40, label = '' }) {
  if (data.length === 0) return null;

  const chartData = data.map((v, i) => ({ step: i + 1, value: v }));

  return (
    <div>
      {label && <span className="text-[10px] font-mono text-text-secondary/60 mb-2.5 block select-none">{label}</span>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
