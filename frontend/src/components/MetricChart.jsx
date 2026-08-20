import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

/**
 * Reusable chart component for metrics visualization.
 */
export default function MetricChart({
  data = [],
  lines = [{ key: 'value', color: '#3b82f6', label: 'Value' }],
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
      <div className={`glass-card p-6 ${className}`}>
        {title && <h3 className="text-sm font-semibold text-white mb-3">{title}</h3>}
        <div className="skeleton" style={{ height }} />
        {caption && <p className="text-xs text-slate-400/60 mt-2 italic">{caption}</p>}
      </div>
    );
  }

  return (
    <div className={`glass-card p-6 ${className}`}>
      {title && <h3 className="text-sm font-semibold text-white mb-4">{title}</h3>}

      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(59, 130, 246, 0.08)" />
          <XAxis
            dataKey={xKey}
            stroke="#64748b"
            fontSize={11}
            fontFamily="'JetBrains Mono', monospace"
            label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 11 } : undefined}
          />
          <YAxis
            stroke="#64748b"
            fontSize={11}
            fontFamily="'Fira Code', monospace"
            label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: '#b8a99a', fontSize: 11 } : undefined}
          />
          <Tooltip
            contentStyle={{
              background: 'rgba(26, 26, 46, 0.75)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(212, 168, 83, 0.12)',
              borderRadius: '10px',
              fontSize: '12px',
              fontFamily: "'Fira Code', monospace",
              boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.35)',
            }}
            labelStyle={{ color: '#b8a99a' }}
          />
          {lines.length > 1 && (
            <Legend
              wrapperStyle={{ fontSize: '11px', fontFamily: "'Fira Code', monospace" }}
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
        <p className="text-xs text-slate-400/60 mt-3 italic">{caption}</p>
      )}
    </div>
  );
}

/**
 * Sparkline — compact inline chart for per-frame metrics.
 */
export function Sparkline({ data = [], color = '#3b82f6', height = 40, label = '' }) {
  if (data.length === 0) return null;

  const chartData = data.map((v, i) => ({ step: i + 1, value: v }));

  return (
    <div>
      {label && <span className="text-xs font-mono text-slate-400 mb-1 block">{label}</span>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData}>
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
