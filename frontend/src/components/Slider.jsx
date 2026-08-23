import React from 'react';

/**
 * Custom range slider with filled gold track gradient, touch target thumb glow,
 * and layout-shift-free Fira Code numeric readout.
 */
export default function Slider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  unit = '',
  id,
  formatValue = (val) => val.toFixed(2),
}) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className="space-y-2">
      {/* Slider Header info */}
      <div className="flex justify-between items-baseline select-none">
        <label htmlFor={id} className="text-xs font-semibold text-text-secondary font-sans uppercase tracking-wider">
          {label}
        </label>
        
        {/* Monospace value readout to avoid layout jitter during drag */}
        <span className="text-xs font-mono font-medium text-accent-gold font-mono-tabular w-16 text-right">
          {formatValue(value)}{unit}
        </span>
      </div>

      {/* Range Input element */}
      <input
        type="range"
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full custom-slider cursor-pointer"
        style={{
          background: `linear-gradient(to right, var(--color-accent-gold) 0%, var(--color-accent-gold) ${percentage}%, rgba(212, 168, 83, 0.08) ${percentage}%, rgba(212, 168, 83, 0.08) 100%)`,
        }}
      />
    </div>
  );
}
