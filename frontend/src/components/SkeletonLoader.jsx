import React from 'react';

/**
 * Basic skeleton row loader components.
 */
export default function SkeletonLoader({ className = '', rows = 3 }) {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="skeleton"
          style={{
            height: i === 0 ? '24px' : '16px',
            width: i === 0 ? '60%' : `${70 + ((i * 17) % 25)}%`,
          }}
        />
      ))}
    </div>
  );
}

/**
 * Standardised card block skeleton state.
 */
export function SkeletonCard({ className = '' }) {
  return (
    <div className={`rounded-panel bg-panel-glass border border-panel-border p-6 shadow-panel ${className}`}>
      <div className="skeleton h-4 w-1/3 mb-4" />
      <div className="skeleton h-40 w-full mb-3" />
      <div className="skeleton h-3 w-2/3" />
    </div>
  );
}

/**
 * Standardised chart element skeleton state.
 */
export function SkeletonChart({ className = '' }) {
  return (
    <div className={`rounded-panel bg-panel-glass border border-panel-border p-6 shadow-panel ${className}`}>
      <div className="skeleton h-4 w-1/4 mb-4" />
      <div className="skeleton h-60 w-full" />
    </div>
  );
}
