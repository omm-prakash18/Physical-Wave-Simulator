import React from 'react';

/**
 * Standardised Design System Card component.
 * Features translucent cream-glass background, warm subtle borders,
 * backdrop blur, and top accent colors corresponding to dataset classes.
 */
export default function Card({
  children,
  className = '',
  accent = 'none', // 'gold' | 'rose' | 'sage' | 'none'
  title = '',
  eyebrow = '',
  footer = null,
  onClick = null,
}) {
  const isClickable = typeof onClick === 'function';

  // Accent mapping to top borders
  const accentClasses = {
    none: '',
    gold: 'border-t-2 border-t-accent-gold',
    rose: 'border-t-2 border-t-accent-rose',
    sage: 'border-t-2 border-t-accent-sage',
  };

  const Component = isClickable ? 'button' : 'div';

  return (
    <Component
      onClick={onClick}
      className={`
        relative flex flex-col w-full text-left rounded-panel bg-panel-glass backdrop-blur-xl border border-panel-border shadow-panel transition-snappy
        ${accentClasses[accent] || ''}
        ${isClickable ? 'hover:bg-white/[0.06] hover:border-accent-gold/30 hover:shadow-glow-gold focus:outline-none focus:ring-2 focus:ring-accent-gold/40 cursor-pointer active:scale-[0.99]' : ''}
        ${className}
      `}
    >
      {/* Header section if title or eyebrow is provided */}
      {(title || eyebrow) && (
        <div className="px-6 pt-5 pb-3 md:px-8 md:pt-6 md:pb-4 border-b border-panel-border/30">
          {eyebrow && <p className="eyebrow-label mb-1">{eyebrow}</p>}
          {title && <h3 className="text-text-primary font-semibold">{title}</h3>}
        </div>
      )}

      {/* Main card body */}
      <div className="flex-1 p-6 md:p-8">
        {children}
      </div>

      {/* Footer section if provided */}
      {footer && (
        <div className="px-6 py-4 md:px-8 md:py-5 border-t border-panel-border/30 bg-white/[0.01] rounded-b-panel">
          {footer}
        </div>
      )}
    </Component>
  );
}
