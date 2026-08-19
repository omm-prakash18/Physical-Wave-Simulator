export default function SkeletonLoader({ className = '', rows = 3 }) {
  return (
    <div className={`space-y-3 ${className}`}>
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="skeleton"
          style={{
            height: i === 0 ? '24px' : '16px',
            width: i === 0 ? '60%' : `${70 + Math.random() * 30}%`,
          }}
        />
      ))}
    </div>
  );
}

export function SkeletonCard({ className = '' }) {
  return (
    <div className={`glass-card p-6 ${className}`}>
      <div className="skeleton h-5 w-1/3 mb-4" />
      <div className="skeleton h-40 w-full mb-3" />
      <div className="skeleton h-3 w-2/3" />
    </div>
  );
}

export function SkeletonChart({ className = '' }) {
  return (
    <div className={`glass-card p-6 ${className}`}>
      <div className="skeleton h-5 w-1/4 mb-4" />
      <div className="skeleton h-64 w-full" />
    </div>
  );
}
