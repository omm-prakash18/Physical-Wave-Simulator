import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Synced frame player with scrubber bar.
 * Displays base64-encoded frame sequences side-by-side.
 */
export default function FramePlayer({
  frames = [],
  label = 'Sequence',
  fps = 6,
  size = 192,
  showScrubber = true,
  className = '',
  onFrameChange = null,
  currentFrame: externalFrame = null,
}) {
  const [internalFrame, setInternalFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const intervalRef = useRef(null);

  const currentFrame = externalFrame !== null ? externalFrame : internalFrame;
  const setCurrentFrame = externalFrame !== null ? () => {} : setInternalFrame;

  const play = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setInternalFrame((prev) => (prev + 1) % frames.length);
    }, 1000 / fps);
    setIsPlaying(true);
  }, [frames.length, fps]);

  const pause = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    setIsPlaying(false);
  };

  useEffect(() => {
    if (onFrameChange) {
      onFrameChange(internalFrame);
    }
  }, [internalFrame, onFrameChange]);

  useEffect(() => {
    if (frames.length > 0 && externalFrame === null) {
      play();
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [frames.length, play, externalFrame]);

  if (frames.length === 0) {
    return (
      <div className={`glass-card p-4 ${className}`}>
        <div className="skeleton" style={{ width: size, height: size }} />
      </div>
    );
  }

  return (
    <div className={`glass-card p-4 ${className}`}>
      {/* Label */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono text-text-secondary uppercase tracking-wider">{label}</span>
        <span className="text-xs font-mono text-gold">
          {currentFrame + 1}/{frames.length}
        </span>
      </div>

      {/* Frame display */}
      <div className="flex justify-center mb-3">
        <img
          src={`data:image/png;base64,${frames[currentFrame]}`}
          alt={`Frame ${currentFrame + 1}`}
          className="frame-display rounded-lg border border-gold/10"
          style={{ width: size, height: size }}
        />
      </div>

      {/* Controls */}
      {showScrubber && externalFrame === null && (
        <div className="space-y-2">
          {/* Scrubber */}
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={currentFrame}
            onChange={(e) => {
              const val = parseInt(e.target.value);
              setCurrentFrame(val);
              if (onFrameChange) onFrameChange(val);
              pause();
            }}
            className="w-full h-1 rounded-full appearance-none bg-gold/10 cursor-pointer"
          />

          {/* Play/Pause */}
          <div className="flex justify-center">
            <button
              onClick={() => (isPlaying ? pause() : play())}
              className="px-3 py-1 rounded-md text-xs font-medium bg-gold/15 text-gold hover:bg-gold/25 transition-all"
            >
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
