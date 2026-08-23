import React, { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Synced frame player with scrubber bar.
 * Displays base64-encoded frame sequences.
 * Timeline track is styled to visually separate context frames (Rose) and predicted frames (Gold).
 * Includes hover-to-preview thumbnails and precise scrubber tick lines.
 */
export default function FramePlayer({
  frames = [],
  label = 'Sequence',
  fps = 6,
  size = 240,
  showScrubber = true,
  className = '',
  onFrameChange = null,
  currentFrame: externalFrame = null,
  contextCount = 10, // First N frames are context, rest are predicted
}) {
  const [internalFrame, setInternalFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const intervalRef = useRef(null);
  
  // Scrubber hover preview states
  const scrubberContainerRef = useRef(null);
  const [hoverFrame, setHoverFrame] = useState(null);
  const [hoverX, setHoverX] = useState(0);

  const currentFrame = externalFrame !== null ? externalFrame : internalFrame;
  const setCurrentFrame = externalFrame !== null ? (val) => {
    if (onFrameChange) onFrameChange(val);
  } : setInternalFrame;

  const play = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (frames.length === 0) return;
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
    if (onFrameChange && externalFrame === null) {
      onFrameChange(internalFrame);
    }
  }, [internalFrame, onFrameChange, externalFrame]);

  useEffect(() => {
    if (frames.length > 0 && externalFrame === null) {
      play();
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [frames.length, play, externalFrame]);

  // Handle mouse hovering over scrubber for image preview
  const handleScrubberMouseMove = (e) => {
    if (!scrubberContainerRef.current || frames.length === 0) return;
    const rect = scrubberContainerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const targetFrame = Math.min(frames.length - 1, Math.floor(percentage * frames.length));
    
    setHoverFrame(targetFrame);
    setHoverX(e.clientX - rect.left);
  };

  const handleScrubberMouseLeave = () => {
    setHoverFrame(null);
  };

  if (frames.length === 0) {
    return (
      <div className={`rounded-panel bg-panel-glass border border-panel-border p-5 flex flex-col items-center justify-center ${className}`}>
        <span className="eyebrow-label mb-3 block">{label}</span>
        <div className="skeleton animate-pulse bg-white/5 border border-panel-border/30 rounded-lg" style={{ width: size, height: size }} />
      </div>
    );
  }

  // Calculate context vs prediction track percentage
  const totalFrames = frames.length;
  const contextRatio = Math.min(1, contextCount / totalFrames);
  const contextPercent = contextRatio * 100;

  // Linear gradient color split: Rose (context) to Gold (prediction)
  const timelineTrackBackground = `linear-gradient(to right, var(--color-accent-rose) 0%, var(--color-accent-rose) ${contextPercent}%, var(--color-accent-gold) ${contextPercent}%, var(--color-accent-gold) 100%)`;

  const isCurrentFrameContext = currentFrame < contextCount;

  return (
    <div className={`rounded-panel bg-panel-glass border border-panel-border p-5 relative overflow-visible ${className}`}>
      {/* Upper info panel */}
      <div className="flex items-center justify-between mb-4 border-b border-panel-border/20 pb-2">
        <div className="flex items-center gap-2">
          <span className="eyebrow-label">{label}</span>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-sans font-medium uppercase tracking-wider ${
            isCurrentFrameContext 
              ? 'bg-accent-rose/10 text-accent-rose border border-accent-rose/20' 
              : 'bg-accent-gold/10 text-accent-gold border border-accent-gold/20'
          }`}>
            {isCurrentFrameContext ? 'Context' : 'Forecast'}
          </span>
        </div>
        <span className="text-xs font-mono font-medium text-accent-gold font-mono-tabular">
          {currentFrame + 1} / {totalFrames}
        </span>
      </div>

      {/* Frame image display panel */}
      <div className="flex justify-center mb-4">
        <div className="relative group rounded-xl overflow-hidden border border-panel-border shadow-inner bg-[#15141f]">
          <img
            src={`data:image/png;base64,${frames[currentFrame]}`}
            alt={`${label} Frame ${currentFrame + 1}`}
            className="object-contain block transition-smooth"
            style={{ width: size, height: size, imageRendering: 'pixelated' }}
          />
        </div>
      </div>

      {/* Interactive Controls & Scrubbers */}
      {showScrubber && (
        <div className="space-y-4">
          <div className="relative select-none" ref={scrubberContainerRef}>
            {/* Scrubber slider track */}
            <input
              type="range"
              min={0}
              max={totalFrames - 1}
              value={currentFrame}
              onChange={(e) => {
                const val = parseInt(e.target.value);
                setCurrentFrame(val);
                pause();
              }}
              onMouseMove={handleScrubberMouseMove}
              onMouseLeave={handleScrubberMouseLeave}
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer relative z-10 custom-slider"
              style={{ background: timelineTrackBackground }}
            />

            {/* Hover-to-preview thumbnail popup */}
            {hoverFrame !== null && frames[hoverFrame] && (
              <div
                className="absolute z-30 pointer-events-none transform -translate-x-1/2 bottom-8 bg-[#15141f] border border-accent-gold/30 p-1.5 rounded-lg shadow-xl backdrop-blur-md flex flex-col items-center animate-fade-in"
                style={{ left: `${hoverX}px` }}
              >
                <img
                  src={`data:image/png;base64,${frames[hoverFrame]}`}
                  alt={`Preview ${hoverFrame + 1}`}
                  className="w-16 h-16 rounded border border-panel-border bg-[#15141f] object-cover mb-1"
                  style={{ imageRendering: 'pixelated' }}
                />
                <span className="text-[9px] font-mono text-accent-gold">
                  t = {hoverFrame + 1} ({hoverFrame < contextCount ? 'Ctx' : 'Pred'})
                </span>
              </div>
            )}

            {/* Scrubber ticks */}
            <div className="flex justify-between px-1.5 pt-1.5 select-none" aria-hidden="true">
              {frames.map((_, i) => (
                <div key={i} className="flex flex-col items-center">
                  <span className={`h-1 w-[1px] ${i < contextCount ? 'bg-accent-rose/40' : 'bg-accent-gold/40'}`} />
                  <span className={`text-[8px] font-mono mt-1 ${
                    i === currentFrame
                      ? 'text-accent-gold font-bold scale-110'
                      : i < contextCount
                      ? 'text-text-secondary/50'
                      : 'text-text-secondary/40'
                  }`}>
                    {i + 1}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Action buttons */}
          {externalFrame === null && (
            <div className="flex justify-center">
              <button
                onClick={() => (isPlaying ? pause() : play())}
                className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-accent-gold text-canvas-deep hover:bg-accent-gold-hover active:scale-95 transition-snappy shadow-md shadow-accent-gold/10 cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-gold/40"
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
