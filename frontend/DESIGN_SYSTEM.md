# Latent Video Prediction Studio — Design System Tokens & Patterns

This design system establishes visual consistency, typographic rhythm, and micro-interaction polish for the scientific ML wave simulation dashboard. The goals are high visual clarity, ergonomic scanning for temporal predictions, and full accessibility.

---

## 1. Color System

The palette features a warm, deep "control room at night" theme, optimizing contrast and visual comfort for multi-hour research analysis.

### Canvas & Base Layers
- **Canvas Base (`canvas-base`)**: `#1a1a2e` (App background gradient)
- **Canvas Deep (`canvas-deep`)**: `#15141f` (Gradient endpoint / vignette overlay)
- **Panel Surface (`panel-glass`)**: Translucent cream-glass `rgba(250, 245, 235, 0.04)` with backdrop-blur (`24px`).
- **Panel Border (`panel-border`)**: Subtle warm tone `rgba(212, 168, 83, 0.12)`.

### Semantic Accents
- **Primary Accent (`accent-gold`)**: `#d4a853`
  - *Usage*: Primary actions, active states, key metric highlights, and "Prediction" series representations.
- **Secondary Accent (`accent-rose`)**: `#c97b7b`
  - *Usage*: Context/Ground-truth series data, secondary focus elements.
- **Success (`accent-sage`)**: `#7eb09b`
  - *Usage*: PSNR/SSIM improvements, training loops convergence indicators.
- **Danger (`accent-coral`)**: `#e0685f`
  - *Usage*: Warnings, offline status, validation failures (use sparingly).

### Text Scale
- **Primary Text (`text-primary`)**: `#f5f0e8` (Cream white for high-contrast readability).
- **Secondary Text (`text-secondary`)**: `#a8a196` (Soft warm taupe for body prose).
- **Tertiary/Disabled (`text-tertiary`)**: `#6b6558` (Muted labels).

### Data Visualization Palettes
1. **Temporal Gradient (UMAP)**: Gold to Rose (`#d4a853` → `#c97b7b`). Used sequentially to track the temporal trajectory of latent space tokens.
2. **Diverging Scale (Attention)**: Dark Blue (`#233a55`) → Neutral Gray (`#6b6558`) → Gold (`#d4a853`). Colorblind-safe diverging scale to show relative magnitude of attention values without red-green conflicts.

---

## 2. Typography

We enforce a strict font hierarchy to ensure scientific density without visual noise.

| Token | Class / Element | Font Family | Size | Weight | Tracking / Leading |
|---|---|---|---|---|---|
| **Heading 1** | `h1` | `Outfit` | `2.0rem` (32px) | Bold (`700`) | `-0.02em` / `1.25` |
| **Heading 2** | `h2` | `Outfit` | `1.375rem` (22px) | Semi-Bold (`600`) | `-0.01em` / `1.35` |
| **Heading 3** | `h3` | `Outfit` | `1.0rem` (16px) | Semi-Bold (`600`) | `0.01em` / `1.4` |
| **Eyebrow** | `.eyebrow-label` | `Fira Code` | `0.6875rem` (11px) | Medium (`500`) | `0.15em` / Uppercase |
| **Body text** | `body`, `p` | `Source Sans 3` | `0.875rem` (14px) | Regular (`400`) | `0.01em` / `1.6` |
| **Mono Stats**| `.font-mono-tabular` | `Fira Code` | Dynamic | Medium/Bold | `tabular-nums` |

### Prose Rule
To maximize readability, all prose-heavy elements (e.g. Model Card descriptions) must be constrained with the `.prose-panel` utility capping line lengths at **`70ch`**.

---

## 3. Spacing, Radius & Elevation

### Spacing Scale
Consistent gutters and paddings mapping:
- Gutters: `gap-4` (16px) / `gap-6` (24px) for layouts.
- Card Padding: `p-5` (20px) as default container padding.
- Input Padding: `px-3 py-2` (12px / 8px).

### Radius Scale
- Card Surfaces: `rounded-panel` (16px / `1rem`).
- Buttons & Input fields: `rounded-lg` (8px / `0.5rem`).

### Elevation Layers
1. **Canvas base**: Root canvas with dark gradient mesh background.
2. **Panel surfaces**: Standard `.rounded-panel bg-panel-glass` container cards.
3. **Floating overlays**: Tooltips, scrubber previews, dropdown selectors. Styled with extra opacity `bg-[#15141f]/95`, custom borders, and heavy `shadow-2xl`.

---

## 4. Reusable Component Patterns

### 1. Card Container (`Card.jsx`)
Standard wrapper enforcing the cream-glass texture and backdrop filters. Supports an accent border-top indicating category:
- `accent="none"`: Neutral card.
- `accent="gold"`: Prediction block.
- `accent="rose"`: Reference/Context block.
- `accent="sage"`: Metric/Success block.

### 2. Metric Statistics (`MetricStat.jsx`)
Presents key scalar numbers. Always aligns values to the right (or centers) using `font-mono-tabular` in `Fira Code` to avoid shifting Layouts during live training updates.
Includes a supporting muted description block.

### 3. Smooth Slider (`Slider.jsx`)
Custom-styled parameter slider. The track is dynamically filled with a gold gradient corresponding to value percentage. Utilizes an oversized touch target thumb with a soft glow effect on hover/drag.

### 4. Collapsible Sidebar Navigation (`NavRail.jsx`)
Ensures constant spatial navigation orientation. Collapsible to icon-only mode to free up visual space for high-density plots.
*Accessibility*: Fully keyboard-navigable via Up/Down arrow keys. Focus outlines are styled with gold accent rings (`focus-visible:ring-accent-gold/45`).

### 5. Timestep Scrubber Frame Player (`FramePlayer.jsx`)
Synchronized image slider providing video scrubbing.
- Scrubber track is split: Context frames are colored in Rose, Predicted frames are colored in Gold.
- Displays tick lines matching frame numbers.
- Displays floating image thumbnails on timeline hover.
