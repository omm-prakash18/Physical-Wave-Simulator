/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: {
          base: '#1a1a2e',
          deep: '#15141f',
        },
        panel: {
          glass: 'rgba(250, 245, 235, 0.04)',
          border: 'rgba(212, 168, 83, 0.12)',
        },
        accent: {
          gold: {
            DEFAULT: '#d4a853',
            hover: '#e2ba6e',
            glow: 'rgba(212, 168, 83, 0.15)',
          },
          rose: {
            DEFAULT: '#c97b7b',
            hover: '#d99292',
            glow: 'rgba(201, 123, 123, 0.15)',
          },
          sage: {
            DEFAULT: '#7eb09b',
            hover: '#95c2b0',
          },
          coral: {
            DEFAULT: '#e0685f',
            hover: '#eb7f77',
          },
        },
        text: {
          primary: '#f5f0e8',
          secondary: '#a8a196',
          tertiary: '#6b6558',
        },
      },
      fontFamily: {
        heading: ['Outfit', 'sans-serif'],
        sans: ['Source Sans 3', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      },
      boxShadow: {
        'panel': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'glow-gold': '0 0 16px rgba(212, 168, 83, 0.25)',
        'glow-rose': '0 0 16px rgba(201, 123, 123, 0.25)',
      },
      borderRadius: {
        'panel': '16px',
        'element': '8px',
      },
      lineHeight: {
        'relaxed-body': '1.6',
      },
    },
  },
  plugins: [],
}
