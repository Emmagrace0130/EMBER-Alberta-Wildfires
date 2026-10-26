/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Space Grotesk", "Inter", "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        charcoal: {
          DEFAULT: "#0b0c0e",
          light: "#15171b",
          deep: "#070809",
        },
        surface: "#15171bcc",
        ember: {
          DEFAULT: "#ff6a3d",
          light: "#ff9466",
          dark: "#c9491f",
        },
        amber: {
          DEFAULT: "#ffb347",
        },
        copper: {
          DEFAULT: "#c8783c",
        },
        gold: {
          DEFAULT: "#e8b923",
        },
        slate: {
          DEFAULT: "#64748b",
          deep: "#1c2333",
        },
        offwhite: "#f5f1ea",
      },
      backgroundImage: {
        "ember-glow":
          "radial-gradient(circle at 30% 20%, rgba(255,106,61,0.18), transparent 55%), radial-gradient(circle at 80% 0%, rgba(232,185,35,0.12), transparent 50%)",
      },
      boxShadow: {
        glow: "0 0 40px rgba(255,106,61,0.25)",
      },
    },
  },
  plugins: [],
};
