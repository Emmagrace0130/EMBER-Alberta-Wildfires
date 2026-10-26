import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Flame, Menu, X } from "lucide-react";

const links = [
  { to: "/", label: "Home" },
  { to: "/about", label: "About" },
  { to: "/mitigation-plans", label: "Mitigation Plans" },
  { to: "/risk-insights", label: "Risk Insights" },
  { to: "/fire-map", label: "Fire Map" },
  { to: "/assistant", label: "Ask EMBER" },
  { to: "/what-ember-can-do", label: "What EMBER Can Do" },
  { to: "/how-it-works", label: "How It Works" },
  { to: "/resources", label: "Resources" },
  { to: "/contact", label: "Contact" },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-charcoal/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4 sm:px-8">
        <NavLink to="/" className="flex items-center gap-2" onClick={() => setOpen(false)}>
          <Flame className="h-6 w-6 text-ember" strokeWidth={2.2} />
          <span className="font-display text-lg font-semibold tracking-wide">EMBER</span>
        </NavLink>

        <nav className="hidden items-center gap-1 lg:flex">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) =>
                `rounded-full px-3 py-2 text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-white/10 text-ember"
                    : "text-offwhite/70 hover:text-offwhite"
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <button
          className="rounded-full border border-white/10 p-2 lg:hidden"
          onClick={() => setOpen((v) => !v)}
          aria-label="Toggle navigation"
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {open && (
        <nav className="border-t border-white/10 px-6 pb-4 lg:hidden">
          <div className="flex flex-col gap-1 pt-2">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm font-medium ${
                    isActive ? "bg-white/10 text-ember" : "text-offwhite/70"
                  }`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </div>
        </nav>
      )}
    </header>
  );
}
