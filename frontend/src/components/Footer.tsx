import { Link } from "react-router-dom";
import { Flame } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-white/10 bg-charcoal-deep">
      <div className="mx-auto max-w-7xl px-6 py-12 sm:px-8">
        <div className="flex flex-col gap-8 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <Flame className="h-5 w-5 text-ember" />
              <span className="font-display text-base font-semibold">EMBER</span>
            </div>
            <p className="mt-3 max-w-sm text-sm text-offwhite/60">
              A wildfire mitigation and decision-support platform helping
              communities understand risk and act before disaster happens.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-8 text-sm sm:grid-cols-3">
            <div className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-wide text-offwhite/40">
                Platform
              </span>
              <Link to="/mitigation-plans" className="text-offwhite/70 hover:text-ember">
                Mitigation Plans
              </Link>
              <Link to="/risk-insights" className="text-offwhite/70 hover:text-ember">
                Risk Insights
              </Link>
              <Link to="/resources" className="text-offwhite/70 hover:text-ember">
                Resources
              </Link>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-wide text-offwhite/40">
                About
              </span>
              <Link to="/about" className="text-offwhite/70 hover:text-ember">
                About EMBER
              </Link>
              <Link to="/how-it-works" className="text-offwhite/70 hover:text-ember">
                How It Works
              </Link>
              <Link to="/what-ember-can-do" className="text-offwhite/70 hover:text-ember">
                Capabilities
              </Link>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-wide text-offwhite/40">
                Connect
              </span>
              <Link to="/contact" className="text-offwhite/70 hover:text-ember">
                Contact Us
              </Link>
            </div>
          </div>
        </div>

        <div className="mt-10 border-t border-white/5 pt-6 text-xs text-offwhite/40">
          © {new Date().getFullYear()} EMBER. Built for safer, better-prepared communities.
        </div>
      </div>
    </footer>
  );
}
