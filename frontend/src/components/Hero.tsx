import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import type { ReactNode } from "react";

interface HeroProps {
  eyebrow?: string;
  title: string;
  description: string;
  primaryCta?: { label: string; to: string };
  secondaryCta?: { label: string; to: string };
  children?: ReactNode;
}

export default function Hero({
  eyebrow,
  title,
  description,
  primaryCta,
  secondaryCta,
  children,
}: HeroProps) {
  return (
    <section className="relative overflow-hidden bg-ember-glow">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -top-32 left-1/4 h-96 w-96 rounded-full bg-ember/20 blur-3xl" />
        <div className="absolute -bottom-24 right-0 h-80 w-80 rounded-full bg-gold/10 blur-3xl" />
        <svg className="absolute inset-0 h-full w-full opacity-[0.04]" aria-hidden="true">
          <defs>
            <pattern id="grid" width="48" height="48" patternUnits="userSpaceOnUse">
              <path d="M 48 0 L 0 0 0 48" fill="none" stroke="white" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <div className="section relative z-10 py-28 sm:py-32">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-3xl"
        >
          {eyebrow && <span className="pill mb-6 inline-block text-ember">{eyebrow}</span>}
          <h1 className="text-4xl font-semibold leading-tight sm:text-5xl lg:text-6xl">
            {title}
          </h1>
          <p className="mt-6 max-w-xl text-lg text-offwhite/70">{description}</p>

          {(primaryCta || secondaryCta) && (
            <div className="mt-10 flex flex-wrap gap-4">
              {primaryCta && (
                <Link to={primaryCta.to} className="btn-primary">
                  {primaryCta.label}
                </Link>
              )}
              {secondaryCta && (
                <Link to={secondaryCta.to} className="btn-secondary">
                  {secondaryCta.label}
                </Link>
              )}
            </div>
          )}
        </motion.div>

        {children}
      </div>
    </section>
  );
}
