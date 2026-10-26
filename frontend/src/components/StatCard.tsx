import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface StatCardProps {
  value: string;
  label: string;
  icon?: ReactNode;
  delay?: number;
}

export default function StatCard({ value, label, icon, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay }}
      className="glass-panel flex flex-col gap-2 p-6"
    >
      {icon && <div className="text-ember">{icon}</div>}
      <span className="font-display text-3xl font-semibold text-offwhite">
        {value}
      </span>
      <span className="text-sm text-offwhite/60">{label}</span>
    </motion.div>
  );
}
