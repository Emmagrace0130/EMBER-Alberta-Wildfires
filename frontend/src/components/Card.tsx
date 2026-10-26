import { motion } from "framer-motion";
import type { ReactNode } from "react";

export default function Card({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.45, delay }}
      className={`glass-card ${className}`}
    >
      {children}
    </motion.div>
  );
}
