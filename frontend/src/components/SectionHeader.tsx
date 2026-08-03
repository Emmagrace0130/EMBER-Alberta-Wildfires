import { motion } from "framer-motion";

interface SectionHeaderProps {
  eyebrow?: string;
  title: string;
  description?: string;
  align?: "left" | "center";
}

export default function SectionHeader({
  eyebrow,
  title,
  description,
  align = "left",
}: SectionHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5 }}
      className={`mb-12 max-w-2xl ${align === "center" ? "mx-auto text-center" : ""}`}
    >
      {eyebrow && (
        <span className="pill mb-4 inline-block text-ember">{eyebrow}</span>
      )}
      <h2 className="text-3xl font-semibold sm:text-4xl">{title}</h2>
      {description && (
        <p className="mt-4 text-base text-offwhite/65">{description}</p>
      )}
    </motion.div>
  );
}
