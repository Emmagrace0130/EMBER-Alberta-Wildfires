import { useState } from "react";
import type { FormEvent } from "react";
import { CheckCircle2, Mail } from "lucide-react";
import Hero from "../components/Hero";
import Card from "../components/Card";
import { submitContactInquiry } from "../lib/api";

const inquiryTypes = [
  "General inquiry",
  "Research collaboration",
  "Community partnership",
  "Demo request",
];

export default function Contact() {
  const [form, setForm] = useState({
    name: "",
    email: "",
    organization: "",
    inquiryType: inquiryTypes[0],
    message: "",
  });
  const [status, setStatus] = useState<"idle" | "submitting" | "success" | "error">("idle");

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setStatus("submitting");
    try {
      await submitContactInquiry(form);
      setStatus("success");
      setForm({ name: "", email: "", organization: "", inquiryType: inquiryTypes[0], message: "" });
    } catch {
      setStatus("error");
    }
  }

  return (
    <div>
      <Hero
        eyebrow="Contact / Collaborate"
        title="Let's talk about wildfire mitigation"
        description="Whether it's a project inquiry, a research collaboration, a community partnership, or a demo request — we'd like to hear from you."
      />

      <section className="section">
        <div className="grid gap-10 lg:grid-cols-[1fr_1.2fr]">
          <Card>
            <Mail className="mb-4 h-6 w-6 text-ember" />
            <h3 className="text-lg font-semibold">Get in touch</h3>
            <p className="mt-2 text-sm text-offwhite/65">
              Use the form to reach the EMBER team about research
              collaboration, community partnerships, or demo requests. We
              read every message.
            </p>
          </Card>

          <Card>
            {status === "success" ? (
              <div className="flex flex-col items-center gap-3 py-10 text-center">
                <CheckCircle2 className="h-10 w-10 text-ember" />
                <h3 className="text-lg font-semibold">Message sent</h3>
                <p className="text-sm text-offwhite/65">
                  Thanks for reaching out — the EMBER team will follow up soon.
                </p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs uppercase tracking-wide text-offwhite/40">
                      Name
                    </label>
                    <input
                      required
                      value={form.name}
                      onChange={(e) => setForm({ ...form, name: e.target.value })}
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm focus:border-ember/50 focus:outline-none"
                    />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs uppercase tracking-wide text-offwhite/40">
                      Email
                    </label>
                    <input
                      required
                      type="email"
                      value={form.email}
                      onChange={(e) => setForm({ ...form, email: e.target.value })}
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm focus:border-ember/50 focus:outline-none"
                    />
                  </div>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs uppercase tracking-wide text-offwhite/40">
                      Organization (optional)
                    </label>
                    <input
                      value={form.organization}
                      onChange={(e) => setForm({ ...form, organization: e.target.value })}
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm focus:border-ember/50 focus:outline-none"
                    />
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <label className="text-xs uppercase tracking-wide text-offwhite/40">
                      Inquiry type
                    </label>
                    <select
                      value={form.inquiryType}
                      onChange={(e) => setForm({ ...form, inquiryType: e.target.value })}
                      className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm focus:border-ember/50 focus:outline-none"
                    >
                      {inquiryTypes.map((t) => (
                        <option key={t} value={t} className="bg-charcoal">
                          {t}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="flex flex-col gap-1.5">
                  <label className="text-xs uppercase tracking-wide text-offwhite/40">
                    Message
                  </label>
                  <textarea
                    required
                    rows={5}
                    value={form.message}
                    onChange={(e) => setForm({ ...form, message: e.target.value })}
                    className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm focus:border-ember/50 focus:outline-none"
                  />
                </div>

                {status === "error" && (
                  <p className="text-sm text-ember">
                    Something went wrong sending your message. Please try again.
                  </p>
                )}

                <button type="submit" disabled={status === "submitting"} className="btn-primary mt-2 self-start">
                  {status === "submitting" ? "Sending..." : "Send message"}
                </button>
              </form>
            )}
          </Card>
        </div>
      </section>
    </div>
  );
}
