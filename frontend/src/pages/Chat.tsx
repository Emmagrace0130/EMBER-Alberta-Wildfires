import { useEffect, useRef, useState } from "react";
import { BookOpen, ChevronDown, ChevronUp, Flame, Send } from "lucide-react";

interface Source {
  title: string;
  raw: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
  error?: boolean;
}

const STARTERS = [
  "What are the key steps for creating a defensible space around a home?",
  "What causes most wildfires in Alberta?",
  "How should a community plan for wildfire evacuation?",
  "What fuel management strategies are most effective near the WUI?",
];

function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  return (
    <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.03]">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs text-offwhite/50 hover:text-offwhite/80"
      >
        <BookOpen className="h-3.5 w-3.5" />
        <span>{sources.length} source{sources.length > 1 ? "s" : ""} retrieved</span>
        {open ? <ChevronUp className="ml-auto h-3.5 w-3.5" /> : <ChevronDown className="ml-auto h-3.5 w-3.5" />}
      </button>
      {open && (
        <ul className="border-t border-white/10 px-3 py-2 space-y-1">
          {sources.map((s) => (
            <li key={s.raw} className="text-xs text-offwhite/60">
              · {s.title}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[85%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        {!isUser && (
          <div className="mb-1 flex items-center gap-1.5">
            <Flame className="h-3.5 w-3.5 text-ember" />
            <span className="text-xs font-semibold text-ember">EMBER</span>
          </div>
        )}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-ember text-charcoal-deep font-medium"
              : msg.error
              ? "bg-red-900/30 text-offwhite/70 border border-red-500/20"
              : "glass-panel text-offwhite/90"
          }`}
        >
          {msg.content}
          {msg.streaming && (
            <span className="ml-1 inline-block h-3.5 w-0.5 bg-ember animate-pulse" />
          )}
        </div>
        {msg.sources && <SourcesPanel sources={msg.sources} />}
      </div>
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send(text: string) {
    const msg = text.trim();
    if (!msg || loading) return;
    setInput("");

    const userMsg: Message = { role: "user", content: msg };
    const history = messages
      .filter((m) => !m.streaming)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      userMsg,
      { role: "assistant", content: "", streaming: true },
    ]);
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, history }),
      });

      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);
          if (payload === "[DONE]") break;

          let event: { type: string; content?: string; sources?: Source[] };
          try { event = JSON.parse(payload); } catch { continue; }

          if (event.type === "text" && event.content) {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last.role === "assistant") {
                next[next.length - 1] = { ...last, content: last.content + event.content };
              }
              return next;
            });
          } else if (event.type === "sources") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last.role === "assistant") {
                next[next.length - 1] = { ...last, sources: event.sources, streaming: false };
              }
              return next;
            });
          } else if (event.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  content: event.content ?? "An error occurred.",
                  streaming: false,
                  error: true,
                };
              }
              return next;
            });
          }
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last.role === "assistant") {
          next[next.length - 1] = {
            ...last,
            content: "Connection error. Please try again.",
            streaming: false,
            error: true,
          };
        }
        return next;
      });
    } finally {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last.role === "assistant" && last.streaming) {
          next[next.length - 1] = { ...last, streaming: false };
        }
        return next;
      });
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  }

  return (
    <div className="flex flex-col" style={{ height: "calc(100vh - 65px)" }}>
      {/* Header */}
      <div className="border-b border-white/10 bg-charcoal-light px-6 py-4 sm:px-8 flex-none">
        <div className="mx-auto max-w-3xl">
          <p className="text-xs font-semibold uppercase tracking-widest text-ember mb-0.5">
            AI Assistant
          </p>
          <h1 className="text-xl font-semibold">Ask EMBER</h1>
          <p className="mt-0.5 text-xs text-offwhite/50">
            RAG-powered guidance from 510 indexed chunks across 6 wildfire mitigation documents
          </p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 sm:px-8">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center gap-6 pt-8">
              <div className="rounded-full border border-ember/30 bg-ember/10 p-4">
                <Flame className="h-8 w-8 text-ember" />
              </div>
              <div className="text-center">
                <h2 className="text-lg font-semibold">Wildfire mitigation guidance</h2>
                <p className="mt-1 text-sm text-offwhite/50 max-w-sm">
                  Ask EMBER anything about wildfire risk, defensible space, community preparedness, or mitigation strategies.
                </p>
              </div>
              <div className="grid gap-2 w-full sm:grid-cols-2">
                {STARTERS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="glass-card text-left text-sm text-offwhite/70 hover:text-offwhite hover:border-ember/40 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="flex-none border-t border-white/10 bg-charcoal-light px-6 py-4 sm:px-8">
        <div className="mx-auto max-w-3xl flex gap-3 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask about wildfire risk, defensible space, mitigation strategies…"
            rows={1}
            disabled={loading}
            className="flex-1 resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-offwhite placeholder:text-offwhite/30 focus:border-ember/50 focus:outline-none disabled:opacity-50"
            style={{ maxHeight: 120, overflowY: "auto" }}
            onInput={(e) => {
              const t = e.currentTarget;
              t.style.height = "auto";
              t.style.height = Math.min(t.scrollHeight, 120) + "px";
            }}
          />
          <button
            onClick={() => send(input)}
            disabled={!input.trim() || loading}
            className="rounded-2xl bg-ember p-3 text-charcoal-deep transition-transform hover:scale-105 hover:bg-ember-light disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-xs text-offwhite/30 text-center">
          Press Enter to send · Shift+Enter for newline · powered by llama3.1:8b via Ollama
        </p>
      </div>
    </div>
  );
}
