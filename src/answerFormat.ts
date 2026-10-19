const SECTION_HEADINGS = [
  "Direct Answer",
  "Multi-hop Synthesis",
  "Multi-Hop Synthesis",
  "Evidence",
  "Graph Connections",
  "Key Takeaway",
  "Details",
  "Reasoning",
  "Citations",
];

/** Normalize model output into readable Markdown for the chat panel. */
export function formatGeneratedAnswer(raw: string): string {
  if (!raw) return "";

  let text = raw.replace(/\r\n/g, "\n").trim();

  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  text = text.replace(/<\/?think>/gi, "").trim();

  const fenced = text.match(/^```(?:json|markdown|md)?\s*([\s\S]*?)\s*```$/i);
  if (fenced) {
    text = fenced[1].trim();
  }

  if (
    (text.startsWith("{") || text.startsWith("[")) &&
    /"answer"\s*:/.test(text)
  ) {
    try {
      const parsed = JSON.parse(text);
      const nested =
        typeof parsed?.answer === "string"
          ? parsed.answer
          : typeof parsed?.[0]?.answer === "string"
            ? parsed[0].answer
            : "";
      if (nested.trim()) text = nested.trim();
    } catch {
      // Keep original text when the payload is not valid JSON.
    }
  }

  if (!text.includes("\n") && /\\n/.test(text)) {
    text = text.replace(/\\n/g, "\n");
  }
  text = text.replace(/\\t/g, "  ");

  text = text.replace(/\\xrightarrow\{[^}]*\}/g, " → ");
  text = text.replace(/\$\\xrightarrow\{\\text\{([^}]*)\}\}\$/g, " → $1 → ");
  text = text.replace(/\$\$[\s\S]*?\$\$/g, (block) =>
    block.replace(/\$/g, "").replace(/\s+/g, " ").trim()
  );

  // Normalize multi-hop arrow chains: replace double arrows or messy spaces
  text = text.replace(/\s*->\s*/g, " → ");
  text = text.replace(/\s*--+>\s*/g, " → ");
  text = text.replace(/(\s*→\s*){2,}/g, " → ");

  const headingPattern = SECTION_HEADINGS.map((name) =>
    name.replace(/[-\s]/g, "[-\\s]")
  ).join("|");
  text = text.replace(
    new RegExp(`^[ \\t]*(?:#{1,4}[ \\t]*)?(${headingPattern})[ \\t]*:?[ \\t]*`, "gim"),
    "### $1\n\n"
  );

  text = text.replace(/^[ \t]*[-*][ \t]+/gm, "- ");
  text = text.replace(/^[ \t]*(\d+)[.)][ \t]+/gm, "$1. ");

  text = text.replace(/[ \t]+\n/g, "\n");
  text = text.replace(/\n{3,}/g, "\n\n");

  if (!/^#{1,4}\s/m.test(text) && text.length > 180) {
    const paragraphs = text
      .split(/\n{2,}/)
      .map((part) => part.trim())
      .filter(Boolean);
    if (paragraphs.length >= 2) {
      text = `### Direct Answer\n\n${paragraphs[0]}\n\n### Details\n\n${paragraphs.slice(1).join("\n\n")}`;
    } else if (paragraphs.length === 1 && paragraphs[0].length > 400) {
      const sentences = paragraphs[0].split(/(?<=[.!?])\s+/);
      const lead = sentences.slice(0, 2).join(" ");
      const rest = sentences.slice(2).join(" ");
      text = rest
        ? `### Direct Answer\n\n${lead}\n\n### Details\n\n${rest}`
        : `### Direct Answer\n\n${paragraphs[0]}`;
    }
  }

  return text.trim();
}
