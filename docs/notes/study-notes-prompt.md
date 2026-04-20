# Prompt — Study Notes from a Model Repository

> A prompt template for generating deep, teaching-oriented study notes on any open-source language-model repository. Designed to be copy-pasted into Claude Code or similar coding agents, run from inside the model's repo directory.
>
> Polished through real use — every instruction below exists because it was missing the first time and cost us an iteration. Don't trim it, it's load-bearing.

Copy everything below the line. Run it from inside the model's repo.

---

You are my personal teacher. I am motivated but inexperienced — assume I understand basic Python and what a neural network is, but not transformer internals, modern LLM tricks (RoPE, RMSNorm, GQA, SwiGLU, MoE, MLA, etc.), or advanced PyTorch patterns. Explain as if I'm smart but new. **Never skip a step because "it's obvious."**

## 1. Read before you write

Do not write a single sentence of notes until you have finished these:

1. List the top-level directory contents.
2. Identify the core modeling file(s) (`model.py`, `modeling_*.py`, `transformer.py`, `llama.py`, etc.) and the config (`config.json`, `configuration_*.py`, or a `ModelArgs` dataclass).
3. Read every function and class in the core modeling file top-to-bottom. No skimming. Follow imports if they define architectural pieces.
4. Read the generation/inference code if present — it's where KV cache, sampling, and causal masking live.
5. Read the tokenizer source to understand vocab size, special tokens, chat format.
6. Read the README and model card so your framing matches the authors'.

Before writing, state back to me (in plain chat, not in the notes file):
- What model/repo you identified
- What files you plan to read and in what order
- Anything ambiguous that I should clarify first

Only after that, write the notes.

## 2. Where to put the output

- **Notes file:** `NOTES.md` in the current directory (or `docs/notes/<model>.md` if I've got an MkDocs-style layout).
- **Diagram/plot files:** an adjacent subfolder called `notes_assets/` (or `docs/assets/notes/` in an MkDocs layout). Put one file per figure there. Reference them from the notes using `![alt](relative/path.svg)` — never inline `<svg>…</svg>`, which many Markdown viewers silently refuse to render.

## 3. Structure of the notes file

Follow this structure exactly. Each numbered heading is a section you must produce.

### § 1. TL;DR (one paragraph)

Plain-English summary: what this model is, what family it belongs to, what's notable. No jargon without immediate definition. End with the one-line "what's the actual innovation" answer.

### § 2. The 30-second mental model

A Mermaid flowchart of the end-to-end forward pass: tokens → embeddings → N× transformer blocks → norm → LM head → logits. Then a second Mermaid flowchart of a single block (pre-norm residual structure). Label every arrow with the tensor shape at that point.

### § 3. Key hyperparameters — with intuition, not just numbers

Pull the real values from the config. Format as a 4-column table:

| Param | Value | What it means | Why this value |

**Critical**: for every param, explain the *concept*, not just the name. For `vocab_size = 128,256`, don't stop at "size of the vocabulary" — explain that tokens are BPE sub-word pieces (not words), that the vocab is finite but can represent anything, and include concrete examples of how words tokenize. For `rope_theta`, explain what a bigger/smaller base frequency does to long-context behavior.

A good test: a reader who knows nothing should be able to predict what would happen if you doubled or halved any number in this table.

### § 4. Architecture walk-through

Walk through the model component-by-component, **in data-flow order**. For EACH component, use this sub-structure:

- **What it is** — one sentence.
- **Why it exists** — what problem it solves.
- **How it works** — intuition first, math second, code third.
- **Math block** — formal notation in a fenced code block using Unicode (see rendering rules below). Never use LaTeX `$…$` delimiters — they don't render in most Markdown viewers without extensions.
- **Code snippet** — from the actual repo, trimmed to the essential ~15–40 lines. Inline `# comment` on every non-trivial line explaining *why*, not just *what*.
- **Shape annotations religiously** — `# x: (B, T, 4096)` at every tensor step. Tensor shapes are where understanding lives or dies.
- **Worked numerical example** when the math is non-obvious (e.g. RoPE: pick `head_dim=4`, compute the rotated vector at two positions by hand). If there's no worked example, the reader will stay confused.
- **"Why this, not the alternative?"** callout whenever the component is a "modernization" over a predecessor (e.g. RoPE vs. 2017-sinusoidal, RMSNorm vs. LayerNorm, GQA vs. MHA, SwiGLU vs. ReLU). List 2–4 concrete advantages over the alternative, not hand-waving.
- **"Why does this actually work?"** callout for non-obvious tricks (e.g. why GQA doesn't lose much quality; why KV caching is correct; why the causal mask is shaped the way it is). Don't leave magic unexplained.

Components to cover (adapt to what's actually in the repo — drop anything not present):

- Tokenizer & embeddings (include what "vocab size" *means* — tokens ≠ words)
- Positional encoding (RoPE? ALiBi? Sinusoidal? Learned absolute?)
- Normalization (LayerNorm vs RMSNorm; pre-norm vs post-norm)
- Attention (MHA vs GQA vs MQA vs MLA); **also** explain the KV cache layout, what's cached vs. what isn't, and how a single decode step reads/writes it
- MLP / FFN (SwiGLU? GeGLU? ReLU? — explain the gating mechanism if gated)
- MoE routing if present (top-k, expert selection, load balancing)
- Residual + block structure, LM head, weight tying (or lack thereof)
- Any unusual tricks (sliding-window attention, multi-token prediction, speculative decoding hooks, etc.)

### § 5. A concrete forward pass — NOT an abstract narration

Pick a short real sentence (e.g. `"The cat sat on"`) and trace it through the model. Not "a token enters the model, step 1 is the embedding lookup"; instead: "the five tokens are `[BOS, 'The', ' cat', ' sat', ' on']` with shape `(1, 5)`. After embedding: shape `(1, 5, 4096)`. After RoPE in layer 1's attention: the vector for ' cat' now carries position information for index 2. After attention..." — concrete, shape-annotated, semantically meaningful.

Include a diagram (SVG file) showing how the semantic content of each token's vector evolves across selected layers (embedding → early → middle → late). End with the last token's vector being fed into the LM head, producing a plausible top-k of predicted next tokens. Then show how the KV cache is used for the *next* token's decode step.

This section is the single most important one. Make it impossible to misunderstand.

### § 6. Training vs inference differences

What changes at inference time? KV cache allocation and use, causal masking with cached prefix, `@torch.inference_mode()` decorators, sampling (top-p, temperature, stop tokens). Show the key code paths for both if present.

### § 7. Zoom-out: how this compares

Short comparison section in tables:

- **vs. the original Transformer (2017)** — what's the same, what changed, why.
- **vs. one older sibling in the same family** (e.g. if Llama 3, compare to Llama 2; if Qwen3, compare to Qwen2). Call out the specific hyperparameter and architectural deltas, not vibes.
- **vs. one contemporary competitor** (pick a relevant one: Mistral, DeepSeek, Qwen, Gemma) — 2–3 architectural differences with brief rationales.

### § 8. The spotlight: what's actually novel

One paragraph. If I had to explain to a friend "what's the one thing that makes this model special," what's the answer? Be honest — if it's mostly a well-scaled standard transformer with no new architectural idea, say so explicitly. "The innovation is that there isn't one, and that's the interesting part" is a legitimate answer.

### § 9. Glossary

Every acronym and piece of jargon you used, defined in one line. Alphabetical.

### § 10. Where to go next

- 2–3 specific files/functions in this repo worth re-reading after this overview (with line ranges).
- 1–2 papers that would deepen understanding.
- 1 suggested hands-on exercise (e.g. "change `rope_theta` from 500,000 to 10,000, feed a 5K-token prompt, predict what breaks and why").

## 4. Rendering rules — what to use for what

The notes must render correctly in every Markdown viewer (VS Code preview, GitHub, MkDocs, Obsidian, Typora) with **no extensions required**. That means:

### Math

Write math in **Unicode symbols inside fenced code blocks**. Never use LaTeX delimiters (`$…$`, `$$…$$`) — they silently fail in most viewers.

Good:
````
           x
RMSNorm(x) = ──────────────────  ⊙ γ
           √((1/d) Σᵢ xᵢ² + ε)
````

Bad: `$\text{RMSNorm}(x) = x / \sqrt{(1/d) \sum x_i^2 + \epsilon} \cdot \gamma$`

Useful Unicode: θ σ ε μ γ λ Σ Π √ ⊙ ⊕ ∈ ℝ ℂ ≈ ≪ ≫ → ← ↔ ⟨ ⟩ · × · subscripts ₀₁₂...ₙ superscripts ⁰¹²...ⁿˣᵀ.

### Diagrams — pick the right tool per diagram

- **Mermaid** inside ` ```mermaid ` code fences — use for high-level architecture flows (e.g. the 30-second mental model). Works on GitHub and MkDocs Material out of the box.
- **Hand-authored SVG files** in `notes_assets/` — use for anything geometric, spatial, or non-flowchart. Reference via `![alt](notes_assets/name.svg)`. Examples of diagrams that warrant a custom SVG:
    - RoPE rotation in a 2D plane (axes, unit circle, rotated vectors, angle arcs)
    - GQA head-sharing structure (32 Q boxes grouped into 8 colored groups, each sharing one KV box)
    - KV cache evolution across decode steps (row of slots with write/read arrows)
    - A concrete sentence tracing through layers (columns = tokens, rows = layers, colored boxes with semantic labels)
- **matplotlib plots** saved as SVG — use for functions or empirical curves. Example: SiLU / ReLU / y = x on one axis to show the activation's shape, minimum, and asymptotic behavior. Save the generating `.py` script next to the SVG so the plot is reproducible.
- **ASCII art** — only for tiny sketches where the time-to-draw beats every other option. Don't use ASCII for anything a reader will stare at for more than five seconds.

For SVG files, make sure they:
- Start with `<?xml version="1.0" encoding="UTF-8"?>` and a `<svg>` with explicit `viewBox`.
- Have a white `<rect>` background (renderers default to transparent — ugly in dark mode).
- Space labels so they never overlap with arrows, axes, or each other. If the diagram looks cramped, enlarge the viewBox. Cramped diagrams fail the whole purpose.
- Include a short title and, where useful, a footer caption *inside* the SVG.

### Code blocks

Use standard fenced blocks with a language tag (` ```python `). Keep snippets to the essential lines — every extra line is noise.

### Callouts

Use Markdown blockquote with an emoji to visually flag things:

- `> ⚠️ **Confusion point:** …` — honest uncertainty you ran into
- `> 💡 **Why this works:** …` — non-obvious correctness argument
- `> 🔍 **Compare:** …` — a small "why this not that" note inline

## 5. Style rules — follow strictly

- **Teach, don't summarize.** If the reader could get the same from the README or model card, you've failed. Add intuition they don't.
- **Zoom out, then zoom in, then zoom out.** Every major section opens with big-picture "why" before any code.
- **Every code snippet must have comments on non-obvious lines** — and the comments must explain the *why*, not restate the *what*. No `# attention here` type comments.
- **Shape annotations religiously.** `(B, T, C)` at every tensor step. Never present a tensor operation without the shape before and after.
- **Prefer analogies for hard concepts.** RoPE rotation → clock hand. MoE routing → hospital triage nurse. KV cache → a scratchpad you never erase, only append to. GQA → detectives sharing evidence lockers. Concrete mental images stick; abstract descriptions don't.
- **No hedging fluff.** No "it's worth noting that…", "as you may know…", "it is interesting to observe…". Direct sentences, owned opinions.
- **When confused, say so.** Use the `> ⚠️ Confusion point:` callout rather than glossing over something you didn't understand. Honest uncertainty beats fake confidence.
- **Length is whatever the material requires.** Don't pad. Don't truncate. A real study guide for a modern LLM is typically 3,000–6,000 words.

## 6. Output checklist

Before you say "done," verify:

- [ ] All 10 sections are present and in order.
- [ ] Every hyperparameter in the table has a *conceptual* explanation, not just a technical definition.
- [ ] Every "modernization" component has a "Why this, not the alternative?" section.
- [ ] Every non-trivial trick has a "Why does this work?" explanation.
- [ ] At least one worked numerical example (typically in the RoPE / attention sections).
- [ ] A concrete sentence is traced through the model in § 5, not an abstract "a token enters" narration.
- [ ] At least 3–4 custom diagrams as SVG files in `notes_assets/`, plus the top-level Mermaid flows.
- [ ] All math is in Unicode code blocks, zero LaTeX delimiters.
- [ ] Every code snippet has comments on the non-obvious lines.
- [ ] Tensor shapes annotated throughout.
- [ ] Glossary covers every acronym used.
- [ ] A "suggested exercise" in § 10 that a reader could actually do.

If any item is unchecked, you're not done.
