---
hide:
  - navigation
  - toc
---

# LLM Study Notes

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Careful notes on modern LLM internals__

    ---

    Walkthroughs of open-weight large language models, written for someone who
    knows Python and neural networks but wants to understand **why** each piece
    exists — not just what it does.

-   :material-image-outline:{ .lg .middle } __Visual and grounded__

    ---

    Every architectural trick (RoPE, GQA, SwiGLU, KV cache, …) comes with
    diagrams, a worked numerical example, and code from the reference
    implementation — annotated line-by-line.

-   :material-book-open-page-variant:{ .lg .middle } __Teach, don't summarize__

    ---

    If you could get the same thing from the paper's abstract or the model
    card, I haven't done my job. The goal is intuition that sticks.

-   :material-update:{ .lg .middle } __A living collection__

    ---

    One note today, more coming. Each model gets its own deep dive.

</div>

## Available notes

- [**Llama 3** (Meta, 2024)](notes/llama3.md) — full architecture walkthrough of the
  reference implementation, including RoPE, Grouped-Query Attention, SwiGLU,
  the KV cache, and a concrete worked example of a sentence flowing through
  the model.

## Coming soon

- Mistral 7B — sliding-window attention
- DeepSeek V2 — Multi-head Latent Attention (MLA) and mixture-of-experts
- Qwen 2 — how a pragmatic Chinese-first model differs

## What each note contains

Each model note follows the same skeleton so you can compare apples to apples:

1. **TL;DR** — one paragraph, zero jargon.
2. **Mental-model diagram** — end-to-end forward pass in one picture.
3. **Key hyperparameters** — with intuition for what each knob controls.
4. **Architecture walkthrough** — component by component, data-flow order.
5. **Forward pass, narrated** — a concrete sentence traced through the model.
6. **Training vs inference** — what changes between them.
7. **Zoom-out comparisons** — vs. the 2017 Transformer, vs. an older model in
   the same family, vs. one contemporary competitor.
8. **What's actually novel** — and what isn't.
9. **Glossary + further reading.**

## About

Written by Jun Yu Chen while learning this stuff. If you spot an error, a
confusion, or a better intuition, please open an issue — I'd rather get it
right than save face.

<small>Built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
Source Markdown on [GitHub](https://github.com/spearsheep/johnny-is-confused).</small>
