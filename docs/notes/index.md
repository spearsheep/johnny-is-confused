# Notes

Slow, careful write-ups of things I'm trying to understand. Mostly machine
learning and AI — LLM internals so far, with more territory to cover.

Every note here tries to hit the same bar: **could a motivated reader who
wasn't there when I wrote it still follow every step?** If the answer is no,
the note isn't done yet.

## Published

<div class="grid cards" markdown>

-   __[Llama 3 (Meta, 2024)](llama3.md)__

    ---

    Full architecture walkthrough of the reference implementation. RoPE,
    Grouped-Query Attention, SwiGLU, KV cache, the whole stack — plus a
    concrete sentence traced layer-by-layer through the model. My first real
    deep-dive and probably the most polished thing here.

    *~5,000 words · 5 diagrams · worked numerical examples*

</div>

## Working on / wanting to get to

Things that have caught my attention, roughly in the order I'm likely to
tackle them. No promises on timing — I write when I'm confused enough to
need to.

### Large language models

- **Mistral** — sliding-window attention; how does the local/global tradeoff
  actually work in practice?
- **DeepSeek V2 / V3** — Multi-head Latent Attention, MoE routing, the
  engineering behind the scaling claims
- **Qwen** — the pragmatic Chinese-first model family; what's different
- **Gemma** — Google's compact open models; architectural choices

### Computer vision

- **Vision Transformers (ViT)** — the first time attention worked on images
- **DINOv2** — self-supervised learning on images at scale
- **SAM** — Segment Anything, promptable segmentation

### Recommendation / retrieval

- Two-tower models — the workhorse of modern retrieval
- Dense vs sparse retrieval
- Approximate nearest neighbor search (HNSW, IVF)

### And probably

- Transformers from scratch in plain PyTorch
- A diffusion-model walkthrough when I'm brave enough
- Something on RL from human feedback

## Found an error?

Please [open an issue on GitHub](https://github.com/spearsheep/johnny-is-confused/issues).
I'd rather fix it than be wrong.
