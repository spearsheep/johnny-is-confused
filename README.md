# johnny-is-confused

> Hi — I'm Johnny, and I'm frequently confused. This repo is where I write
> down what I'm trying to figure out. Mostly machine learning and AI, but
> occasionally other stuff. Writing slowly helps me think.

**→ <https://spearsheep.github.io/johnny-is-confused/>**

Nothing in here claims to be authoritative. It's more of a public notebook
than a textbook. If something helps you too, great; if you find a mistake,
please tell me — I'd rather be right than look right.

## What's in here

### Study notes
Careful, diagram-heavy write-ups of ML/AI things I've actually sat down and
worked through. Each note aims to explain something to a version of me who
hasn't learned it yet.

Published so far:
- [**Llama 3** (Meta, 2024)](docs/notes/llama3.md) — architecture walkthrough
  of the reference implementation. RoPE, GQA, SwiGLU, KV cache, plus a
  concrete sentence traced layer-by-layer. ~5,000 words.

Working on / want to get to:
- LLMs: Mistral (sliding-window attention), DeepSeek V2 (MLA, MoE), Qwen, Gemma
- Computer vision: ViT, DINOv2, SAM
- Recommendation / retrieval: two-tower models, dense vs sparse retrieval
- Whatever confuses me next.

### Blog
Shorter, less-structured pieces. Sometimes about what I'm studying,
sometimes about life. None posted yet.

## Local setup

For me, mostly, but the site builds the same way on any machine:

```bash
pip install -r requirements.txt
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build   # static site into ./site/
```

Every push to `main` auto-deploys via
[GitHub Actions](.github/workflows/deploy.yml).

## Layout

```
docs/
  index.md              landing page
  notes/                long-form study notes
  blog/                 shorter posts
  assets/notes/         diagrams, plots, images (per note)
mkdocs.yml              site config
requirements.txt        MkDocs deps
```

The `llama3-ref/` directory in my local workspace holds upstream
Meta-llama source code that I read while writing the Llama 3 note. It's
gitignored — the notes link to the official upstream repo instead.

## Found an error?

Please [open an issue](https://github.com/spearsheep/johnny-is-confused/issues)
or a pull request. I'd rather fix it than save face.

## License

Writing and diagrams here are licensed under
[Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/):
read them, share them, learn from them. Commercial reuse — including
reselling the notes as a paid product — needs my permission. Short code
snippets from third-party projects (like Meta's llama3) are quoted for
teaching purposes; the full upstream sources remain with their original
authors and licenses.
