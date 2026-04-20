# LLM Study Notes

> Careful, visual, pedagogical study notes on the internals of modern large
> language models. Written for someone who knows Python and neural networks
> but wants to understand _why_ each piece exists, not just _what_ it does.

**→ Read online: <https://spearsheep.github.io/johnny-is-confused/>**

Each note walks through one model's architecture component-by-component with
diagrams, annotated reference code, and a worked example of a sentence flowing
through the model end-to-end.

## Notes in this repo

| Model | Year | Status |
| --- | --- | --- |
| [Llama 3 (Meta)](docs/notes/llama3.md) | 2024 | Published |
| Mistral 7B | 2023 | Planned |
| DeepSeek V2 | 2024 | Planned |
| Qwen 2 | 2024 | Planned |

## What's in each note

1. **TL;DR** — one paragraph, zero jargon
2. **Mental-model diagram** — the whole forward pass in one picture
3. **Hyperparameters with intuition** — what each knob actually controls
4. **Architecture walkthrough** — every component in data-flow order, with
   annotated code from the reference implementation
5. **A concrete forward pass** — one real sentence, traced through the model
6. **Training vs inference** — what changes
7. **Zoom-out comparisons** — vs. the 2017 Transformer, vs. the prior
   generation, vs. a contemporary competitor
8. **Glossary + further reading**

## Local setup (for contributors)

```bash
pip install -r requirements.txt
mkdocs serve           # preview at http://127.0.0.1:8000
mkdocs build           # build static site into ./site/
```

The Markdown sources live under [`docs/`](docs/). Images and diagrams under
[`docs/assets/`](docs/assets/). Every push to `main` automatically rebuilds
and deploys the site via GitHub Actions
([`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)).

## Contributing

Found an error, a confusion, or a cleaner explanation? Please open an issue
or a pull request. I'd rather be right than save face.

## License

The written text, diagrams, and code snippets authored by me in this repo are
released under [**Creative Commons BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/)
— you're welcome to read, share, and learn from them; commercial use
(including reselling the notes as a paid product) requires permission.

Code from the reference implementations I reference (Llama 3, etc.) is owned
by their respective authors under their own licenses; I include only short
annotated snippets for teaching purposes. The full upstream code is **not**
redistributed in this repo — follow the links in each note to the original
repos.
