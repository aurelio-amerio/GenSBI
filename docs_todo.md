# Documentation TODOs for GenSBI

This file collects actionable suggestions to improve the accessibility and usability of the GenSBI library documentation and codebase.

---

## 1. Quick Start & Installation - Done
- Add a minimal working example (5–10 lines) to the top of the main README and in `docs/getting_started/quick_start.md`.
- Include a copy-paste block for: install, import, define a toy simulator, train, sample posterior.

## 2. API Reference - Done
- Add or auto-generate (with Sphinx autodoc or mkdocs) an “API Reference” section.
- Include short usage examples for each main class/method (e.g., `Flux1FlowPipeline`, `PosteriorWrapper`).

## 3. Model & Pipeline Overview
- Add a conceptual diagram or a “How GenSBI is structured” page.
- Explain: what is a pipeline, what is a model, what is a recipe, and how do they relate.

## 4. Tutorials & Examples
- Link to the most important notebooks from the main README and from `docs/index.md`.
- Add a “Tutorials” or “Examples” section in the docs sidebar, with a short description for each notebook.

## 5. Validation & Best Practices
- Add a “Common Pitfalls” or “FAQ” section to the validation guide (e.g., shape mismatches, memory errors, device/cuda issues).
- Add a “Best Practices” page: how to choose batch size, debug training, check if your model is learning.

## 6. Model Cards
- Expand `docs/getting_started/model_cards.md` with a table summarizing available models, their strengths/weaknesses, and recommended use cases.

## 7. Codebase Navigation
- Add a “Contributing” page and a “Codebase Overview” for developers.
- Explain the code layout, how to add a new model, and how to run tests.

## 8. Docstrings & Inline Examples
- Expand docstrings in `src/gensbi/recipes/`, `src/gensbi/models/`, etc., with usage examples and parameter explanations.

## 9. Troubleshooting
- Add a “Troubleshooting” section to the docs, with solutions to common errors (import errors, CUDA issues, shape mismatches, etc.).

## 10. Search & Navigation
- Ensure the documentation site has a working search bar and a clear sidebar structure (group by “Getting Started”, “API Reference”, “Tutorials”, “Validation”, “Developer”, etc.).

---

**Summary Table**

| Area                | Current State         | Recommendation                                 |
|---------------------|----------------------|------------------------------------------------|
| Quick Start         | Exists, but could be more prominent | Add minimal example to README/docs front page |
| API Reference       | Lacking              | Add auto-generated API docs with examples      |
| Model/Pipeline Docs | Not explicit         | Add conceptual overview/diagram                |
| Tutorials/Examples  | Present, not highlighted | Link from main pages, add descriptions     |
| Validation/FAQ      | Good, but no FAQ     | Add FAQ/pitfalls/troubleshooting section       |
| Model Cards         | Exists, could be richer | Add summary table, use cases                |
| Codebase Navigation | No dev guide         | Add “Contributing” and “Codebase Overview”     |
| Docstrings          | Basic                | Expand with usage and parameter docs           |
| Troubleshooting     | Missing              | Add dedicated section                          |
| Search/Navigation   | Depends on Sphinx config | Ensure sidebar/search are clear             |

---

Would you like a concrete example for any of these (e.g., a sample quick start, API doc, or FAQ entry)?
