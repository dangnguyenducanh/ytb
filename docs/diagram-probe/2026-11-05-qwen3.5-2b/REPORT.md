# Qwen3.5-2B zero-shot diagram probe — 1 sample, text→TikZ and image→TikZ

- **Date:** 2026-11-05
- **Model:** `Qwen/Qwen3.5-2B` (VLM), bf16, greedy, thinking off, ≤ 1500 new tokens
- **Design under test:** [`2026-11-05-diagram-codegen-eval-design.md`](../../superpowers/specs/2026-11-05-diagram-codegen-eval-design.md)
- **Status:** throwaway spike. One sample is an anecdote, not a benchmark.

## Setup

- **Sample** (`sample.json`): DiagramGen-style text-to-diagram instruction for a 7-node, 7-edge flowchart of the ytb pipeline, with one labelled edge. The official `OpenRaiser/DiagramGen` files are gated for this HF account, so the sample is hand-made and its ground truth is exact.
- **Reference** (`reference.tex` → [reference.png](reference.png)): compiles, and the extractor scores it 1.0 / 1.0 / 1.0. That check caught two bugs before the model ran; see "Probe bugs fixed" below.
- **Tasks:**
  1. **Text → TikZ:** the instruction only.
  2. **Image → TikZ:** `reference.png` plus "reproduce this diagram".
- **Loop:** `pdflatex -halt-on-error`. On failure, the first `!` error block goes back to the model, for at most 2 fix rounds (the Check & Fix agent).
- **Scoring** (design §4.2):
  - node and directed-edge F1 with fuzzy label match, plus edge-label accuracy
  - text collisions from `pdftotext -bbox`
  - `analyze.py` also scores structure straight from the uncompiled code, and tests a deterministic pre-compile lint

## Results

| | Text → TikZ | Image → TikZ |
|---|---|---|
| Compile@1 | ✗ | ✗ |
| Compile@fix (2 compiler-feedback rounds) | ✗, same error all 3 times | ✗, same error all 3 times |
| Tokens / seconds per attempt | 815–1,329 / 16–27 s | 473–652 / 10–13 s |
| **Node F1** (read from code, uncompiled) | **1.00** (7/7) | **1.00** (7/7) |
| **Edge F1** (read from code) | **1.00** (7/7) | 0.92: missed Fetch captions → Whisper fallback |
| Edge-label accuracy | 1.00 | 0.00: "no captions" put on a duplicate Whisper → cache edge |
| Compiles after deterministic lint | ✗, a 6th error class remains | ✓: [image2code-lint.png](image2code-lint.png) |
| Text collisions after lint | — | 3: overlapping boxes, illegible |
| **Strict Success@1 / @fix** (design §4.3) | **0 / 0** | **0 / 0** |

## Why it failed: error taxonomy

Every attempt hit error 1 first. Each later error only showed up once the one before it was patched.

| # | Error | Seen in | Effect |
|---|---|---|---|
| 1 | Style named after the shape it uses, e.g. `ellipse/.style={ellipse, …}` or `rectangle/.style={rectangle, …}` | both tasks, all 6 attempts | endless self-expansion → "TeX capacity exceeded" |
| 2 | `rounded rectangle` without `\usetikzlibrary{shapes.misc}` | text | unknown shape |
| 3 | undefined colour `darkgreen` | text | xcolor error |
| 4 | `edge[…] {no captions}`, which needs `node {…}` | image | "(, +, coordinate, pic, or node expected" |
| 5 | `\path[…]` with a style that has no `draw` | image | compiles, but every edge is **invisible**; only a render check catches it |
| 6 | `edge label={…}` on a `--` path (only valid on `to`/`edge`) | text | "Undefined control sequence" |

In the image task's layout, `right of=` with 1.5 cm spacing makes wide boxes overlap, which caused the 3 collisions.

The compiler feedback never helped: for error 1 the model's "fix" was to try to raise LaTeX's memory limit. "TeX capacity exceeded" doesn't name the cause, and a 2B model can't infer it.

## What this changes in the design

1. **The model already understands the diagrams; it doesn't know TikZ's rules.** It got 7/7 nodes on both tasks, and in the text task every edge and the label too. The failures are dialect errors, which SFT on compile-verified code (design §5.1–5.2) targets directly. Zero-shot Compile@1 is 0 here.
2. **Raw compiler errors aren't enough for a 2B Check & Fix loop.** Add a **deterministic lint** before compiling (rules 1–5 above are a few lines each, in `analyze.py`). Rewrite common TeX errors into plain hints before sending them back. Harvest (broken → fixed) pairs as training data (design §5.4).
3. **Score structure from code as a separate diagnostic** (T1-static). It separates "understood the diagram" from "wrote valid code"; here those two point in opposite directions.
4. **Legibility must be part of the headline.** The lint-fixed image output scores node F1 1.0 and edge F1 0.92, yet the picture is unreadable. Strict Success (zero collisions) fails it; a compile-plus-structure metric would pass it.
5. **Speed:** the optimised kernels for Qwen3.5's linear-attention layers (`causal-conv1d`, `flash-linear-attention`) aren't installed, so generation fell back to a slower reference path. Install them before running the dev set or RL rollouts.

## Probe bugs fixed along the way

These were found by the reference self-check and by reading the outputs.

- The reference used a style named `step`, which clashes with TikZ's built-in `step` key. It is renamed to `proc`.
- The extractor skipped the first node, because statements can start with the picture options. It now starts at the first `\node`, `\draw` or `\path`.
- The extractor required a literal `->`, but arrowheads often come from a style. It now counts any `\draw`/`\path` connecting two named nodes, and reads `edge label=`.

## Files

| File | What |
|---|---|
| `sample.json`, `reference.tex/.pdf/.png` | instruction, ground truth, reference |
| `probe.py` | runs both tasks with the fix loop and scores them (`uv run --with pillow --with torchvision python …/probe.py`) |
| `analyze.py` | static structure scores and the deterministic lint (`uv run python …/analyze.py`) |
| `text2code.attempt{0,1,2}.tex`, `image2code.attempt{0,1,2}.tex` | raw model outputs per attempt |
| `text2code-lint.tex`, `image2code-lint.tex/.pdf/.png` | after the lint |
| `results.json` | every number above |

## Next, to get real numbers

1. **Accept the dataset gate** at https://huggingface.co/datasets/OpenRaiser/DiagramGen (account `thecapybaracoder`). Then run this same probe over all of DiagramGeneration and DiagramCoding.
2. **Build the 300-sample synthetic dev set** (design §5.1). This sample is the template.
