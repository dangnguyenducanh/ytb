# Diagram code generation (DiagramGen, Qwen3.5-2B) — redesigned evaluation and update loop

> Written 2026-11-05 from the user's pasted design (Vietnamese, reproduced verbatim in the Appendix) and their request: "brainstorm and redesign the eval and updates, also test this models with 1 sample graph and report".
>
> Companion spike: `docs/diagram-probe/2026-11-05-qwen3.5-2b/` holds the one-sample zero-shot test of `Qwen/Qwen3.5-2B` and its `REPORT.md`.
>
> Status: **design for review**. Nothing beyond the throwaway spike has been built.

## 1. Intent

Train a small model, the backbone `Qwen/Qwen3.5-2B`, to write **diagram code** that compiles into a correct diagram. The code is LaTeX/TikZ, Graphviz DOT or Mermaid. The model is not asked to paint pixels. The three tasks come from DiagramAgent (arXiv:2411.11916):

| Task | Input | Output |
|---|---|---|
| Generation | text description | diagram code |
| Coding | diagram image | code that reproduces it |
| Editing | existing code (or image) + change request | modified code |

For this repo the eventual use is turning a video summary into a flowchart or mind map. That integration is a later step, not part of this design.

**Why redesign.** The pasted design scores mostly whether code *compiles*, plus loose visual similarity. Both can be gamed: an empty `tikzpicture` compiles, and a picture can look similar while having the wrong nodes. It also plans to *train* on DiagramGen, which turns out to be a test-only benchmark (see §2).

## 2. Facts found while checking (2026-11-05)

| Fact | Consequence |
|---|---|
| `OpenRaiser/DiagramGen` has only **test** splits, under 1K samples, in 3 configs: `DiagramGeneration`, `DiagramCoding`, `DiagramEditing` | It cannot be the training set. Using it for training or prompt tuning contaminates the evaluation, so it becomes the **frozen held-out benchmark** and training data must come from elsewhere (§5). |
| The dataset is **gated** (auto-approve). HF account `thecapybaracoder` gets 403 | The user must click "Agree" once on the dataset page. Until then, the spike uses a hand-made sample in the same style. |
| `Qwen/Qwen3.5-2B` is already a **vision-language** model (`Qwen3_5ForConditionalGeneration`, image-text-to-text), and transformers 5.22 supports it | One backbone covers all three tasks. The pasted "use Qwen-VL for image tasks" split is unnecessary. |
| Its chat template only thinks when `enable_thinking=True` | Default answer mode is fine for code generation; thinking stays off. |
| This host has `pdflatex`, `pdftoppm` and `pdftotext`, but **no Graphviz `dot` and no Mermaid CLI** | TikZ can be evaluated now. DOT and Mermaid need their toolchain installed first (conda-forge `graphviz` without sudo; Mermaid CLI pulls headless Chromium). |
| Full fine-tuning of 2B with AdamW needs ~32 GB of optimizer state alone | Training uses **LoRA** (the pasted r=64, α=128), bf16: a fraction of the memory and faster iterations. |
| **Spike (1 sample, zero-shot)**: Compile@1 0/2, Compile@fix 0/2, Strict Success 0/2. But nodes 7/7 on both tasks, and edges 7/7 (text) and 6/7 (image) read from the uncompiled code. Every failure was a TikZ dialect error, led by a style named after its own shape. Compiler feedback never fixed it. After a deterministic lint the image output compiled but was illegible (3 text collisions) | The model understands diagrams but not TikZ's rules, so SFT on compile-verified code is the main lever. Add a pre-compile lint stage and plain-language error hints (§5.4), a static structure diagnostic (§4.2), and keep legibility in the headline. Details: `docs/diagram-probe/2026-11-05-qwen3.5-2b/REPORT.md` |

## 3. Goals, non-goals, anti-goals

- **Goals**
  - A single headline number that only goes up when the diagram is *right*: compiles **and** has the right nodes and edges **and** is legible.
  - Every stage of the update loop (SFT → RL → self-correction) is gated by that number on a dev set.
  - DiagramGen is touched only for final reports.
- **Non-goals**
  - Pixel-level image generation.
  - Matching the reference *code* text.
  - Beating the paper's numbers on day one.
- **Anti-goals.** Each of these would count as failure even if the numbers look good:
  - a metric that rewards compiling-but-empty or compiling-but-wrong diagrams
  - any DiagramGen sample, or near-duplicate, in the training data
  - image similarity as the deciding metric, because many different layouts are equally correct
  - compiling model-written LaTeX with shell-escape enabled

## 4. Redesigned evaluation

### 4.1 Splits

| Split | Source | Use |
|---|---|---|
| `dev` | 300 held-out synthetic samples (§5.1) + 100 compile-filtered real samples | every decision: checkpoints, prompts, reward weights |
| `test` | DiagramGen, all three configs, frozen | final report per stage only; never inspected sample-by-sample for tuning |

Results are reported per task × diagram type × code language. Averages hide where the model fails.

### 4.2 Metric tiers

| Tier | What | How | Why |
|---|---|---|---|
| **T0 Executability** | `Compile@1` (first answer) and `Compile@fix` (after ≤ 2 compiler-feedback rounds), reported **separately** | Sandboxed compile: `pdflatex -no-shell-escape -halt-on-error`, 90 s timeout, no network. `dot -Tjson` and `mmdc` once installed | Keeps the Check & Fix agent's contribution measurable instead of hiding it in one number |
| **T1 Content fidelity** (primary) | node-label F1, directed-edge F1, edge-label accuracy | Extract a graph from the generated code: TikZ statement parser (built in the spike), DOT via `dot -Tjson`, Mermaid via its parser. Compare with the ground-truth graph using fuzzy label matching (normalised, ratio ≥ 0.8, greedy one-to-one) | "Is it the right diagram?" is what users care about, and it is independent of code style |
| T1-static (diagnostic) | node/edge F1 read from the code without compiling | Same extractor run on the raw answer | Separates "understood the diagram" from "wrote valid code". In the spike these diverged completely: 1.0 vs 0. |
| T1-edit | edit success + preservation | Diff the before and after graphs. The requested change must happen (edit F1), and everything else must stay unchanged (preservation rate) | The pasted design had no editing metric |
| **T2 Instruction adherence** | per-sample checklist: shapes, direction, colours, labels, grouping | Rule checks where the code makes it explicit (e.g. node style resolves to `ellipse`); otherwise a VLM judge on the render with a fixed rubric, calibrated against 30 human-labelled samples | Content can be right while the requested style is ignored |
| **T3 Legibility** | rendered-text collisions, clipped text, edge crossings | `pdftotext -bbox` word-box intersections (built in the spike); DOT layout JSON for crossings | Replaces "visual aesthetic" with measurable defects |
| T3-diag | image similarity to the reference | DINOv2 / CLIP cosine on renders | **Diagnostic only**, never a gate, because different correct layouts score low |
| **T4 Cost** | output tokens, latency, fix rounds used | from generation logs | A 2B model that needs 3 fix rounds may be slower than a bigger model |
| Paper-compat | ROUGE-L, CodeBLEU, chrF vs reference code | as in DiagramAgent | Only to compare with published numbers; not used for decisions |

### 4.3 Headline metrics

- **Strict Success@1:** compiles at the first attempt, node F1 ≥ 0.9, edge F1 ≥ 0.9, and zero text collisions.
- **Strict Success@fix:** the same, allowing ≤ 2 compiler-feedback rounds.
- **Gate:** a training stage is kept only if Strict Success@1 on `dev` beats the previous stage. A drop of more than 2 points in any T0–T3 tier blocks it, even if the headline improved.

### 4.4 Report format

One `results.json` per run, plus a Markdown table and a gallery of the 10 worst samples rendered side by side with their references. The spike's `results.json` is the prototype for this format.

## 5. Redesigned update loop

### 5.1 Stage 0 — data (structure first, so ground truth is exact)

1. **Synthetic structure-first generator.**
   - Sample random graphs of several types: flowchart, DAG, tree, mind map, state machine, cycle; 3–20 nodes; optional edge labels and groups.
   - Render each to TikZ (and later DOT/Mermaid) through style templates, varying shapes, colours, direction and spacing.
   - Write the description from the graph: a template sentence, then a paraphrase by a local LLM (`gemma4:12b` is already in Ollama).
   - The ground-truth graph is known exactly, so T1 scoring and the RL reward are free.
2. **Real code corpora** (e.g. DaTikZ-style TikZ collections) for diversity. Every sample goes through the same sandboxed compile filter, and anything that doesn't compile is dropped (as in the pasted design).
3. **Image → code pairs for free:** render every kept code sample to PNG.
4. **Editing pairs by programmatic edits** on synthetic graphs: add, remove or rename a node; add or remove an edge; restyle. The expected code and graph are exact.
5. **Contamination guard:** drop any training sample whose description or code is a near-duplicate (MinHash/n-gram) of a DiagramGen sample.
6. **Format:** ChatML conversations, the same as the pasted design. The target is the full standalone document, wrapper included.

### 5.2 Stage 1 — SFT

- **Model and adapters:** `Qwen/Qwen3.5-2B` with LoRA r = 64, α = 128 on all linear layers of the language model. The vision tower is frozen. bf16; thinking off; Flash-Attention if available.
- **Mixing:** the three tasks are mixed (e.g. 50% generation, 30% image→code, 20% editing).
- **Framework:** HF `Trainer`/TRL + PEFT, the same stack as this repo's summarizer training (my call over LLaMA-Factory/Unsloth: fewer moving parts, one environment).
- **Gate:** Strict Success@1 on `dev` must beat zero-shot.

### 5.3 Stage 2 — compiler-in-the-loop RL

**GRPO with a verifiable reward** (no learned reward model), my call over DPO. With exact ground truth the reward is cheap and dense:

```
r = 1[compiles] · (0.4·nodeF1 + 0.4·edgeF1 + 0.2·constraint_pass_rate) − 0.1·min(collisions, 5) − λ·max(0, tokens − budget)
```

- The reward is 0 when the code doesn't compile. That is the pasted design's R_compilation, used as a *gate*, not as a free point, so an empty diagram earns nothing.
- Group size 4–8 per prompt, with a KL term to the SFT model.
- **Fallback:** DPO on pairs mined from SFT samples (best vs worst under the same reward). Use it if GRPO is too slow.
- **Gate:** Strict Success@1 on `dev` beats SFT, with no tier dropping more than 2 points.

### 5.4 Stage 3 — self-correction (Check & Fix)

- **Stage 3a, deterministic lint before compiling.** The spike found 5 error classes with mechanical fixes:
  - a style named after its own shape
  - a missing TikZ library
  - an undefined colour
  - an `edge` label missing `node`
  - `\path` without `draw`

  The lint runs before the compiler and is reported separately: `Compile@1` (raw) vs `Compile@lint`.
- **Stage 3b, LLM fix loop.** Compile → on error, map known TeX errors to a plain-language hint (e.g. "TeX capacity exceeded" → "a style probably refers to itself; rename it") → the model fixes it. At most 2 rounds. In the spike, raw compiler messages fixed nothing.
- **Training:** harvest (broken code + compiler error → fixed code) pairs from SFT and RL failures and add them to SFT data.
- **Plan agent:** kept only if an ablation on `dev` shows it raises Strict Success for a 2B model. Small models often lose accuracy on long plan-then-code prompts.

## 6. Constraints

- Disk is about 95% full (~50 GB free): the model is 4.5 GB, and datasets plus renders need pruning.
- Compiling untrusted LaTeX: always `-no-shell-escape`, a timeout, and a temp directory.
- DOT and Mermaid need their toolchain installed first. That's the user's call: conda-forge `graphviz` needs no sudo; Mermaid CLI downloads Chromium.

## 7. Calls I made (please check)

1. DiagramGen is test-only. Training data is synthetic-structure-first plus compile-filtered real code.
2. One VLM backbone (Qwen3.5-2B) for all three tasks; no separate Qwen-VL.
3. LoRA (r=64, α=128) rather than full fine-tuning: a fraction of the memory and faster iterations.
4. Headline metric is Strict Success; code-similarity metrics are kept only for comparability.
5. GRPO with a verifiable reward over DPO; DPO is the fallback.
6. TikZ first, because it is the only compiler installed; DOT and Mermaid follow once their tools are installed.
7. HF Trainer/TRL + PEFT rather than LLaMA-Factory/Unsloth.

## 8. Only the user can answer

1. **Accept the DiagramGen gate** on https://huggingface.co/datasets/OpenRaiser/DiagramGen with the `thecapybaracoder` account. This is needed for any real benchmark number.
2. **Code language priority:** TikZ only, or install Graphviz/Mermaid tooling to cover DOT/Mermaid too?
3. **Scope:** stand-alone diagram model, or wire it into ytb (summary → diagram) as the end goal? This changes which diagram types the synthetic data should favour, e.g. mind maps and flowcharts of talks.

---

## Appendix — source design (user paste, verbatim)

Để huấn luyện một mô hình sinh biểu đồ/đồ thị từ tập dữ liệu OpenRaiser/DiagramGen, bản chất là huấn luyện mô hình sinh mã biểu đồ có cấu trúc (LaTeX TikZ / Graphviz DOT / Mermaid) từ mô tả văn bản hoặc ảnh, thay vì tạo ảnh trực tiếp (pixel diffusion). [1, 2]
Quy trình end-to-end (E2E) từ khâu xử lý dữ liệu đến huấn luyện và triển khai được tổ chức theo kiến trúc tương tự nghiên cứu DiagramAgent (arXiv:2411.11916): [1, 2]

1. Phân loại tác vụ & Cấu trúc dữ liệu
Tập DiagramGen cung cấp các cặp dữ liệu đa phương thức: [1, 2]
Tác vụ 1: Text-to-Diagram Code (Code Agent): Nhận văn bản mô tả (instruction/prompt) → Sinh mã nguồn (LaTeX/TikZ, DOT, Mermaid).
Tác vụ 2: Image-to-Code / Editing (Diagram-to-Code Agent): Nhận hình ảnh biểu đồ hiện tại + yêu cầu chỉnh sửa → Tái tạo mã nguồn hoặc sửa đổi các node/cạnh tương ứng. [1, 2, 3]
Input (Text/Image) ──> LLM / VLM ──> Structured Code (.dot/.tex) ──> Compiler (pdflatex/dot) ──> Visual Graph

2. Thiết kế luồng huấn luyện End-to-End (E2E Pipeline)
Chuẩn bị và Lọc dữ liệu (Data Preprocessing & Compilation Filtering):
Compiler Verification: Chạy một pipeline tự động biên dịch toàn bộ mã trong tập dữ liệu (pdflatex, graphviz) sang ảnh PNG/SVG. Loại bỏ ngay các mẫu bị lỗi cú pháp để tránh huấn luyện mô hình sinh mã "rác" (non-compilable code).
Chuẩn hóa định dạng: Chuyển đổi dữ liệu về dạng hội thoại đa lượt (ChatML format) tương thích với thư viện fine-tuning.
Lựa chọn mô hình nền tảng (Backbone Model):
 Qwen/Qwen3.5-2B
Giai đoạn Huấn luyện Supervised Fine-Tuning (SFT):
Đầu vào (Prompt): Chỉ dẫn chi tiết về cấu trúc thực thể, quan hệ (nodes, edges, styles, subgraphs).
Mục tiêu (Target): Toàn bộ khối mã sạch, bao gồm cấu trúc bao ngoài (ví dụ môi trường \begin{tikzpicture} hoặc digraph G { ... }).
Chiến lược: Áp dụng Full Fine-Tuning hoặc LoRA (r=64, α=128) với bfloat16 và Flash-Attention 2 qua các framework như LLaMA-Factory hoặc Unsloth.
Tối ưu hóa phản hồi (Compiler-in-the-Loop RL / DPO):
Reinforcement Learning: Dùng DPO (Direct Preference Optimization) hoặc GRPO/PPO để phạt nặng trường hợp mã không biên dịch được.
Reward Function:
\(R_{\text{compilation}}\): Điểm nhị phân (0 nếu lỗi cú pháp, 1 nếu biên dịch thành công ra ảnh).
\(R_{\text{visual\_match}}\): Đánh giá độ tương đồng layout/cấu trúc (CLIP/DINO visual score hoặc graph edit distance đối với số lượng node/edge).
Xây dựng Vòng lặp Tự sửa lỗi (Self-Correction / Multi-Agent Loop):
Tương tự kiến trúc DiagramAgent gồm 3 thành phần:
Plan Agent: Lập dàn ý cấu trúc các node/edge từ văn bản.
Code Agent: Sinh mã nguồn.
Check & Fix Agent: Đọc traceback lỗi của compiler (nếu có lỗi biên dịch) rồi ném ngược lại vào mô hình để tự sửa (self-debugging) trước khi trả kết quả cuối cùng cho người dùng. [1, 2, 3]

3. Đánh giá chất lượng (Evaluation)
Compilation Success Rate (CSR): Tỷ lệ phần trăm đoạn mã sinh ra có thể render thành công mà không gặp lỗi cú pháp.
Graph Structural Accuracy: Đo lường độ chính xác của các cặp quan hệ thực thể (Precision/Recall của nodes và directed edges).
Visual Aesthetic & Overlap: Đo mức độ chồng lấn chữ (text overlap) và tính cân đối bố cục sau khi render.
