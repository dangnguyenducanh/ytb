# autoresearch — ytb summarizer

Adapted from karpathy/autoresearch `program.md` (commit 228791f; the upstream reference copy is in `/home/aiface/dangnguyenducanh/autoresearch-upstream`). Same loop, different target: instead of pretraining a small GPT, improve the **ytb YouTube summarizer**, a fine-tuned `Qwen/Qwen3-0.6B` that the app serves.

## Setup

The human launched this run and pre-approved the setup, so do not wait for confirmation.

1. **Run tag:** `nov5`. Create the branch `autoresearch/nov5` from the current commit. If it exists, use `nov5b`, `nov5c` and so on.
2. **Read the in-scope files:**
   - `autoresearch/prepare.py` is fixed: constants, the dev set, the ground-truth evaluation. **Do not modify it.**
   - `autoresearch/train.py` is **the only file you edit**.
   - For context only (read, never edit): `src/ytb/data.py` (examples, prompt masking, caption-style augmentation), `src/ytb/prompt.py` (prompt and decoding, which the app uses), `src/ytb/rouge.py`, `docs/training/2026-11-05-loss-plateau-assessment.md` (what has been tried and learned).
3. **Verify the starting model** exists: `ls /home/aiface/dangnguyenducanh/ytb/models/ytb-summarizer/model.safetensors`.
4. **Initialize `autoresearch/results.tsv`** with the header row. Then run `uv run python autoresearch/prepare.py --score-init > autoresearch/run.log 2>&1` and record its line as status `reference` with the description `v1 as-is (no training)`. This is the score to beat.
5. **Start the loop.** The first experiment is `train.py` exactly as it is (the baseline).

## Experimentation

- **The run:** each experiment is `uv run python autoresearch/train.py > autoresearch/run.log 2>&1`.
- **Time:** about 10 minutes of training (`prepare.TIME_BUDGET`), plus model load and data prep, plus scoring 200 articles. Expect ~15–17 minutes in total.
- **Goal:** the **highest `val_score`**, the mean of ROUGE-1, ROUGE-2 and ROUGE-Lsum (×100) on 200 fixed CNN/DailyMail validation articles. It is scored with the app's own prompt and decoding. **Higher is better** (the opposite of upstream's val_bpb).

**What you CAN do:** edit `autoresearch/train.py`. Everything in it is fair game:
- learning rate, schedule, warmup, optimizer, label smoothing, batch and accumulation
- how many unseen rows to use, data filtering and ordering, the caption-style augmentation fraction (re-implement it locally if you want a different one)
- freezing layers, LoRA versus full fine-tuning, loss variants
- self-distillation or rejection sampling (generate candidates, keep the best by ROUGE against the reference, train on them), and so on

**What you CANNOT do:**
- Modify `autoresearch/prepare.py` or anything under `src/ytb/`. The app serves this model with `ytb.prompt`, so the prompt and decoding are fixed. Training examples must keep using `ytb.data.prepare` / `ytb.prompt.build_prompt` so the trained model matches what the app sends.
- Train on, or look at, the CNN/DM **validation** split (the dev set comes from it) or **test** split (the human's final check). Training data is the train split only; use `prepare.unseen_train_rows(n)` for rows v1 never saw.
- Install packages or add dependencies.
- Kill other users' GPU processes. The GPU is shared, but VRAM is a soft constraint, as upstream.
- `git push`. Disable or bypass hooks: if a GateGuard hook asks for facts before a command, state them briefly and retry.

**Noise and simplicity:** ROUGE on 200 articles moves about ±0.5 from noise alone. **Keep a change only if `val_score` beats the best kept score by ≥ 0.3.** Otherwise discard it, unless it is a pure simplification at equal score. Upstream's simplicity criterion applies: a small gain that adds ugly complexity isn't worth keeping.

## Output format

`train.py` ends by printing:

```
---
val_score:        40.123456
rouge1:           43.21
rouge2:           19.87
rougeLsum:        40.12
training_seconds: 600.3
total_seconds:    940.0
peak_vram_mb:     17290.1
num_steps:        157
dev_articles:     200
```

Read it with `grep "^val_score:\|^rouge1:\|^peak_vram_mb:" autoresearch/run.log`.

## Logging results

`autoresearch/results.tsv` is tab-separated and untracked by git. Its header is:

```
commit	val_score	rouge1	memory_gb	status	description
```

- **commit:** the short hash.
- **val_score:** use `0.000000` for a crash.
- **memory_gb:** `peak_vram_mb / 1024` to 1 decimal place; `0.0` for a crash.
- **status:** `reference`, `keep`, `discard` or `crash`.
- **description:** a short description of the idea.

## The experiment loop

LOOP FOREVER:

1. Look at the git state.
2. Change `autoresearch/train.py` with one idea, then `git commit`.
3. Run the experiment as above. Never `tee` it, and never let the output flood your context.
4. Read the result. If the grep output is empty, the run crashed: read `tail -n 50 autoresearch/run.log`. Fix it if it's a typo; otherwise log it as `crash` and move on.
5. Log the result to `results.tsv`.
6. If it is a **keep** (≥ 0.3 above the best kept score), advance the branch and save the weights:
   `rm -rf /home/aiface/dangnguyenducanh/ytb/models/ytb-summarizer-auto && cp -r runs/autoresearch/last /home/aiface/dangnguyenducanh/ytb/models/ytb-summarizer-auto`
   Otherwise `git reset --hard` to the last kept commit.

**Timeout:** if a run passes 30 minutes, kill it and treat it as a failure.

**NEVER STOP.** Once the loop has begun, do not stop to ask the human whether to continue. They may be away and expect you to keep going until they stop you. If you run out of ideas, re-read the in-scope files and the training assessment, combine near-misses, and try bolder recipes (rejection-sampling self-distillation, LoRA with a higher learning rate, a data curriculum). Each experiment takes about 15 minutes, so that's about 4 per hour.
