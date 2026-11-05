# Manual session — fine-tuned model (`models/ytb-summarizer`, v1)

- **Date:** 2026-11-05
- **Model:** v1 fine-tune of Qwen3-0.6B, ROUGE-1/2/Lsum **40.06 / 17.84 / 37.35** on 500 CNN/DM test articles, which beats lead-3 (39.96 / 17.27 / 36.14). Decoding: greedy with `repetition_penalty=1.1`.
- **App:** `uv run python -m ytb.api --model models/ytb-summarizer`. It uses the redesigned TypeScript page, whose top bar shows "ytb-summarizer · fine-tuned · ROUGE-1 40.1".
- **Driver:**

  ```
  uv run python scripts/manual_session.py --out docs/manual-test/2026-11-05-finetuned --storage runs/session-storage.json
  ```

  The browser started with the four base-model runs from [the base session](../2026-11-05-base-model-redesign/), so the history and the "Base vs fine-tuned" panel show both models on the same videos.
- **Recording:** [session.webm](session.webm)
- **Rounds:** 1. Every app check passed; no code changes were needed for this session.

## Checks

| # | Scenario | Observed | OK | Screens |
|---|---|---|---|---|
| 01 | Open page | Top bar: green ready dot, `ytb-summarizer`, fine-tuned, ROUGE-1 40.1. Title "Four summaries so far." from the carried history | ✅ | [home](01-home.png) |
| 02 | TED talk `arj7oStGLkU` | Summary in 3.4 s, 4 chunks, English captions | ✅ | [waiting](02-captions-waiting.png) · [done](02-captions-done.png) |
| 03 | Same talk via `youtu.be` | App log shows `cache hit arj7oStGLkU`, no new fetch | ✅ | [done](03-cached-done.png) |
| 04 | 1-hour talk `zjkBMFhNj_g` | 14 chunks, 8.9 s. The compare panel shows base and fine-tuned side by side | ✅ | [done](04-long-done.png) |
| 05 | `https://vimeo.com/123` | "Check the link or text: That is not a YouTube video link." Run details cleared | ✅ | [done](05-invalid-done.png) |
| 06 | Pasted article | 3 sentences in 0.6 s server time, all facts correct | ✅ | [done](06-text-done.png) |
| log | `runs/app.log` | Only cache hits; no traceback, no 500 | ✅ | — |

## Base vs fine-tuned, same inputs

Base summaries come from [the base session](../2026-11-05-base-model-redesign/session.json); fine-tuned ones from this session's [session.json](session.json).

**06 — pasted news paragraph** (facts: 2028, 240 buses, three garages, 60 %, first 40 buses, *drivers* trained)

| Base Qwen3-0.6B | Fine-tuned v1 |
|---|---|
| The city plans to replace its diesel bus fleets with electric ones by **302**, covering **250** buses, including new charging stations. The switch will reduce emissions by **65%** and save around $9 million annually, with the first **100** buses arriving next spring. Officials mention federal grants covering nearly half the cost, and **riders** will receive training. A meeting is scheduled to finalize depot locations. | Riverton will replace all its diesel buses with electric ones by **2028**. The plan includes new charging depots at **three existing garages**. The switch will cut the fleet's carbon emissions by **about 60 percent**. |
| 5 factual errors, 5 sentences | 0 errors, 3 sentences |

**02 — TED talk (procrastination)**

| Base Qwen3-0.6B | Fine-tuned v1 |
|---|---|
| In college writing a thesis required structured planning, starting from light work to a 9-week senior thesis. … procrastination (the "Instant Gratitude Monkey") and decision-makers (the Rational Deciders) … the "Dark Playground" … the Panic Monster … *(5 sentences, 5.9 s)* | Tim Urban wrote a 90-page paper comparing his brain to a non-procrastinator's. He found an MRI lab to compare his brain to a non-procrastinator's. The instant gratification monkey doesn't care about the future, he only cares about easy and fun things. The rational decision-maker knows the future, but he wants to do things that make sense to him right now. *(4 sentences, 3.4 s)* |

**04 — 1-hour talk (intro to LLMs)**

| Base Qwen3-0.6B | Fine-tuned v1 |
|---|---|
| Hi everyone! LLMs, like human language models in games, learn from data and reward systems to improve performance. … security in LLM development, as seen in threats like prompt injections … Llama 2 and Scale-Ai … *(17.9 s)* | Llama **270B** is an open-source model, so we know quite a bit about how it was trained. The training process involves taking a chunk of the internet and compressing it into a zip file. This is essentially a compression of the internet into a single **parameter**. The neural network is used to predict the next word in a sequence. *(8.9 s)* |

## Quality notes

- **The fine-tune writes like news highlights:** shorter, concrete, names the subject, gets the numbers right on news-like text (06), and runs about twice as fast because the summaries are shorter.
- **Spoken transcripts expose its limits:**
  - 02 repeats a clause.
  - 04 summarizes mainly the *opening* of a 1-hour talk: the combine step keeps the first chunks' content.
  - 04 also merges "Llama 2 70B" into "270B" and says "single parameter" where the talk says a file of parameters.

  Both models were trained only on news. Transcript-style training data is the next lever (training assessment, solution 6).
- **v2** (smoothing 0, lr 3e-5) scored 0.3 ROUGE below v1, which is within noise, so v1 is the served model. See [the training assessment](../../training/2026-11-05-loss-plateau-assessment.md).
