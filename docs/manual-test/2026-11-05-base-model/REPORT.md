# Manual session — base model (`Qwen/Qwen3-0.6B`)

- **Date:** 2026-11-05
- **Code:** branch `ytb-e2e` at `a358c7f`, plus `scripts/manual_session.py`
- **App:** `uv run python -m ytb.api --model Qwen/Qwen3-0.6B` (GPU, bf16), Whisper `small` on CPU
- **Driver:** `uv run python scripts/manual_session.py --out docs/manual-test/2026-11-05-base-model`. It runs headless Chromium at 1280×860, types links at 60 ms per key, and pauses like a reader.
- **Recording:** [session.webm](session.webm), 65 s, the whole session
- **Rounds:** 1. Every app checklist item passed the first time, so no code fixes were needed.

| # | Scenario | Expected | Observed | OK | Screens |
|---|---|---|---|---|---|
| 01 | Open page | Model in header; field + button; clean layout | Header shows `Qwen/Qwen3-0.6B`; nothing overlaps | ✅ | [home](01-home.png) |
| 02 | TED talk `arj7oStGLkU` (captions) | Button disabled + timer while waiting; summary; stats `en` / `captions` | Disabled button with "Working… 2 s" plus the local-transcription hint. Summary done in 6 s: `en`, `captions`, 4 chunks, 2,898 → 130 tokens | ✅ | [input](02-captions-input.png) · [waiting](02-captions-waiting.png) · [done](02-captions-done.png) |
| 03 | Same talk as `youtu.be/arj7oStGLkU` | Transcript served from cache (no new YouTube request) | App log shows `cache hit arj7oStGLkU` and no second fetch; same summary in 5 s (summarization reruns) | ✅ | [input](03-cached-input.png) · [done](03-cached-done.png) |
| 04 | ~1 h talk `zjkBMFhNj_g` | Finishes; chunks > 1 | 12,944 tokens → **14 chunks** → combined summary in 19 s | ✅ | [input](04-long-input.png) · [waiting](04-long-waiting.png) · [done](04-long-done.png) |
| 05 | `https://vimeo.com/123` | Plain-language error; button enabled again | "Check the link or text: That is not a YouTube video link." No raw JSON; button enabled | ✅ | [input](05-invalid-input.png) · [done](05-invalid-done.png) |
| 06 | Pasted ~150-word news paragraph | Summary; source `—` | Summary in 1 s, 1 chunk, source `—` | ✅ | [input](06-text-input.png) · [done](06-text-done.png) |
| 07 | Caption-less video | Source "local transcription" | **Not run.** The one candidate checked (`jNQXAC9IVRw`) has captions, and the search stopped there to avoid YouTube's rate limit. The Whisper fallback was verified directly on the same video in Task 3, returning `en` plus the elephants quote in 3 s on the CPU. | — | — |
| log | `runs/app.log` | No `Traceback`, no 500 | Only `fetch`/`cache hit` INFO lines and 200/422 responses | ✅ | — |

The `waiting` shots for 05 and 06 already show the final state, because those requests finished in under the 1.5 s pause.

## Summary quality (base model — expected to improve after fine-tuning)

The app works, but the zero-shot base model is a weak summarizer.

- **06 (pasted text) has many wrong facts:**
  - "by **302**" instead of 2028
  - "**250** buses" instead of 240
  - "**65%**" instead of 60%
  - "first **100** buses" instead of 40
  - "**riders** will receive training" instead of drivers
- **02 has five sentences**, not 3–4, but the gist is right: procrastination, the Instant Gratification Monkey, the Rational Decision-Maker, the Panic Monster.
- **04 opens with "Hi everyone!"**, which is copied from the talk. The rest is a reasonable but generic gist of the LLM talk.

These are what the fine-tuned model is measured against in `docs/manual-test/<date>-finetuned/`.
