# Manual session: the app with its newest design

- **Date:** 2026-11-05 (re-recorded with the customer-facing redesign, the new logo and light/dark themes).
- **Model:** base `Qwen/Qwen3-0.6B` writing structured notes (overview, key points per part, takeaway). This is the app's default since commit 6033aca.
  - This folder keeps its old name.
  - Its earlier capture, the v1 fine-tune with the old page, is in git at commit 79f55fc. It is summarized at the end.
- **App:** `uv run python -m ytb.api`, served on `http://127.0.0.1:18500`.
- **Driver:** `uv run python scripts/manual_session.py --out docs/manual-test/2026-11-05-finetuned`.
  - It uses a fresh browser with no carried history, so 01 is a real first visit.
  - After the five scenarios it switches the theme and re-opens the page at phone and tablet sizes.
- **Recording:** [session.webm](session.webm). **Log:** [session.json](session.json).
- **Rounds:** 3. Fixed during this session:
  1. The recorder waited for the old model chip, which the redesign removed. It now waits for the top bar's status. It also gained the dark-theme step and the phone and tablet pass.
  2. The two cached transcripts had no video length, so the page estimated it from the word count, at 150 words a minute.
     - That said "about 1 h 21 min" for the 1-hour talk. Its speaker talks fast.
     - The exact lengths are now backfilled from the caption timings: 14:03 and 59:50.
     - New videos record their length when fetched.
  3. During a new request, and after an error, the headline still described the previous clip. It now resets, the same way the details panel does.

## Checks

Desktop runs are 1280 × 860.

| # | Scenario | Observed | OK | Screens |
|---|---|---|---|---|
| 01 | First visit | Top bar "ytb \| YouTube video summarizer", moon toggle, Ready. Headline "Any YouTube video, read in a minute." Tiles at zero with "no videos yet". | ✅ | [home](01-home.png) |
| 02 | TED talk `arj7oStGLkU` | 2.8 s. **"A 14-minute video, read in 1 minute."** The line under it names "Inside the Mind of a Master Procrastinator \| Tim Urban \| TED" from TED, 2,250 words became 193 to read, 13 min saved. 2 parts, from English captions. | ✅ | [input](02-captions-input.png) · [waiting](02-captions-waiting.png) · [done](02-captions-done.png) |
| 03 | Same talk via `youtu.be` | App log `cache hit arj7oStGLkU`; same result. | ✅ | [done](03-cached-done.png) |
| 04 | 1-hour talk `zjkBMFhNj_g` | 6.9 s. **"A 60-minute video, read in 2 minutes."** "[1hr Talk] Intro to Large Language Models" from Andrej Karpathy, 12,156 → 465 words, 58 min saved. 7 parts, with times under the summary: 0:00–8:33 … 51:18–59:50. | ✅ | [done](04-long-done.png) |
| 05 | `https://vimeo.com/123` | "Check the link or text: That is not a YouTube video link." with Retry. Headline back to the default; About this video cleared. | ✅ | [done](05-invalid-done.png) |
| 06 | Pasted article | 2.3 s. "137 words, read in under a minute." "About this text": 137 → 131 words, pasted text, all at once. | ✅ | [done](06-text-done.png) |
| 08 | Theme toggle | One click: dark page, with a sun icon to switch back. The choice is remembered. | ✅ | [dark](08-dark-theme.png) |
| 09 | Phone 390 × 844, light | 1-hour talk picked from the history. No horizontal overflow. The full title fits next to the toggle and the status dot. Tiles in 2 columns, About this video in one column. | ✅ | [phone](09-phone-light.png) |
| 10 | Phone 390 × 844, dark | Same layout in the dark theme. No overflow. | ✅ | [phone dark](10-phone-dark.png) |
| 11 | Tablet 768 × 1024 | No overflow. Summary full width. The 7-part timeline on one line. About this video's three topics side by side. | ✅ | [tablet](11-tablet.png) |
| log | `runs/app.log` | Only `cache hit` for the two videos, and one 422 for the Vimeo link. No traceback, no 500. | ✅ | — |

The design inspector, in app-screen mode, passes the page in light and dark at 1440, 768 and 390 px with 0 FAIL and 0 WARN.

Timings were taken while an autoresearch training run shared the GPU.

### Running totals after the session

"This browser" tiles, in the dark screenshot:

| Tile | Value |
|---|---|
| Time saved | **1 h 24 min** (13 + 13 + 58) |
| Video covered | **1 h 28 min** (14:03 + 14:03 + 59:50) |
| Words read for you | **16,793** (condensed to 982) |
| Faster than watching | **25×** |

## Quality notes on the structured notes

- **The notes cover the whole video in order.** The 1-hour talk gets key points from all 7 parts, from LLM basics through tool use and multimodality to the security attacks at the end. The old one-pass summary covered mostly the opening.
- **The pasted article (06):** every number is right.
  - It gets 2028, 240 buses, $310 million with nearly half from grants, 60 %, $9 million a year, next spring and the busiest downtown routes.
  - It leaves out the first 40 buses, the drivers' training and the public meeting.
  - For a text this short the notes are about as long as the text (137 → 131 words), because the overview and the takeaway repeat the key points.
- **Errors to fix in the model or the prompt, not the page:**
  - 04 says "Llama 270b"; the captions say "Llama 2 70b". The v1 fine-tune made the same merge.
  - 04 states a Scale AI valuation of "around $2 trillion by 2025". That was not checked against the talk; treat it as doubtful.
  - 02 opens with "The text discusses how a writer…" and does not name the speaker.
- **Length:** the 1-hour talk becomes 37 key points, a 2-minute read. Complete, but long. A cap per part, or a short "top 5" above the full list, would help readers who skim.

## Earlier capture: v1 fine-tune, old page

The capture before this one is still in git (`git show 79f55fc --stat`), with its screenshots, video and summaries. It served the CNN/DM fine-tune `models/ytb-summarizer`, which scores ROUGE-1/2/Lsum 40.06 / 17.84 / 37.35 on 500 test articles. It used the old page: a model chip in the top bar and a "Base vs fine-tuned" panel. The base-model side of that comparison is [base-session.json](base-session.json).

What it showed, compared with base Qwen3-0.6B's one-paragraph summaries on the same inputs:

- **The fine-tune wrote like news highlights:** 3–4 short sentences, no number errors on the pasted article (base made 5 number errors), about twice as fast.
- **Its limits on spoken transcripts:** it summarized mainly the opening of the 1-hour talk, repeated a clause on the TED talk, and also wrote "Llama 270B".
- **Why the app now serves the base model:** the fine-tune ignores the structured-notes prompts and keeps writing short prose. The base model follows them.
