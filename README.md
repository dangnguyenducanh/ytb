# ytb — YouTube summarizer with a fine-tuned Qwen3-0.6B

Paste a YouTube link into a local web page and get structured notes written by `Qwen/Qwen3-0.6B`: an overview, key points in the order the video covers them, and a takeaway. The notes grow with the video (3–5 points per ~8 minutes of speech). The repo also fine-tunes the model on CNN/DailyMail, but that fine-tune only writes 3–4 sentences of news-style prose and ignores the structured prompts.

- **Ingestion:** uses the video's captions, preferring English. Videos with no captions are downloaded with `pytubefix` (no `yt-dlp`) and transcribed locally with `faster-whisper`. Transcripts are cached in `.cache/transcripts/`.
- **Long videos:** split into equal parts of up to 2,048 tokens. Each part gives 3–5 key points, all kept in order, and the overview and takeaway are written from those points. Prompts are in `src/ytb/prompt.py`; training and ROUGE keep the CNN/DM prompt.
- **Backend:** FastAPI. **Frontend:** one static page with no build step and no CDN. Both are served on `http://127.0.0.1:18500`.
- **Docs:** spec in `docs/superpowers/specs/2026-11-05-ytb-summarization-e2e.md`, plan in `docs/superpowers/plans/`.

## Setup

```bash
uv sync
uv run playwright install chromium   # only for the manual-session recorder
```

## Serve

```bash
uv run python -m ytb.api                                 # base Qwen3-0.6B: structured notes
uv run python -m ytb.api --model models/ytb-summarizer   # CNN/DM fine-tune: short prose, ignores the structure
```

Then open http://127.0.0.1:18500.

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-0.6B` | HF model folder or Hub id |
| `--device` | `cuda` if available | `cuda` or `cpu` for the summarizer |
| `--host` / `--port` | `127.0.0.1` / `18500` | bind address |
| `--cache-dir` | `.cache/transcripts` | transcript cache |
| `--whisper-model` | `small` | `base` is faster, `small` more accurate |
| `--whisper-device` | `cpu` | `cuda` is faster |

**API**
- `GET /api/health` returns `{status, model}`.
- `POST /api/summarize` takes `{"url": …}` or `{"text": …}` and returns `{summary, title, channel, duration, words_in, words_out, video_id, language, source, chunks, chunk_tokens, input_tokens, output_tokens, seconds, model}`. `summary` is plain text: `Overview: …`, `Key points:` with `- ` lines, `Takeaway: …`. `title` and `channel` come from YouTube's oEmbed endpoint (null when it cannot be reached); `duration` is the video's length in seconds from the captions or the audio (null for transcripts cached before it was recorded).
- The page follows the system's light or dark theme; the sun/moon button in the top bar switches it and remembers the choice, and `?theme=light|dark` previews one without saving it.
- Errors:
  - 422: bad input
  - 404: unavailable video, or no captions or audio
  - 413: over 150,000 characters, or a caption-less video over 60 minutes
  - 503: YouTube is rate-limiting this machine

## Manual session (screenshots + screen recording)

With the app running:

```bash
uv run python scripts/manual_session.py --out docs/manual-test/$(date +%F)-<label> [--no-captions-url <url>]
```

The script uses the page like a person: it types links, clicks, waits and reads. It saves a screenshot of every state, `session.webm`, and `session.json`. Review the captures and write a `REPORT.md` beside them. Past sessions are in `docs/manual-test/`.

## Train and evaluate

```bash
uv run python -m ytb.train --out runs/smoke-model --train-samples 64 --eval-samples 16 --max-steps 20   # smoke: loss falls, prints peak_gpu_gb
nohup uv run python -m ytb.train --out models/ytb-summarizer > runs/train.log 2>&1 &              # full run, about 40 min
nohup uv run python scripts/track_training.py --out docs/training/$(date +%F)-ytb-summarizer-progress.md > runs/track.log 2>&1 &   # live progress file
uv run python -m ytb.evaluate --model Qwen/Qwen3-0.6B --out runs/eval-base.json                     # base-model baseline
uv run python -m ytb.evaluate --model models/ytb-summarizer                                          # writes models/ytb-summarizer/eval.json
```

- **Data:** 20,000 CNN/DailyMail train articles, 1 epoch. About 30% of articles are rewritten as lowercase, punctuation-free "caption style". Highlights are cleaned into prose.
- **Training:** full fine-tune, fp32 weights with bf16 autocast, label smoothing 0.1, loss on summary tokens only, Adafactor, effective batch 32.
- **Evaluation:** ROUGE-1/2/Lsum on 500 test articles, next to lead-3. Acceptance: the fine-tuned model beats the base model on all three, with ROUGE-1 no more than 3 points below lead-3.

## Notes

- **YouTube rate limit:** YouTube blocks this host after a handful of quick transcript requests. The cache prevents repeat fetches; a block shows as a 503.
- **`av` is pinned below 19:** PyAV 19 removed an argument that `faster-whisper` 1.2.1 still passes.
- **Thinking off:** Qwen3's thinking mode is disabled everywhere through the shared prompt in `src/ytb/prompt.py`.
