# ytb — YouTube summarization with a fine-tuned Qwen3-0.6B (end-to-end)

> Spec written 2026-11-05 from the user's directives:
> - "https://huggingface.co/Qwen/Qwen3-0.6B use this model"
> - "today I need the full finetuned model with backend and frontend to serve that first"
> - "e2e design for ytb summarization with backend frontend simple with finetuned model … plan basic"
> - "Node 1: Ingestion (ingest_transcript) Extract the video ID from the URL and fetch the transcript. If youtube-transcript-api … add this"
> - "yt-dlp, remove this" (the audio fallback uses `pytubefix` instead)
> - "remove all the test there, split tasks into 2 phases … manual test: pretend like human to use this in frontend … just use base model, then training model … capture frontend and record screen during test for me store them in docs"
>
> It supersedes the larger designs in this folder for today's work. Those designs (knowledge graph, PPO/RLHF, production deployment) stay here as background for later.

## Problem Statement

I want to paste a YouTube link and get a short, readable summary of the video. The summary should come from **my own fine-tuned small model running on my own machine**, not from a cloud API or an off-the-shelf model. Today the `ytb` repo is empty. The earlier designs are far more than I need right now. I need the simplest thing that works end to end, today. I want to see it working as a person would use it, with screenshots and a screen recording I can look at later.

## Solution

One small Python project, built in two phases that share a single prompt definition.

**Phase 1 — the system (base model).**
- **Backend:** a local backend loads a model (the base `Qwen/Qwen3-0.6B` in this phase). Given a YouTube URL, it fetches the video's captions. When the video has none, it downloads the audio-only stream with `pytubefix` and transcribes it locally with `faster-whisper`. It then summarizes the transcript (long videos are split into chunks, summarized, then combined) and returns the summary with a few stats.
- **Frontend:** one web page, served by the same backend.
- **Manual check:** once the system is built, a browser is driven like a person using the page. Screenshots and a screen recording are captured and reviewed, problems are fixed, and the session is re-run until it looks right. The captures and a short report are stored in the repo's docs.

**Phase 2 — training.**
- **Fine-tune:** fully fine-tune `Qwen/Qwen3-0.6B` (all weights) on CNN/DailyMail, with part of the training text rewritten to look like raw YouTube auto-captions.
- **Evaluate:** score it with ROUGE against the base model and the lead-3 baseline, then export it as a self-contained model folder.
- **Switch over:** point the app at the fine-tuned model and repeat the captured human-like session.

The user flow: start the app, open the page, paste a URL, click **Summarize**, read the summary a few seconds later. Long or caption-less videos take tens of seconds to minutes.

## User Stories

1. As a viewer, I want to paste a YouTube URL and get a summary, so that I can decide whether the video is worth watching.
2. As a viewer, I want every common YouTube link form to work (watch links, `youtu.be` short links, Shorts, embeds, live links, mobile links, a bare video ID), so that I don't have to clean up links by hand.
3. As a viewer, I want a clear message when the link is not a YouTube video, so that I know to fix the input.
4. As a viewer, I want a clear message when the video is private, deleted or unavailable, so that I don't wait for nothing.
5. As a viewer, I want a video without captions transcribed locally from its audio, so that I can still summarize it.
6. As a viewer, I want a clear "YouTube is rate-limiting this machine, try again later" message when YouTube blocks requests, so that I know it isn't a bug.
7. As a viewer, I want English captions used when they exist, whether manual or auto-generated, so that the model gets the language it was trained on.
8. As a viewer, I want a video that only has captions in another language to still be summarized, so that I get something rather than an error.
9. As a viewer, I want long videos (podcasts, lectures) summarized as a whole, not just their first minutes, so that the summary reflects the full video.
10. As a viewer, I want caption noise like `[Music]` or `[Applause]` ignored, so that it doesn't pollute the summary.
11. As a viewer, I want the summary to be a few plain sentences, so that it reads quickly.
12. As a viewer, I want to see how long summarizing took and how much text was processed, so that I can judge speed and coverage.
13. As a viewer, I want to see which model produced the summary, so that I know whether I'm looking at the fine-tuned or the base model.
14. As a viewer, I want the Summarize button disabled with a visible progress indicator while it works, so that I don't submit twice.
15. As a viewer, I want to paste plain text instead of a URL, so that I can summarize a transcript or article I already have.
16. As a viewer, I want text that is empty or absurdly long rejected with a clear message, so that the app never hangs or crashes.
17. As a viewer, I want summarizing the same video again not to hit YouTube again, so that I don't get rate-limited.
18. As a viewer, I want the page to work without internet-loaded scripts or styles, so that the UI is fully local.
19. As a viewer, I want to see whether the summary came from captions or from local transcription, so that I understand differences in speed and quality.
20. As a viewer, I want a caption-less video longer than 60 minutes rejected before anything is downloaded, so that I don't wait half an hour.
21. As a viewer, I want a clear message when neither captions nor audio can be obtained, so that I know the video cannot be summarized.
22. As a viewer, I want a transcribed video cached like a captioned one, so that summarizing it again is instant.
23. As an operator, I want one command to start the app, bound to localhost on a known free port, so that running it is trivial and not exposed to the network by default.
24. As an operator, I want a health endpoint that reports the loaded model, so that I can check the app is up.
25. As an operator, I want the app to use the GPU when available and fall back to CPU, with local transcription on the CPU by default, so that it runs anywhere.
26. As the developer, I want the system to work with the base model first, so that the app is proven before any training time is spent.
27. As the developer, I want a browser session that uses the page like a person (typing links, clicking, waiting, reading results and errors), so that I see the real user experience.
28. As the developer, I want screenshots of every state and a screen recording of that session stored in the repo's docs, so that I can review the app without running it.
29. As the developer, I want a short report next to the captures listing what was tried, what was seen and what was fixed, so that I know the session was actually checked.
30. As the developer, I want problems found in the session fixed and the session re-run, so that the stored captures show a working app.
31. As an ML developer, I want one command that fine-tunes Qwen3-0.6B on CNN/DailyMail, so that I can produce the model today.
32. As an ML developer, I want a quick smoke run (a few steps on a few examples) that reports peak GPU memory before the real run, so that I catch crashes and memory problems in minutes.
33. As an ML developer, I want label-smoothed cross-entropy (0.1) computed only on the summary tokens, so that the model learns to summarize and not to repeat the prompt.
34. As an ML developer, I want articles truncated rather than summaries, and part of the training articles rewritten as lowercase, punctuation-free text, so that targets stay complete and the model copes with raw auto-captions.
35. As an ML developer, I want periodic validation loss and checkpoints, so that I can watch progress and resume after an interruption.
36. As an ML developer, I want one command that scores a model with ROUGE-1/2/Lsum on a fixed test sample next to lead-3, using exactly the app's prompt and generation settings, so that I can compare base and fine-tuned models fairly.
37. As an ML developer, I want the exported model folder to contain everything the app needs, so that serving it is just "point at the folder".
38. As the developer, I want the same captured human-like session repeated on the fine-tuned model, so that I can compare it with the base model side by side.

## Implementation Decisions

**Project shape**
- One Python project (Python 3.12, managed with uv) containing both phases. Phase 2 adds its training dependencies.
- The `ytb` folder becomes its own git repository with origin `github.com/dangnguyenducanh/ytb`. Today it is nested inside the home-directory repo, whose remote is unrelated.
- Model weights, checkpoints, the transcript cache and logs are git-ignored. Manual-session captures are committed under docs.
- There is no automated test suite (user decision). Verification is done by run-and-look steps and the captured manual sessions.

**Prompt module (shared by serving, evaluation and training)**
- The single source of truth for how text becomes model input. A fixed user instruction ("Summarize the following text in 3 to 4 sentences.") followed by the text goes through Qwen3's chat template with a generation prompt and **thinking disabled**.
- It also owns the generation settings: greedy decoding, at most 200 new tokens, no repeated 3-grams.

**Model loader**
- Loads any Hugging Face causal-LM folder or Hub ID: the base `Qwen/Qwen3-0.6B` in Phase 1, or the exported fine-tuned folder in Phase 2.
- Uses bf16 on CUDA when available, fp32 on CPU otherwise.
- Exposes a generate function (text → summary) and the tokenizer.

**Transcript module (Node 1, ingestion)**
- Parses any common YouTube link form or a bare 11-character ID into a video ID; anything else is an invalid URL.
- Fetches captions with `youtube-transcript-api`, preferring English (manual or generated) and otherwise taking the first listed transcript. It joins snippet texts and removes bracketed cues (`[Music]`).
- **Audio fallback** runs only when the video has no captions at all:
  - Read the duration first and refuse if it is longer than 60 minutes.
  - Download the audio-only stream with `pytubefix` into a temporary directory that is always deleted. There is no `yt-dlp` and no ffmpeg step.
  - Transcribe with `faster-whisper`: model `small` by default (`base` is faster), CPU int8 by default, CUDA opt-in, voice-activity filter on. Whisper's detected language becomes the transcript language.
- Every transcript records its source, captions or whisper. Each is cached on disk by video ID, because YouTube blocked this machine after about six quick requests during research.
- Maps failures to five domain errors:
  - invalid URL
  - video unavailable
  - no transcript (neither captions nor downloadable audio)
  - too long (a caption-less video over 60 minutes)
  - YouTube blocked (IP/request blocked on captions; bot detection or a proof-of-origin token demand on audio)

**Summarizer module**
- Text within the 1,024-token input budget is summarized in one pass.
- Longer text is split into token windows of 1,024 with 100-token overlap. Each window is summarized, the window summaries are joined, and the result is summarized again. If it still doesn't fit it repeats, up to 3 rounds, then truncates. This bound guarantees termination.
- Returns the summary, number of first-round chunks, input tokens, output tokens and elapsed seconds. Model calls are serialized, one request at a time.

**Backend HTTP API (FastAPI)**
- `GET /` serves the frontend page.
- `GET /api/health` returns `{status, model}`.
- `POST /api/summarize` takes JSON with exactly one of `url` or `text`. It returns `{summary, video_id, language, source, chunks, input_tokens, output_tokens, seconds, model}`. `video_id`, `language` and `source` are null for text input.
- Error contract:
  - 422 for bad input: both or neither field, an empty field, or an invalid URL.
  - 404 for an unavailable video or no transcript.
  - 413 for text or a transcript over 150,000 characters, or a caption-less video over 60 minutes.
  - 503 when YouTube is blocking.
  - Expected failures never produce a 500.
- Started with one command taking:
  - a model path or ID
  - host (default `127.0.0.1`)
  - port (default `18500`)
  - Whisper model size (default `small`)
  - Whisper device (default `cpu`)

**Frontend**
- One static HTML page with inline CSS and vanilla JavaScript, served by the backend: no build step, no CDN.
- It has a URL field and a "paste text instead" textarea, plus a Summarize button that is disabled while a request runs. While waiting it shows an elapsed-time indicator, and notes that caption-less videos are transcribed locally and can take minutes.
- The result card shows the summary, language, source, chunk count, token counts, seconds and model. Each error status maps to a plain-language message. The page header shows the loaded model.

**Manual human-like session with captures**
- A small driver uses Playwright with headless Chromium to use the page like a person:
  - types links character by character
  - pauses between actions
  - clicks Summarize
  - waits for the result or error the way a person would
- It records a video of the whole session and saves a screenshot of every meaningful state: empty page, waiting, result, error.
- Scenarios:
  - the empty page
  - a captioned talk
  - the same talk again as a `youtu.be` link (it must come back from cache)
  - a one-hour talk (several chunks)
  - a non-YouTube link (an error)
  - pasted plain text
  - a caption-less video, when one can be found
- After each run, the executor reviews every screenshot and the server log against a checklist. Any problem gets fixed and the session is re-run, up to 3 rounds.
- Captures, a JSON log of what each scenario showed, and a short report are stored in a per-session folder under docs: one for the base model in Phase 1, one for the fine-tuned model in Phase 2.

**Training data, trainer, evaluation (Phase 2)**
- **Data:**
  - CNN/DailyMail 3.0.0 from the Hub's parquet files: `article` is the input, `highlights` the target (newlines become spaces). Seeded shuffle; default 20,000 train / 500 validation.
  - Each example is the prompt tokens (labels −100) followed by the summary tokens plus Qwen's end-of-turn token. The article is truncated to 1,024 tokens, keeping the beginning; the summary to 256 tokens.
  - About 30% of training articles, chosen deterministically, are lowercased and stripped of Unicode punctuation.
- **Trainer:**
  - Hugging Face `Trainer` with full-parameter fine-tuning: fp32 weights with bf16 autocast, gradient checkpointing, `label_smoothing_factor=0.1`.
  - Learning rate 1e-5, cosine schedule with 3% warmup, effective batch 32, 1 epoch. It evaluates and saves every 200 steps.
  - A 20-step smoke run checks the loss and reports peak GPU memory before the real run.
- **Evaluation:**
  - ROUGE-1/2/Lsum ×100 with stemming and NLTK sentence splitting for Lsum, on 500 fixed test articles, next to lead-3.
  - Acceptance: the fine-tuned model beats the base model zero-shot on all three metrics, with ROUGE-1 no more than 3 points below lead-3.
- **Export:** the final model, tokenizer, training config and eval report go into one folder that the app loads.

## Testing Decisions

- **No automated tests** (user decision: "remove all the test there"). There is no pytest suite, no test files and no test dependencies.
- **Run-and-look checks** at the end of each build task, using real commands with an expected output:
  - a one-line summary from the base model
  - one real transcript fetch plus a cache hit
  - `curl` against the running API, including an error case
  - a print-out of one training example to confirm prompt masking and prompt parity
  - a smoke training run reporting loss and peak memory
- **Captured manual sessions** are the acceptance check for the app. They run in Phase 1 on the base model and again in Phase 2 on the fine-tuned model, as described above. The checklist review and the fix-and-rerun loop are part of the task, not optional.
- **Model quality** is checked by the ROUGE evaluation and by reading the summaries in the captured sessions.
- **Prior art:** none. The repo is greenfield.

## Out of Scope

- An automated test suite.
- PPO/RLHF and reward models: the next step after this works.
- Knowledge-graph extraction and rendering (LangGraph, PyVis, Mermaid, Neo4j). Only Node 1 (ingestion with the Whisper fallback) of that design is included.
- Using the audio fallback for videos that do have captions in some language.
- Production deployment: Docker, Celery/Redis queue, Kubernetes/KEDA, Prometheus/Grafana, Phoenix/Langfuse, vLLM/TGI, quantization.
- Streaming responses, authentication and multiple users, summary history, timestamps in summaries, and quality guarantees for non-English videos.

## Further Notes

- **Domain gap.** The model trains on news and runs on spoken transcripts. Caption-style augmentation narrows the gap; the captured sessions show the real quality.
- **Local transcription.** Whisper `small` on CPU int8 is roughly 2–5× faster than real time. Audio comes from `pytubefix` (11.2.1, API checked 2026-11-05) because of the directive against `yt-dlp`. A proof-of-origin token demand from YouTube surfaces as the "blocked" error.
- **Playwright.** It needs a one-time Chromium download (~150 MB). If Chromium cannot start because system libraries are missing, installing them needs sudo, so that is the user's call.
- **Thinking off everywhere.** Qwen3's thinking mode is disabled in serving, evaluation and training.
- **Licences.** Qwen3 is Apache-2.0; CNN/DailyMail is intended for research use.
- **Issue tracker.** None is configured for this repo, so this spec lives here rather than in an issue.
