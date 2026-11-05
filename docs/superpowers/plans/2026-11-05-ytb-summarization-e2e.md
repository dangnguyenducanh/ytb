# ytb YouTube Summarizer (Qwen3-0.6B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Paste a YouTube URL into a local web page and get a summary from Qwen3-0.6B: the base model in Phase 1, the fully fine-tuned model in Phase 2. Each phase ends with a captured, human-like browser session stored in `docs/`.

**Architecture:** One uv Python package `ytb`.
- **Phase 1:** a FastAPI app (`api`) loads a model (`model`), ingests transcripts (`transcript`: captions, or the `audio` pytubefix + faster-whisper fallback), summarizes with chunk-then-combine for long videos (`summarize`), and serves one static page. A Playwright driver (`scripts/manual_session.py`) uses the page like a person and captures screenshots plus a video.
- **Phase 2:** `data` → `train` → `evaluate` fine-tunes `Qwen/Qwen3-0.6B` on CNN/DailyMail. The app is then pointed at the result.

**Tech Stack:** Python 3.12, uv, torch, transformers, FastAPI, uvicorn, youtube-transcript-api, pytubefix, faster-whisper, Playwright (Chromium); Phase 2 adds datasets, rouge-score, nltk, tensorboard.

**Spec:** `docs/superpowers/specs/2026-11-05-ytb-summarization-e2e.md`

## Global Constraints

- **No automated tests** (user directive). Do not add pytest, test files or test dependencies. Verify with the run-and-look steps and the captured sessions.
- **Repo root and git:** the repo root is `/home/aiface/dangnguyenducanh/ytb`. Until Task 1 runs, `git` there resolves to the parent repo `/home/aiface`, whose remote is unrelated, so never commit before Task 1. Never `git push` without asking the user.
- **Model and thinking mode:** Phase 1 uses the base `Qwen/Qwen3-0.6B` only. Thinking is disabled everywhere (`enable_thinking=False`).
- **Shared GPU:**
  - Training peak memory must stay at or below **13 GB**, since other services hold 14–31 GB of the 49 GB card.
  - The app uses about 1.5 GB.
  - Whisper runs on the CPU by default.
  - Never kill other users' GPU processes.
- **Port and bind address:** the app port is **18500** and it binds to `127.0.0.1`.
- **No `yt-dlp`** (user directive): audio for the Whisper fallback comes from `pytubefix` (11.2.1; the API used in Task 3 was checked against it).
- **YouTube rate limit:** YouTube blocked this host after about six quick transcript requests, so never loop over YouTube calls. The transcript cache exists for this reason.
- **Ignored paths:** `models/`, `runs/` and `.cache/` are git-ignored. `docs/manual-test/` is committed.

## Review Focus

1. **Prompt drift** between serving (Phase 1) and training (Phase 2) would silently ruin the fine-tuned model. Task 7's print-out must show the decoded prompt equal to `build_prompt(...)`.
2. **Re-fetching from YouTube** gets the host blocked. Scenario `03-cached` in Task 6 must come back in seconds, with no new fetch in the app log.
3. **Long transcripts** must finish. Scenario `04-long` must end with `chunks` > 1; the 3-round cap in `summarize` guarantees termination even if the model doesn't shrink text.
4. **Caption-less videos** must use the Whisper fallback once and then be cached. Task 3 Step 3 exercises it directly; scenario `07-no-captions` exercises it through the page when such a video is found.
5. **Label smoothing on a causal LM with masked prompt labels:** Task 9's smoke run must show a finite starting loss in the plausible 1–5 range that falls over 20 steps.

---

## Phase 1 — The system: backend + frontend on the base model

### Task 1: Repo bootstrap and project skeleton

**Files:**
- Create: `.gitignore`, `pyproject.toml` (via `uv init`), `src/ytb/__init__.py`

**Interfaces:**
- Produces: importable package `ytb`; Chromium available to Playwright.

- [ ] **Step 1: Make `ytb` its own git repo**

```bash
cd /home/aiface/dangnguyenducanh/ytb
git init -b main
git remote add origin https://github.com/dangnguyenducanh/ytb.git
git rev-parse --show-toplevel   # must print /home/aiface/dangnguyenducanh/ytb
```

- [ ] **Step 2: Create the uv package and add Phase 1 dependencies**

```bash
uv init --package --name ytb --python 3.12 --vcs none .
uv add torch transformers accelerate fastapi "uvicorn[standard]" youtube-transcript-api pytubefix faster-whisper
uv add --dev playwright
uv run playwright install chromium
```

- [ ] **Step 3: Write `.gitignore`**

Lines: `.venv/`, `__pycache__/`, `*.egg-info/`, `models/`, `runs/`, `.cache/`, `.claude/`.

- [ ] **Step 4: Verify the environment**

Run: `uv run python -c "import torch, faster_whisper, pytubefix; print(torch.__version__, torch.cuda.is_available())"`
Expected: a version and `True`.

Run: `uv run python -c "from playwright.sync_api import sync_playwright as s; p=s().start(); b=p.chromium.launch(); print(b.version); b.close(); p.stop()"`
Expected: a Chromium version.

If Chromium fails to start because system libraries are missing, stop and tell the user. The fix (`uv run playwright install-deps chromium`) needs sudo.

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml uv.lock .python-version README.md src docs
git commit -m "chore: bootstrap ytb package and docs"
```

---

### Task 2: Shared prompt and model loader

**Files:**
- Create: `src/ytb/prompt.py`, `src/ytb/model.py`

**Interfaces:**
- Produces in `ytb.prompt`:
  - `INSTRUCTION = "Summarize the following text in 3 to 4 sentences."`
  - `GENERATION = {"do_sample": False, "max_new_tokens": 200, "no_repeat_ngram_size": 3}`
  - `build_prompt(tokenizer, text: str) -> str`
- Produces in `ytb.model`: `load(model_id: str) -> tuple[Callable[[str], str], PreTrainedTokenizerBase]`, where the callable maps text to summary.

- [ ] **Step 1: Implement `ytb/prompt.py`**

`build_prompt` = `tokenizer.apply_chat_template([{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)`.

- [ ] **Step 2: Implement `ytb/model.py`**

- **`load`:** `AutoTokenizer` + `AutoModelForCausalLM`, bf16 on CUDA, fp32 on CPU, `model.eval()`.
- **`generate(text)`:**
  - Build the prompt with `build_prompt`.
  - Call `model.generate(**inputs, **GENERATION, pad_token_id=tok.pad_token_id)` under `torch.inference_mode()`.
  - Decode only the new tokens with `skip_special_tokens=True`, then `.strip()`.

- [ ] **Step 3: Run-and-look check**

Run: `uv run python -c "from ytb.model import load; g,_ = load('Qwen/Qwen3-0.6B'); print(g('The city council voted on Tuesday to build a new bridge over the river. Construction starts next spring and should take two years.'))"`
Expected: a short English summary of the bridge text, with no `<think>` in it. The first run downloads the model (~1.5 GB).

- [ ] **Step 4: Commit**

```bash
git add src/ytb/prompt.py src/ytb/model.py
git commit -m "feat: shared prompt and model loader"
```

---

### Task 3: Transcript ingestion — captions, Whisper fallback, cache

**Files:**
- Create: `src/ytb/transcript.py`, `src/ytb/audio.py`

**Interfaces:**
- Produces in `ytb.transcript`:
  - Exceptions `TranscriptError(Exception)` and its subclasses `InvalidUrl`, `VideoUnavailable`, `NoTranscript`, `TooLong`, `YouTubeBlocked`
  - `@dataclass Transcript(video_id: str, language: str, text: str, source: str)`, where `source` is `"captions"` or `"whisper"`
  - `parse_video_id(url: str) -> str`, which raises `InvalidUrl`
  - `fetch_transcript(video_id: str, cache_dir: Path, fallback: Callable[[str], tuple[str, str]] | None = None) -> Transcript`. `fallback(video_id) -> (language, text)` is called only when the video has no captions.
- Produces in `ytb.audio`: `transcribe_audio(video_id: str, model_size: str = "small", device: str = "cpu", max_seconds: int = 3600) -> tuple[str, str]`, which raises `TooLong`, `NoTranscript`, `VideoUnavailable` or `YouTubeBlocked`.

- [ ] **Step 1: Implement `ytb/transcript.py`**

- **`parse_video_id`:**
  - Accept a bare 11-character `[A-Za-z0-9_-]{11}`.
  - Accept hosts `youtube.com`, `www.youtube.com`, `m.youtube.com` with `/watch?v=ID`, `/shorts/ID`, `/embed/ID` or `/live/ID`.
  - Accept `youtu.be/ID`.
  - Ignore extra query params.
  - Raise `InvalidUrl` for anything else.
- **Cache:** the file is `cache_dir / f"{video_id}.json"`, holding all four `Transcript` fields. On a hit, return it without any network call. Log `cache hit <id>` or `fetch <id>` with `logging.info`; the Task 6 review reads these lines.
- **Captions path:**
  - `YouTubeTranscriptApi().fetch(video_id, languages=["en"])`. On `NoTranscriptFound`, take the first entry of `.list(video_id)` and `.fetch()` it.
  - Join the snippet texts, remove `\[[^\]]*\]`, collapse whitespace. Set `source="captions"`.
- **Whisper path:** on `TranscriptsDisabled`, or `NoTranscriptFound` with an empty list, call `fallback`. Use `source="whisper"`, or raise `NoTranscript` when there is no fallback.
- **Error mapping:**
  - `IpBlocked`, `RequestBlocked` → `YouTubeBlocked` (never fall back)
  - `VideoUnavailable` → `VideoUnavailable`
  - any other `CouldNotRetrieveTranscript`, or empty text → `NoTranscript`

- [ ] **Step 2: Implement `ytb/audio.py`**

- **Metadata first:** `yt = pytubefix.YouTube(f"https://www.youtube.com/watch?v={video_id}")`. If `yt.length > max_seconds`, raise `TooLong` before downloading anything.
- **Download:** inside a `tempfile.TemporaryDirectory()`, `stream = yt.streams.get_audio_only()`. If it is `None`, raise `NoTranscript`. Otherwise `path = stream.download(output_path=tmp, filename=f"{video_id}.m4a")`.
- **Whisper model:** cache it with `@functools.lru_cache(maxsize=2) def _whisper(size, device)` returning `WhisperModel(size, device=device, compute_type="int8" if device == "cpu" else "float16")`.
- **Transcribe:** `segments, info = model.transcribe(path, vad_filter=True)`. Return `(info.language, " ".join(s.text.strip() for s in segments))`.
- **Errors from `pytubefix.exceptions`,** caught in this order:
  1. `BotDetection`, `PoTokenRequired` → `YouTubeBlocked`
  2. `VideoUnavailable` and its subclasses → `VideoUnavailable`
  3. any other `PytubeFixError` → `NoTranscript`

- [ ] **Step 3: Run-and-look checks** (four YouTube requests in total; do not repeat them in a loop)

```bash
uv run python -c "
import time; from pathlib import Path
from ytb.transcript import fetch_transcript, TranscriptError
d = Path('runs/cache-check')
t = time.time(); a = fetch_transcript('arj7oStGLkU', d); print(a.language, a.source, len(a.text), round(time.time()-t, 1))
t = time.time(); b = fetch_transcript('arj7oStGLkU', d); print('cached', a == b, round(time.time()-t, 3))
try: fetch_transcript('xxxxxxxxxxx', d)
except TranscriptError as e: print('error', type(e).__name__)
"
uv run python -c "from ytb.audio import transcribe_audio; print(transcribe_audio('jNQXAC9IVRw', model_size='base'))"
```

Expected:
1. `en captions <several thousand> <a few seconds>`
2. `cached True` with a time near `0.0`
3. `error VideoUnavailable`
4. `('en', '... elephants ...')`. That is "Me at the zoo", 19 s, transcribed on the CPU.

- [ ] **Step 4: Commit**

```bash
git add src/ytb/transcript.py src/ytb/audio.py
git commit -m "feat: transcript ingestion with captions, whisper fallback and cache"
```

---

### Task 4: Summarizer (chunk then combine) and HTTP API

**Files:**
- Create: `src/ytb/summarize.py`, `src/ytb/api.py`

**Interfaces:**
- Consumes: `load` (Task 2); `Transcript`, `parse_video_id`, `fetch_transcript`, the error classes and `transcribe_audio` (Task 3).
- Produces:
  - `@dataclass Summary(text: str, chunks: int, input_tokens: int, output_tokens: int, seconds: float)`.
  - `Summarizer(generate: Callable[[str], str], tokenizer, max_input_tokens: int = 1024, overlap_tokens: int = 100, max_rounds: int = 3)` with `.summarize(text: str) -> Summary`.
  - `MAX_CHARS = 150_000`.
  - `create_app(summarizer: Summarizer, fetch: Callable[[str], Transcript], model_name: str) -> FastAPI`.
  - `main(argv: list[str] | None = None) -> None`, run as `uv run python -m ytb.api`, with flags `--model` (default `models/ytb-summarizer`), `--host 127.0.0.1`, `--port 18500`, `--cache-dir .cache/transcripts`, `--whisper-model small`, `--whisper-device cpu`.

- [ ] **Step 1: Implement `ytb/summarize.py`**

- `summarize` holds a `threading.Lock` for the whole call. Token counting and windowing use `tokenizer.encode(text, add_special_tokens=False)` and `tokenizer.decode(ids, skip_special_tokens=True)`.
- **Single pass:** if the text has ≤ `max_input_tokens` tokens, make one `generate` call with `chunks=1`.
- **Windowing:** otherwise build windows starting at `0, step, 2·step, …` with `step = max_input_tokens − overlap_tokens`, stopping after the first window that reaches the end.
- **Combining:** summarize each window and join the results with `"\n"`. Repeat on the joined text until it fits, for at most `max_rounds` rounds. Then truncate to `max_input_tokens` and make one final `generate` call.
- **Stats:**
  - `chunks` = first-round window count
  - `input_tokens` = tokens of the original text
  - `output_tokens` = tokens of the final summary
  - `seconds` = `time.perf_counter()` delta

- [ ] **Step 2: Implement `ytb/api.py`**

- **Request model:** `SummarizeRequest(BaseModel)` with `url: str | None = None` and `text: str | None = None`.
- **Input checks:** exactly one field must be non-blank, otherwise `HTTPException(422)`. `InvalidUrl` → 422. Text or transcript longer than `MAX_CHARS` → 413.
- **Error statuses:** map `VideoUnavailable`/`NoTranscript` → 404, `TooLong` → 413 and `YouTubeBlocked` → 503, each with a plain-language `detail`. Expected failures never return a 500.
- **Response:** `{summary, video_id, language, source, chunks, input_tokens, output_tokens, seconds, model}`. `video_id`, `language` and `source` are `None` for text input.
- **Sync endpoints:** endpoints are plain `def`, so FastAPI runs them in its threadpool.
- **`main`:**
  - `logging.basicConfig(level=logging.INFO)`.
  - `generate, tok = load(args.model)`.
  - `fallback = functools.partial(transcribe_audio, model_size=args.whisper_model, device=args.whisper_device)`.
  - `fetch = functools.partial(fetch_transcript, cache_dir=Path(args.cache_dir), fallback=fallback)`.
  - `uvicorn.run(create_app(Summarizer(generate, tok), fetch, args.model), host=args.host, port=args.port)`.
  - Add `if __name__ == "__main__": main()`.

- [ ] **Step 3: Run-and-look checks with curl**

```bash
mkdir -p runs && nohup uv run python -m ytb.api --model Qwen/Qwen3-0.6B > runs/app.log 2>&1 &
until curl -sf http://127.0.0.1:18500/api/health; do sleep 2; done; echo
curl -s -H 'content-type: application/json' -d '{"text":"The city council voted on Tuesday to build a new bridge over the river. Construction starts next spring and should take two years."}' http://127.0.0.1:18500/api/summarize; echo
curl -s -o /dev/null -w '%{http_code}\n' -H 'content-type: application/json' -d '{"url":"https://vimeo.com/123"}' http://127.0.0.1:18500/api/summarize
curl -s -o /dev/null -w '%{http_code}\n' -H 'content-type: application/json' -d '{"url":"arj7oStGLkU","text":"x"}' http://127.0.0.1:18500/api/summarize
python3 -c 'import json; print(json.dumps({"text": "a" * 150001}))' | curl -s -o /dev/null -w '%{http_code}\n' -H 'content-type: application/json' --data @- http://127.0.0.1:18500/api/summarize
```

Expected, in order:
1. `{"status":"ok","model":"Qwen/Qwen3-0.6B"}`
2. JSON with a summary, `"chunks":1` and `"source":null`
3. `422`
4. `422`
5. `413`

`runs/app.log` should have no tracebacks. Leave the server running for Task 5.

- [ ] **Step 4: Commit**

```bash
git add src/ytb/summarize.py src/ytb/api.py
git commit -m "feat: chunk-then-combine summarizer and HTTP API"
```

---

### Task 5: Frontend page

**Files:**
- Create: `src/ytb/static/index.html`
- Modify: `src/ytb/api.py` (add `GET /`)

**Interfaces:**
- Consumes: `POST /api/summarize` and `GET /api/health` (Task 4).
- Produces: `GET /` serves the page. Element ids that the Task 6 driver relies on:
  - `model` (header model name)
  - `url`
  - `paste` (a `<details>` whose `<summary>` opens the textarea)
  - `text`
  - `go`
  - `status`
  - `result` and `error`, both using the `hidden` attribute when not shown
  - `summary`
  - `stats`

- [ ] **Step 1: Implement the page and route**

- **Route:** `GET /` returns `FileResponse(Path(__file__).parent / "static" / "index.html")`.
- **Page:** inline CSS and vanilla JS in one file, with no external URLs.
- **On load:** `GET /api/health` and put the model name in `#model`.
- **On submit:**
  - Hide `#result` and `#error`.
  - Disable `#go`.
  - Show an elapsed-seconds counter in `#status`, with the hint "Videos without captions are transcribed locally — this can take a few minutes".
  - POST `{url}` or `{text}`, whichever is filled.
- **On 200:** fill `#summary`, and fill `#stats` with language, source ("captions" / "local transcription" / "—"), chunks, input/output tokens, seconds and model. Unhide `#result`.
- **On error:** put `detail` in `#error`, prefixed per status, and unhide it:
  - 422 → "Check the link or text"
  - 404 → "No captions or audio for this video"
  - 413 → "Too long (max 150,000 characters, or 60 minutes for videos without captions)"
  - 503 → "YouTube is rate-limiting this machine, try again later"
- **Always:** re-enable `#go` and stop the counter.

- [ ] **Step 2: Run-and-look check**

Restart the server (same command as Task 4 Step 3), then run:
- `curl -s http://127.0.0.1:18500/ | grep -c 'id="go"'` → expected `1`
- `curl -s http://127.0.0.1:18500/ | grep -c -E 'https?://'` → expected `0`

- [ ] **Step 3: Commit**

```bash
git add src/ytb/static/index.html src/ytb/api.py
git commit -m "feat: single-page frontend served by the backend"
```

---

### Task 6: Manual human-like session with captures (base model) — verify, fix, store

**Files:**
- Create: `scripts/manual_session.py`
- Create: `docs/manual-test/<YYYY-MM-DD>-base-model/` containing `NN-*.png`, `session.webm`, `session.json`, `REPORT.md`

**Interfaces:**
- Consumes: the running app (Tasks 4–5) and the element ids from Task 5.
- Produces: CLI `uv run python scripts/manual_session.py --base-url http://127.0.0.1:18500 --out DIR [--no-captions-url URL]`. It writes screenshots, `session.webm` and `session.json`. Each `session.json` entry is `{scenario, input, client_seconds, summary, stats, error}`.

- [ ] **Step 1: Implement `scripts/manual_session.py`** (Playwright sync API, behaving like a person)

- **Browser:** `chromium.launch()`, then `browser.new_context(viewport={"width": 1280, "height": 860}, record_video_dir=str(out / "_video"), record_video_size={"width": 1280, "height": 860})`, and one page.
- **Each scenario:**
  1. `page.goto(base_url)` and wait for `#model` to have text.
  2. Type the URL with `page.locator("#url").press_sequentially(url, delay=60)`. For text, click `#paste summary` and `fill` the text, as if pasted.
  3. Pause 600 ms and screenshot `NN-<name>-input.png`.
  4. Click `#go`, wait 1.5 s and screenshot `NN-<name>-waiting.png`.
  5. Wait for `#result:not([hidden]), #error:not([hidden])` (timeout 15 min) and screenshot `NN-<name>-done.png` with `full_page=True`.
  6. Read the texts of `#summary`, `#stats` and `#error` into the log, then pause 2 s ("reading").
- **Scenarios, in order:**
  - `01-home`: screenshot only
  - `02-captions`: `https://www.youtube.com/watch?v=arj7oStGLkU` (TED talk with English captions, verified)
  - `03-cached`: `https://youtu.be/arj7oStGLkU`
  - `04-long`: `https://www.youtube.com/watch?v=zjkBMFhNj_g`, a ~1-hour captioned talk. If it returns 404, swap in any 45–60 min captioned talk and note that in the report.
  - `05-invalid`: `https://vimeo.com/123`
  - `06-text`: a ~150-word news paragraph you write into the script
  - `07-no-captions`: only when `--no-captions-url` is given
- **Finish:** close the context so the video is finalized, move the video to `out / "session.webm"`, delete `_video/`, and write `session.json`.

- [ ] **Step 2: Find a caption-less video (≤ 10 minutes of effort, at most 5 YouTube checks)**

Check a candidate with `uv run python -c "from youtube_transcript_api import YouTubeTranscriptApi as Y; print(list(Y().list('<ID>')))"`. A `TranscriptsDisabled` error means it qualifies. If none is found, run without `--no-captions-url`; Task 3 Step 3 already covered the fallback, so record that in the report.

- [ ] **Step 3: Run the session against the base model**

Make sure the app from Task 4 is running with `--model Qwen/Qwen3-0.6B` and logging to `runs/app.log`. Then:

```bash
OUT=docs/manual-test/$(date +%F)-base-model
uv run python scripts/manual_session.py --base-url http://127.0.0.1:18500 --out $OUT [--no-captions-url <url>]
```

- [ ] **Step 4: Review every screenshot and the app log against this checklist**

Open each PNG and check `runs/app.log`.

| # | Must be true |
|---|---|
| 01 | Header shows `Qwen/Qwen3-0.6B`; URL field and button visible; nothing overlaps or overflows |
| 02 | `waiting` shot: button disabled and timer running; `done` shot: 3–4 English sentences about the talk; stats show `en`, `captions`, seconds |
| 03 | App log shows `cache hit arj7oStGLkU` and no second `fetch arj7oStGLkU`. Only the transcript is cached, so the summary is regenerated and the time is similar to 02's. |
| 04 | Finishes; `chunks` > 1; the summary is about the talk as a whole |
| 05 | Plain-language "Check the link or text" message; no raw JSON or stack trace; button enabled again |
| 06 | Summary of the pasted paragraph; source shows `—` |
| 07 | (if run) source shows `local transcription` |
| log | No `Traceback` and no ` 500 ` in `runs/app.log` |

Summary *quality* from the base model may be mediocre. Note it, but it is not a failure; the checklist judges the app.

- [ ] **Step 5: Fix and re-run until the checklist passes (at most 3 rounds)**

For each failed item:
1. Fix the code.
2. Restart the app.
3. Delete `$OUT`.
4. Re-run Step 3 and re-review.

`03` re-uses the cache, so re-runs cost at most one YouTube request (scenario 04's first fetch is cached too). If items still fail after 3 rounds, stop and report them to the user with the screenshots.

- [ ] **Step 6: Write `$OUT/REPORT.md`**

- **Header:** date, model, `git rev-parse --short HEAD`, app command.
- **Scenario table:** scenario | expected | observed | ok? | screenshot link.
- **Issues found and fixed:** one line per round.
- **Summary-quality notes:** quote two summaries.
- **Recording:** a link to `session.webm`.

- [ ] **Step 7: Commit**

```bash
git add scripts/manual_session.py docs/manual-test src
git commit -m "docs: base-model session captures (screenshots, recording, report)"
```

---

## Phase 2 — Training the model

### Task 7: Training data

**Files:**
- Create: `src/ytb/data.py`

**Interfaces:**
- Consumes: `build_prompt` (Task 2).
- Produces in `ytb.data`:
  - Constants `MAX_INPUT_TOKENS = 1024`, `MAX_TARGET_TOKENS = 256`, `CAPTION_FRACTION = 0.3`
  - `to_caption_style(text: str) -> str`
  - `build_example(tokenizer, article: str, summary: str) -> dict[str, list[int]]`, returning `input_ids`, `attention_mask`, `labels`
  - `load_cnndm(split: str, n: int, seed: int = 0) -> datasets.Dataset`, with columns `article` and `highlights`
  - `prepare(tokenizer, ds: datasets.Dataset, augment: bool, seed: int = 0) -> datasets.Dataset`, holding tokenized examples only

- [ ] **Step 1: Add the training dependencies**

Run: `uv add datasets rouge-score nltk tensorboard && uv run python -c "import nltk; nltk.download('punkt_tab')"`

- [ ] **Step 2: Implement `ytb/data.py`**

- **`build_example`:**
  - Truncate the article to its first `MAX_INPUT_TOKENS` tokens (encode, slice, decode).
  - Prompt ids = `tokenizer(build_prompt(tokenizer, article), add_special_tokens=False).input_ids`.
  - Target ids = the summary's first `MAX_TARGET_TOKENS` tokens plus the `<|im_end|>` id.
  - Labels = `[-100] * len(prompt) + target`.
- **`to_caption_style`:** casefold, drop every character whose `unicodedata.category` starts with `"P"`, collapse whitespace.
- **`load_cnndm`:**
  - Load with `load_dataset("abisee/cnn_dailymail", data_files={split: f"3.0.0/{split}-*.parquet"}, split=split)`. This form is verified working on this host.
  - Then `.shuffle(seed=seed).select(range(n))`.
  - Replace newlines in `highlights` with spaces.
- **`prepare`:** with `augment=True`, row `i` is caption-styled when `random.Random(seed * 1_000_003 + i).random() < CAPTION_FRACTION`. Drop the original columns.

- [ ] **Step 3: Run-and-look check (prompt masking and prompt parity)**

```bash
uv run python -c "
from transformers import AutoTokenizer
from ytb.data import build_example, to_caption_style
from ytb.prompt import build_prompt
t = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
ex = build_example(t, 'The cat sat on the mat.', 'A cat sat.')
n = ex['labels'].count(-100)
print(repr(t.decode(ex['input_ids'][n:])))
print(t.decode(ex['input_ids'][:n]) == build_prompt(t, 'The cat sat on the mat.'))
print(to_caption_style(\"Hello, World! It's 5 p.m. — Việt Nam.\"))
"
```

Expected, in order:
1. `'A cat sat.<|im_end|>'`
2. `True`
3. `hello world its 5 pm việt nam`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/ytb/data.py
git commit -m "feat: CNN/DailyMail training examples with prompt masking and caption-style augmentation"
```

---

### Task 8: ROUGE evaluation and base-model baseline

**Files:**
- Create: `src/ytb/rouge.py`, `src/ytb/evaluate.py`

**Interfaces:**
- Consumes: `load` (Task 2), `load_cnndm` (Task 7).
- Produces:
  - `rouge(preds: list[str], refs: list[str]) -> dict[str, float]` with keys `rouge1`, `rouge2`, `rougeLsum` (F × 100, 2 decimals)
  - `lead3(article: str) -> str`
  - CLI `uv run python -m ytb.evaluate --model ID_OR_DIR [--samples 500] [--out PATH]`, which writes `{"model", "samples", "lead3": {...}, "scores": {...}}`. `--out` defaults to `<model dir>/eval.json` when `--model` is a directory.

- [ ] **Step 1: Implement `ytb/rouge.py` and `ytb/evaluate.py`**

- **Scorer:** `rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)`. Format texts as `"\n".join(nltk.sent_tokenize(t))` and call `scorer.score(target, prediction)`.
- **`lead3`:** the first 3 `sent_tokenize` sentences.
- **`evaluate`:** sample = `load_cnndm("test", samples, seed=0)`; one `generate` per article, in sequence. `# ponytail: sequential generate (~10-15 min/500); batch with left padding if eval speed matters`.

- [ ] **Step 2: Run-and-look check**

Run: `uv run python -c "from ytb.rouge import rouge; print(rouge(['The cat sat. It slept.'], ['The cat sat. It slept.']), rouge([''], ['The cat sat.']))"`
Expected: all three scores `100.0`, then all `0.0`.

- [ ] **Step 3: Baseline evaluation of the base model**

Run: `uv run python -m ytb.evaluate --model Qwen/Qwen3-0.6B --out runs/eval-base.json`
Expected: prints `lead3` and `scores` rows, with lead-3 ROUGE-1 around 40, and writes `runs/eval-base.json`.

- [ ] **Step 4: Commit**

```bash
git add src/ytb/rouge.py src/ytb/evaluate.py
git commit -m "feat: ROUGE evaluation CLI with lead-3 baseline"
```

---

### Task 9: Fine-tuning — smoke run, then full run

**Files:**
- Create: `src/ytb/train.py`

**Interfaces:**
- Consumes: `load_cnndm`, `prepare` (Task 7).
- Produces: CLI `uv run python -m ytb.train --out DIR [--run-dir DIR] [--train-samples 20000] [--eval-samples 500] [--max-steps -1] [--batch 2] [--grad-accum 16] [--lr 1e-5] [--seed 0]`. It writes the final model, tokenizer and `train_config.json` into `--out`, and prints `peak_gpu_gb=<float>` last.

- [ ] **Step 1: Implement `main(argv: list[str] | None = None) -> None` in `ytb/train.py`**

- **Model loading:** load `Qwen/Qwen3-0.6B` in **fp32**, so bf16 is autocast and the optimizer states stay fp32 (pure-bf16 weights lose 1e-5 updates). Set `model.config.use_cache = False`.
- **Data:** train set = `prepare(tok, load_cnndm("train", n), augment=True)`; validation set = `prepare(tok, load_cnndm("validation", m), augment=False)`.
- **Collator:** `DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)`.
- **`TrainingArguments`:**

```python
TrainingArguments(output_dir=run_dir, per_device_train_batch_size=batch,
    per_device_eval_batch_size=batch, gradient_accumulation_steps=grad_accum,
    learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
    num_train_epochs=1, max_steps=max_steps, bf16=True, gradient_checkpointing=True,
    label_smoothing_factor=0.1, eval_strategy="steps", eval_steps=200,
    save_steps=200, save_total_limit=2, logging_steps=10,
    report_to="tensorboard", seed=seed)
```

- **Train and export:** `Trainer(...)`, then `.train()`. Then `trainer.save_model(out)`, `tok.save_pretrained(out)`, and write `train_config.json`.
- **Last line:** print `peak_gpu_gb=` + `torch.cuda.max_memory_allocated() / 2**30`.
- **Default run dir:** `--run-dir` defaults to `runs/<basename of --out>`.

- [ ] **Step 2: Smoke run (20 steps, 64 examples)**

Run: `uv run python -m ytb.train --out runs/smoke-model --train-samples 64 --eval-samples 16 --max-steps 20`

Expected:
- logged `loss` is finite, starts roughly 1–5, and falls over the 20 steps
- the last line reads `peak_gpu_gb=` with a value ≤ 13.0

If peak > 13: rerun with `--batch 1 --grad-accum 32`. If it is still > 13: add `optim="adafactor"`.

- [ ] **Step 3: Commit**

```bash
git add src/ytb/train.py
git commit -m "feat: full fine-tune of Qwen3-0.6B with label smoothing"
```

- [ ] **Step 4: Full run in the background (about 1 h)**

```bash
nohup uv run python -m ytb.train --out models/ytb-summarizer > runs/train.log 2>&1 &
```

Use the batch and grad-accum values that passed Step 2. Watch with `tail -f runs/train.log`, or `uv run tensorboard --logdir runs --port 18502`.

Expected: `eval_loss` falls at each 200-step eval. At the end, `models/ytb-summarizer/` holds `model.safetensors`, `tokenizer.json` and `train_config.json`.

---

### Task 10: Acceptance — evaluate, serve the fine-tuned model, captured session again

**Files:**
- Create: `docs/manual-test/<YYYY-MM-DD>-finetuned/` (same layout as Task 6)
- Modify: `README.md`

**Interfaces:**
- Consumes: `models/ytb-summarizer/` (Task 9), `ytb.evaluate` (Task 8), the app and `scripts/manual_session.py` (Tasks 4–6).

- [ ] **Step 1: Evaluate the fine-tuned model**

Run: `uv run python -m ytb.evaluate --model models/ytb-summarizer`
Expected: it writes `models/ytb-summarizer/eval.json`.

**Accept** when the fine-tuned model beats `runs/eval-base.json` on `rouge1`, `rouge2` and `rougeLsum`, and its `rouge1` ≥ lead-3 `rouge1` − 3. If it is not accepted, stop and report both JSONs to the user; do not tune blindly.

- [ ] **Step 2: Serve the fine-tuned model and repeat the captured session**

```bash
pkill -f "ytb.api" ; nohup uv run python -m ytb.api --model models/ytb-summarizer > runs/app.log 2>&1 &
until curl -sf http://127.0.0.1:18500/api/health; do sleep 2; done
OUT=docs/manual-test/$(date +%F)-finetuned
uv run python scripts/manual_session.py --base-url http://127.0.0.1:18500 --out $OUT
```

Apply the Task 6 Step 4 checklist (the header must now show `models/ytb-summarizer`), and fix and re-run the same way. Write `$OUT/REPORT.md` as in Task 6 Step 6, plus a side-by-side table of the base and fine-tuned summaries for scenarios 02, 04 and 06, taken from both `session.json` files.

- [ ] **Step 3: Write the README**

Document:
- the serve command (base and fine-tuned)
- the Whisper flags
- the manual-session command
- the training, smoke and evaluate commands
- the port
- the shared-GPU note
- where the model and the captures live

- [ ] **Step 4: Commit**

```bash
git add README.md docs/manual-test
git commit -m "docs: fine-tuned model session captures and README"
```
