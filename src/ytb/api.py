"""FastAPI backend + single-page frontend. Run: uv run python -m ytb.api --model Qwen/Qwen3-0.6B"""

import argparse
import functools
import logging
from collections.abc import Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ytb.summarize import Summarizer
from ytb.transcript import (
    InvalidUrl,
    NoTranscript,
    TooLong,
    Transcript,
    TranscriptError,
    VideoUnavailable,
    YouTubeBlocked,
    fetch_transcript,
    parse_video_id,
)

MAX_CHARS = 150_000
_STATUS = {InvalidUrl: 422, VideoUnavailable: 404, NoTranscript: 404, TooLong: 413, YouTubeBlocked: 503}
_PAGE = Path(__file__).parent / "static" / "index.html"


class SummarizeRequest(BaseModel):
    url: str | None = None
    text: str | None = None


def create_app(summarizer: Summarizer, fetch: Callable[[str], Transcript], model_name: str) -> FastAPI:
    app = FastAPI(title="ytb")

    @app.get("/")
    def index():
        return FileResponse(_PAGE)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.post("/api/summarize")
    def summarize(req: SummarizeRequest):
        url, text = (req.url or "").strip(), (req.text or "").strip()
        if bool(url) == bool(text):
            raise HTTPException(422, "Give exactly one of a YouTube link or text.")
        transcript = None
        if url:
            try:
                transcript = fetch(parse_video_id(url))
            except TranscriptError as e:
                raise HTTPException(_STATUS.get(type(e), 404), str(e)) from e
            text = transcript.text
        if len(text) > MAX_CHARS:
            raise HTTPException(413, f"The text is {len(text):,} characters; the limit is {MAX_CHARS:,}.")
        s = summarizer.summarize(text)
        return {
            "summary": s.text,
            "video_id": transcript.video_id if transcript else None,
            "language": transcript.language if transcript else None,
            "source": transcript.source if transcript else None,
            "chunks": s.chunks,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "seconds": s.seconds,
            "model": model_name,
        }

    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Serve the ytb summarizer (backend + frontend).")
    p.add_argument("--model", default="models/ytb-summarizer", help="HF model folder or Hub id")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18500)
    p.add_argument("--cache-dir", default=".cache/transcripts")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--whisper-device", default="cpu")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from ytb.audio import transcribe_audio
    from ytb.model import load

    generate, tok = load(a.model, a.device)
    fallback = functools.partial(transcribe_audio, model_size=a.whisper_model, device=a.whisper_device)
    fetch = functools.partial(fetch_transcript, cache_dir=Path(a.cache_dir), fallback=fallback)
    uvicorn.run(create_app(Summarizer(generate, tok), fetch, a.model), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
