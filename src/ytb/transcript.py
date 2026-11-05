"""Node 1 (ingestion): YouTube link -> transcript, from captions or a local Whisper fallback, cached on disk."""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)
from youtube_transcript_api import VideoUnavailable as _YTUnavailable

log = logging.getLogger(__name__)


class TranscriptError(Exception):
    """Base for ingestion errors; str(error) is a user-facing message."""


class InvalidUrl(TranscriptError): ...
class VideoUnavailable(TranscriptError): ...
class NoTranscript(TranscriptError): ...
class TooLong(TranscriptError): ...
class YouTubeBlocked(TranscriptError): ...


@dataclass
class Transcript:
    video_id: str
    language: str
    text: str
    source: str  # "captions" | "whisper"


_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com"}
_CUE = re.compile(r"\[[^\]]*\]")  # [Music], [Applause], ...


def parse_video_id(url: str) -> str:
    url = url.strip()
    if _ID.fullmatch(url):
        return url
    u = urlparse(url if "://" in url else "https://" + url)
    host = (u.hostname or "").lower()
    vid = ""
    if host in _HOSTS:
        if u.path == "/watch":
            vid = parse_qs(u.query).get("v", [""])[0]
        else:
            parts = u.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                vid = parts[1]
    elif host == "youtu.be":
        vid = u.path.strip("/").split("/")[0]
    if not _ID.fullmatch(vid):
        raise InvalidUrl("That is not a YouTube video link.")
    return vid


def fetch_transcript(
    video_id: str, cache_dir: Path, fallback: Callable[[str], tuple[str, str]] | None = None
) -> Transcript:
    path = Path(cache_dir) / f"{video_id}.json"
    if path.exists():
        log.info("cache hit %s", video_id)
        return Transcript(**json.loads(path.read_text()))
    log.info("fetch %s", video_id)
    t = _from_captions(video_id) or _from_audio(video_id, fallback)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(t), ensure_ascii=False))
    return t


def _from_captions(video_id: str) -> Transcript | None:
    """Captions transcript, or None when the video has no captions at all."""
    api = YouTubeTranscriptApi()
    try:
        try:
            fetched = api.fetch(video_id, languages=["en"])
        except NoTranscriptFound:
            listed = list(api.list(video_id))
            if not listed:
                return None
            fetched = listed[0].fetch()
    except TranscriptsDisabled:
        return None
    except (IpBlocked, RequestBlocked) as e:
        raise YouTubeBlocked("YouTube is rate-limiting this machine; try again later.") from e
    except _YTUnavailable as e:
        raise VideoUnavailable("This video is unavailable (private, deleted or blocked).") from e
    except CouldNotRetrieveTranscript as e:
        raise NoTranscript("Could not get captions for this video.") from e
    text = " ".join(_CUE.sub(" ", " ".join(s.text for s in fetched.snippets)).split())
    if not text:
        raise NoTranscript("This video's captions contain no speech.")
    return Transcript(video_id, fetched.language_code, text, "captions")


def _from_audio(video_id: str, fallback: Callable[[str], tuple[str, str]] | None) -> Transcript:
    if fallback is None:
        raise NoTranscript("This video has no captions.")
    language, text = fallback(video_id)
    text = " ".join(text.split())
    if not text:
        raise NoTranscript("No speech found in this video's audio.")
    return Transcript(video_id, language, text, "whisper")
