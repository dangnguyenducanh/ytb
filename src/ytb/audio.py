"""Whisper fallback for videos without captions: pytubefix audio-only stream -> faster-whisper (no yt-dlp)."""

import functools
import tempfile

from pytubefix import YouTube
from pytubefix import exceptions as px

from ytb.transcript import NoTranscript, TooLong, VideoUnavailable, YouTubeBlocked


@functools.lru_cache(maxsize=2)
def _whisper(size: str, device: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type="int8" if device == "cpu" else "float16")


def transcribe_audio(
    video_id: str, model_size: str = "small", device: str = "cpu", max_seconds: int = 3600
) -> tuple[str, str, float]:
    """Return (language, text, audio length in seconds) transcribed locally from the video's audio."""
    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        if (yt.length or 0) > max_seconds:
            raise TooLong(f"This video has no captions and is longer than {max_seconds // 60} minutes.")
        stream = yt.streams.get_audio_only()
        if stream is None:
            raise NoTranscript("This video has no captions and no downloadable audio.")
        with tempfile.TemporaryDirectory() as tmp:
            path = stream.download(output_path=tmp, filename=f"{video_id}.m4a")
            segments, info = _whisper(model_size, device).transcribe(path, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments)  # consume the generator before tmp is deleted
            return info.language, text, info.duration
    except (px.BotDetection, px.PoTokenRequired) as e:
        raise YouTubeBlocked("YouTube is blocking audio downloads from this machine; try again later.") from e
    except px.VideoUnavailable as e:
        raise VideoUnavailable("This video is unavailable (private, deleted or blocked).") from e
    except (px.PytubeFixError, OSError) as e:
        raise NoTranscript("Could not download this video's audio.") from e
