"""Summarize any length of text: one pass if it fits, else chunk -> summarize -> combine (bounded rounds)."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Summary:
    text: str
    chunks: int
    input_tokens: int
    output_tokens: int
    seconds: float


class Summarizer:
    def __init__(
        self,
        generate: Callable[[str], str],
        tokenizer,
        max_input_tokens: int = 1024,
        overlap_tokens: int = 100,
        max_rounds: int = 3,
    ):
        self.generate = generate
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.overlap_tokens = overlap_tokens
        self.max_rounds = max_rounds
        self._lock = threading.Lock()  # one generation at a time on the GPU

    def _encode(self, text: str) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _windows(self, ids: list) -> list[list]:
        step = self.max_input_tokens - self.overlap_tokens
        windows = []
        for start in range(0, len(ids), step):
            windows.append(ids[start : start + self.max_input_tokens])
            if start + self.max_input_tokens >= len(ids):
                break
        return windows

    def summarize(self, text: str) -> Summary:
        with self._lock:
            t0 = time.perf_counter()
            ids = self._encode(text)
            input_tokens, chunks, rounds = len(ids), 1, 0
            while len(ids) > self.max_input_tokens and rounds < self.max_rounds:
                windows = self._windows(ids)
                if rounds == 0:
                    chunks = len(windows)
                ids = self._encode("\n".join(self.generate(self._decode(w)) for w in windows))
                rounds += 1
            final_input = text if rounds == 0 and len(ids) <= self.max_input_tokens else self._decode(ids[: self.max_input_tokens])
            summary = self.generate(final_input)
            return Summary(summary, chunks, input_tokens, len(self._encode(summary)), round(time.perf_counter() - t0, 2))
