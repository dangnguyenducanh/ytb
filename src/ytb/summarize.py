"""Summarize any length of text as structured notes: key points from each part in order, then overview + takeaway.

The notes grow with the input (3-5 points per part) and cover every part, so a long video is not summarized from its
opening alone. Self-check: `uv run python -m ytb.summarize`.
"""

import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ytb.prompt import overview_prompt, part_prompt

_MARK = re.compile(r"^\s*(?:(?:[-*•]\s*)+|\d+[.)]\s+)")  # "- ", "- - ", "* ", "• ", "1. ", "2) "


def bullets(reply: str) -> list[str]:
    """A model reply as clean '- ' bullets: only the marked lines (drops 'Here are the points:'), unless none are."""
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    marked = [line for line in lines if _MARK.match(line)]
    items = (_MARK.sub("", line).strip() for line in marked or lines)
    return [f"- {item}" for item in items if item]


@dataclass
class Summary:
    text: str
    chunks: int
    input_tokens: int
    output_tokens: int
    seconds: float
    chunk_tokens: list[int]  # token count of each part


class Summarizer:
    def __init__(self, generate: Callable[[str], str], tokenizer, part_tokens: int = 2048):
        self.generate = generate
        self.tokenizer = tokenizer
        self.part_tokens = part_tokens  # ~8 min of speech: base Qwen3-0.6B gives 3-5 concrete points per part
        self._lock = threading.Lock()  # one generation at a time on the GPU

    def _encode(self, text: str) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, ids: list) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _parts(self, ids: list) -> list[list]:
        """Equal parts of at most part_tokens: no tiny tail part that would pad the notes with filler."""
        size = max(1, math.ceil(len(ids) / max(1, math.ceil(len(ids) / self.part_tokens))))
        return [ids[i : i + size] for i in range(0, len(ids), size)] or [ids]

    def summarize(self, text: str) -> Summary:
        with self._lock:
            t0 = time.perf_counter()
            ids = self._encode(text)
            parts = self._parts(ids)
            points = "\n".join(
                b for i, p in enumerate(parts, 1) for b in bullets(self.generate(part_prompt(self._decode(p), i, len(parts))))
            )
            overview, _, takeaway = self.generate(overview_prompt(points)).partition("Takeaway:")
            overview, takeaway = overview.strip().removeprefix("Overview:").strip(), takeaway.strip()
            sections = [f"Overview: {overview}" if overview else "", f"Key points:\n{points}" if points else "",
                        f"Takeaway: {takeaway}" if takeaway else ""]
            summary = "\n\n".join(s for s in sections if s)
            return Summary(summary, len(parts), len(ids), len(self._encode(summary)),
                           round(time.perf_counter() - t0, 2), [len(p) for p in parts])


if __name__ == "__main__":
    assert bullets("Here are the points:\n- - a\n* b\n\n2. c 140GB\n- 2D plot") == ["- a", "- b", "- c 140GB", "- 2D plot"]
    assert bullets("Plain prose reply.") == ["- Plain prose reply."]

    class Tok:  # one token per word
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    prompts = []

    def fake(prompt: str) -> str:
        prompts.append(prompt)
        return "Overview: About x.\n\nTakeaway: Do y." if prompt.startswith("Here are the key points") else f"- p{len(prompts)}"

    s = Summarizer(fake, Tok(), part_tokens=4).summarize("w " * 9)
    assert s.chunks == 3 and s.chunk_tokens == [3, 3, 3], s.chunk_tokens  # 9 tokens, max 4 -> 3 equal parts, no tail
    assert s.text == "Overview: About x.\n\nKey points:\n- p1\n- p2\n- p3\n\nTakeaway: Do y.", s.text
    assert "part 3 of 3" in prompts[2] and "- p3" in prompts[3]  # every part is read, and all points reach the overview
    print("summarize self-check ok")
