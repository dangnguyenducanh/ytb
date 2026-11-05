"""Use the ytb page like a person (Playwright + headless Chromium).

Captures a screenshot of every state and a video of the whole session, plus session.json.
Usage: uv run python scripts/manual_session.py --base-url http://127.0.0.1:18500 --out docs/manual-test/<date>-base-model
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PASTED_TEXT = (
    "The city of Riverton will replace its entire diesel bus fleet with electric buses by 2028, "
    "the transport department announced on Monday. The plan covers 240 buses and includes new "
    "charging depots at three existing garages. Officials said the switch will cut the fleet's "
    "carbon emissions by about 60 percent and save roughly 9 million dollars a year in fuel and "
    "maintenance. The first 40 buses will arrive next spring and run on the busiest downtown routes. "
    "Drivers will receive two weeks of training on the new vehicles. Some council members questioned "
    "the 310 million dollar cost, but the department said federal grants will cover nearly half of it. "
    "Riders' groups welcomed the news, saying the quieter buses will also reduce noise in residential "
    "streets. A public meeting on the depot locations is scheduled for next month."
)

SCENARIOS = [
    ("02-captions", {"url": "https://www.youtube.com/watch?v=arj7oStGLkU"}),
    ("03-cached", {"url": "https://youtu.be/arj7oStGLkU"}),
    ("04-long", {"url": "https://www.youtube.com/watch?v=zjkBMFhNj_g"}),
    ("05-invalid", {"url": "https://vimeo.com/123"}),
    ("06-text", {"text": PASTED_TEXT}),
]


def _visible_text(page, selector: str) -> str:
    loc = page.locator(selector)
    return loc.inner_text() if loc.is_visible() else ""


def _open(page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_function("document.getElementById('model').textContent !== 'loading…'")
    page.wait_for_timeout(800)


def run_scenario(page, base_url: str, out: Path, name: str, inp: dict) -> dict:
    _open(page, base_url)
    if "url" in inp:
        page.locator("#url").click()
        page.locator("#url").press_sequentially(inp["url"], delay=60)  # type like a person
    else:
        page.locator("#paste summary").click()
        page.wait_for_timeout(400)
        page.locator("#text").fill(inp["text"])  # paste
    page.wait_for_timeout(600)
    page.screenshot(path=out / f"{name}-input.png")
    page.locator("#go").click()
    t0 = time.perf_counter()
    page.wait_for_timeout(1500)
    page.screenshot(path=out / f"{name}-waiting.png")
    page.wait_for_selector("#result:not([hidden]), #error:not([hidden])", timeout=15 * 60 * 1000)
    seconds = round(time.perf_counter() - t0, 2)
    page.wait_for_timeout(400)
    page.screenshot(path=out / f"{name}-done.png", full_page=True)
    entry = {
        "scenario": name,
        "input": inp.get("url") or inp["text"][:80] + "…",
        "client_seconds": seconds,
        "summary": _visible_text(page, "#summary"),
        "stats": _visible_text(page, "#stats"),
        "error": _visible_text(page, "#error"),
    }
    page.wait_for_timeout(2000)  # a person reads the result
    return entry


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:18500")
    p.add_argument("--out", required=True)
    p.add_argument("--no-captions-url", default=None)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS + ([("07-no-captions", {"url": a.no_captions_url})] if a.no_captions_url else [])

    log = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        size = {"width": 1280, "height": 860}
        ctx = browser.new_context(viewport=size, record_video_dir=str(out / "_video"), record_video_size=size)
        page = ctx.new_page()
        _open(page, a.base_url)
        page.screenshot(path=out / "01-home.png")
        for name, inp in scenarios:
            entry = run_scenario(page, a.base_url, out, name, inp)
            log.append(entry)
            print(json.dumps(entry, ensure_ascii=False)[:300], flush=True)
        video = Path(page.video.path())
        ctx.close()  # finalizes the video
        browser.close()
    shutil.move(video, out / "session.webm")
    shutil.rmtree(out / "_video", ignore_errors=True)
    (out / "session.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
