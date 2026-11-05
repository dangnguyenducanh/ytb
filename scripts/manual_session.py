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
    page.wait_for_selector("#service:not([data-state='loading'])")  # the top bar's Ready / Offline status
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
    p.add_argument("--storage", default=None,
                   help="browser storage JSON: loaded before the session if it exists, saved after (keeps the page's run history)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS + ([("07-no-captions", {"url": a.no_captions_url})] if a.no_captions_url else [])

    log = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        size = {"width": 1280, "height": 860}
        state = a.storage if a.storage and Path(a.storage).exists() else None
        ctx = browser.new_context(viewport=size, record_video_dir=str(out / "_video"), record_video_size=size,
                                  storage_state=state)
        page = ctx.new_page()
        _open(page, a.base_url)
        page.screenshot(path=out / "01-home.png")
        for name, inp in scenarios:
            entry = run_scenario(page, a.base_url, out, name, inp)
            log.append(entry)
            print(json.dumps(entry, ensure_ascii=False)[:300], flush=True)
        # the theme toggle in the top bar: a person switches the last result to dark
        page.locator("#theme").click()
        page.wait_for_timeout(800)
        page.screenshot(path=out / "08-dark-theme.png", full_page=True)
        log.append({"scenario": "08-dark-theme", "theme": page.evaluate("document.documentElement.dataset.theme")})
        page.wait_for_timeout(2000)
        video = Path(page.video.path())
        if a.storage:
            ctx.storage_state(path=a.storage)
        state = ctx.storage_state()  # this session's history, carried to the smaller screens
        ctx.close()  # finalizes the video
        # responsive: the same history on a phone and a tablet, the 1-hour talk picked from the history list
        for name, size, theme in [("09-phone-light", (390, 844), "light"), ("10-phone-dark", (390, 844), "dark"),
                                  ("11-tablet", (768, 1024), "light")]:
            small = browser.new_context(viewport={"width": size[0], "height": size[1]}, storage_state=state)
            sp = small.new_page()
            _open(sp, f"{a.base_url}/?theme={theme}")
            sp.locator("#history .row-card").nth(1).click()  # second row: the 1-hour talk
            sp.wait_for_timeout(600)
            sp.screenshot(path=out / f"{name}.png", full_page=True)
            log.append({"scenario": name, "viewport": f"{size[0]}x{size[1]}", "theme": theme,
                        "headline": sp.locator("#greeting").inner_text(),
                        "horizontal_overflow_px": sp.evaluate(
                            "document.documentElement.scrollWidth - document.documentElement.clientWidth")})
            small.close()
        browser.close()
    shutil.move(video, out / "session.webm")
    shutil.rmtree(out / "_video", ignore_errors=True)
    (out / "session.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
