"""Score a model with ROUGE on a fixed CNN/DailyMail test sample, next to lead-3.

uv run python -m ytb.evaluate --model ID_OR_DIR [--samples 500] [--out PATH]
"""

import argparse
import json
from pathlib import Path

from ytb.data import load_cnndm
from ytb.model import load
from ytb.rouge import lead3, rouge


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--out", help="default: <model dir>/eval.json when --model is a directory")
    args = p.parse_args(argv)
    out = args.out or (Path(args.model) / "eval.json" if Path(args.model).is_dir() else None)

    ds = load_cnndm("test", args.samples, seed=0)
    generate, _ = load(args.model)
    # ponytail: sequential generate (~10-15 min/500); batch with left padding if eval speed matters
    preds = []
    for i, article in enumerate(ds["article"], 1):
        preds.append(generate(article))
        if i == 1 or i % 25 == 0:
            print(f"{i}/{len(ds)} {preds[-1][:200]!r}", flush=True)

    report = {
        "model": args.model,
        "samples": args.samples,
        "lead3": rouge([lead3(a) for a in ds["article"]], ds["highlights"]),
        "scores": rouge(preds, ds["highlights"]),
    }
    print("lead3 ", report["lead3"])
    print("scores", report["scores"])
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()
