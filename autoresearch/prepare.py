"""autoresearch harness for the ytb summarizer. FIXED: the agent must not modify this file.

Adapted from karpathy/autoresearch (prepare.py @ 228791f): fixed constants, the fixed dev set and the
ground-truth evaluation. train.py trains a model and calls report() as its very last step.

    uv run python autoresearch/prepare.py --score-init     # score the starting model as-is (reference row)

Env overrides exist only for smoke tests: AR_TIME_BUDGET, AR_DEV_N. Never set them in the experiment loop.
"""

import argparse
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ytb.data import load_cnndm
from ytb.prompt import GENERATION, build_prompt
from ytb.rouge import rouge

ROOT = Path(__file__).resolve().parent.parent  # this checkout (or worktree)
MAIN_CHECKOUT = Path("/home/aiface/dangnguyenducanh/ytb")  # where model weights live (git-ignored)
INIT_MODEL = str(MAIN_CHECKOUT / "models" / "ytb-summarizer")  # v1: the model the app serves today
KEEP_DIR = MAIN_CHECKOUT / "models" / "ytb-summarizer-auto"  # the agent copies a kept run's weights here
LAST_DIR = ROOT / "runs" / "autoresearch" / "last"  # weights of the most recent run (overwritten)

TIME_BUDGET = int(os.environ.get("AR_TIME_BUDGET", "600"))  # training wall clock per experiment, seconds
DEV_N = int(os.environ.get("AR_DEV_N", "200"))  # validation articles scored with ROUGE
DEV_SEED = 1234
V1_TRAIN_ROWS = 20_000  # v1 trained on rows [0, 20000) of load_cnndm("train", n, seed=0)


def unseen_train_rows(n: int) -> Dataset:
    """n CNN/DM training rows that v1 never saw (rows 20000.. of the same seed-0 order)."""
    return load_cnndm("train", V1_TRAIN_ROWS + n, seed=0).select(range(V1_TRAIN_ROWS, V1_TRAIN_ROWS + n))


def evaluate(model, tok) -> dict[str, float]:
    """Ground truth: ROUGE-1/2/Lsum x100 on DEV_N fixed validation articles, with the app's prompt and decoding."""
    ds = load_cnndm("validation", DEV_N, seed=DEV_SEED)
    model.eval()
    preds = []
    with torch.inference_mode():
        for article in ds["article"]:
            inputs = tok(build_prompt(tok, article), return_tensors="pt", add_special_tokens=False).to(model.device)
            out = model.generate(**inputs, **GENERATION, pad_token_id=tok.pad_token_id)
            preds.append(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    s = rouge(preds, ds["highlights"])
    return {"val_score": (s["rouge1"] + s["rouge2"] + s["rougeLsum"]) / 3, **s}


def _print(scores: dict[str, float], training_seconds: float, t_start: float, num_steps: int) -> None:
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    print("---")
    print(f"val_score:        {scores['val_score']:.6f}")
    print(f"rouge1:           {scores['rouge1']:.2f}")
    print(f"rouge2:           {scores['rouge2']:.2f}")
    print(f"rougeLsum:        {scores['rougeLsum']:.2f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"peak_vram_mb:     {peak:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"dev_articles:     {DEV_N}")


def report(model, tok, training_seconds: float, num_steps: int, t_start: float) -> None:
    """Call last in train.py: score the trained model, save it to LAST_DIR, print the summary block."""
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model = model.to(dtype=torch.bfloat16)
    scores = evaluate(model, tok)
    LAST_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(LAST_DIR)
    tok.save_pretrained(LAST_DIR)
    _print(scores, training_seconds, t_start, num_steps)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--score-init", action="store_true", help="score INIT_MODEL as-is (no training)")
    a = p.parse_args()
    if a.score_init:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(INIT_MODEL)
        model = AutoModelForCausalLM.from_pretrained(INIT_MODEL, dtype=torch.bfloat16).to("cuda")
        model.generation_config.use_cache = True
        _print(evaluate(model, tok), 0.0, t0, 0)
