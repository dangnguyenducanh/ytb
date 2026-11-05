"""CNN/DailyMail training examples: prompt-masked labels and caption-style augmentation."""

import random
import unicodedata

from datasets import Dataset, load_dataset

from ytb.prompt import build_prompt

MAX_INPUT_TOKENS = 1024
MAX_TARGET_TOKENS = 256
CAPTION_FRACTION = 0.3


def to_caption_style(text: str) -> str:
    """Make text look like raw YouTube auto-captions: lowercase, no punctuation."""
    text = "".join(c for c in text.casefold() if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def build_example(tokenizer, article: str, summary: str) -> dict[str, list[int]]:
    """Prompt tokens (labels -100) followed by summary tokens + <|im_end|>; the article is truncated, not the target."""
    article = tokenizer.decode(tokenizer.encode(article, add_special_tokens=False)[:MAX_INPUT_TOKENS])
    prompt = tokenizer(build_prompt(tokenizer, article), add_special_tokens=False).input_ids
    target = tokenizer.encode(summary, add_special_tokens=False)[:MAX_TARGET_TOKENS]
    target.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    input_ids = prompt + target
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": [-100] * len(prompt) + target}


def load_cnndm(split: str, n: int, seed: int = 0) -> Dataset:
    ds = load_dataset("abisee/cnn_dailymail", data_files={split: f"3.0.0/{split}-*.parquet"}, split=split)
    ds = ds.shuffle(seed=seed).select(range(n)).select_columns(["article", "highlights"])
    return ds.map(lambda r: {"highlights": r["highlights"].replace("\n", " ")})


def prepare(tokenizer, ds: Dataset, augment: bool, seed: int = 0) -> Dataset:
    def example(row, i):
        article = row["article"]
        if augment and random.Random(seed * 1_000_003 + i).random() < CAPTION_FRACTION:
            article = to_caption_style(article)
        return build_example(tokenizer, article, row["highlights"])

    return ds.map(example, with_indices=True, remove_columns=ds.column_names)
