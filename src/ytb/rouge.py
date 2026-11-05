"""ROUGE-1/2/Lsum (F x 100) with stemming and NLTK sentence splitting, plus the lead-3 baseline."""

import nltk
from rouge_score import rouge_scorer

KEYS = ("rouge1", "rouge2", "rougeLsum")
_scorer = rouge_scorer.RougeScorer(list(KEYS), use_stemmer=True)


def _sentences(text: str) -> str:
    return "\n".join(nltk.sent_tokenize(text))


def rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    scores = [_scorer.score(_sentences(r), _sentences(p)) for p, r in zip(preds, refs, strict=True)]
    return {k: round(100 * sum(s[k].fmeasure for s in scores) / len(scores), 2) for k in KEYS}


def lead3(article: str) -> str:
    return " ".join(nltk.sent_tokenize(article)[:3])
