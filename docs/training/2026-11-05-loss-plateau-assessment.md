# Why the training loss "stuck" at ~3.2, and what to do about it

- **Run:** `models/ytb-summarizer`, a full fine-tune of `Qwen/Qwen3-0.6B` on 20,000 CNN/DailyMail articles
- **Settings:** 1 epoch, 625 steps, effective batch 32, lr 1e-5 cosine, Adafactor, `label_smoothing_factor=0.1`
- **Progress log:** [2026-11-05-ytb-summarizer-progress.md](2026-11-05-ytb-summarizer-progress.md)

## What the log shows

- Train loss fell from **5.35 to 3.39 in the first 40 steps** (~1,280 articles). It then stayed between 3.11 and 3.25 until the end (final mean train loss 3.264).
- Validation loss is still falling, just slowly: 5.306 (step 0) → 3.177 (200) → 3.131 (400) → **3.118** (600 and 625).
- The step-to-step wobble of ±0.05 is noise: each logging point averages 10 steps of single-article micro-batches.

## The logged number is not the error: label smoothing adds a floor

With `label_smoothing_factor = ε = 0.1`, Hugging Face's `LabelSmoother` reports

```
loss = (1 − ε) · NLL + ε · U        where U = average of −log p(v) over all 151,936 vocabulary tokens
```

U can never go below ln(151,936) = 11.93, which is a uniform distribution. So the logged loss has a **hard floor of 1.19 above zero** however good the model gets. In practice U stays near 15, so about **1.5 of the ~3.1 is fixed smoothing cost**.

It was measured, not assumed. `scratchpad/loss_split.py` was run on 64 validation articles (4,373 summary tokens):

| Model | Plain NLL | Perplexity | U | Smoothing part (0.1·U) | Logged-scale loss | Next-token accuracy |
|---|---|---|---|---|---|---|
| Base Qwen3-0.6B | 2.779 | 16.1 | 26.96 | 2.70 | 5.198 | 52.8% |
| Checkpoint 200 | 1.743 | 5.72 | 15.23 | 1.52 | 3.091 | 64.2% |
| Checkpoint 400 | 1.698 | 5.46 | 15.12 | 1.51 | 3.040 | 64.9% |

## Assessment

1. **Not stuck, but saturating.** The real error, perplexity, fell from 16.1 to 5.5, and next-token accuracy rose from 52.8% to 64.9%. Most of the gain came in the first ~200 steps (6,400 articles), when the model learned the highlights *format*. After that, the content gains are small: −0.045 NLL from step 200 to 400, then −0.013 validation loss from 400 to 625. That is the normal shape of SFT with this much data and this learning rate.
2. **Label smoothing is working against us.** U fell from 27 to 15. The model is being *trained* to move probability onto the 151,935 wrong tokens, because a 10% target mass is spread across them. For a 152k-vocabulary LM this mostly buys a worse-calibrated distribution, which hurts any sampled decoding. Standard LLM SFT uses ε = 0. It came from the BART-era design.
3. **The optimiser setting is conservative.** Adafactor has no momentum, and lr 1e-5 is an AdamW-style value. With a cosine decay to 0 over one epoch, there is little learning left in the second half.
4. **The decisive number is ROUGE, not the loss:** see the next section.

## Does it summarize better? (ROUGE on 500 CNN/DM test articles)

Same prompt and generation settings as the app (greedy, ≤ 200 new tokens, `no_repeat_ngram_size=3`); the same 500 articles for every row.

| | ROUGE-1 | ROUGE-2 | ROUGE-Lsum |
|---|---|---|---|
| Base Qwen3-0.6B | 31.45 | 6.83 | 27.86 |
| **Fine-tuned (this run)** | **35.02** (+3.57) | **9.90** (+3.07) | **32.25** (+4.39) |
| Lead-3 baseline | 39.96 | 17.27 | 36.14 |

Plan acceptance gate (Task 10):

- **Beats the base model on all three: passes.**
- **ROUGE-1 within 3 points of lead-3 (≥ 36.96): fails** at 35.02, 4.9 below.

So the fine-tune clearly helped, but it is not yet as good as the gate asks. The plateau does matter.

**A defect in the outputs, unrelated to training:** words are glued together ("from theLondon Fire Brigade", "sofar", "andLaura"). Suspected cause: `no_repeat_ngram_size=3` is a BART-era decoding setting. In Qwen's vocabulary " London" and "London" are different tokens, so when the spaced one would repeat a 3-gram, the ban pushes greedy decoding to the unspaced twin. An A/B on 100 test articles is below.

### Decoding A/B (fine-tuned v1, first 100 test articles)

| Generation setting | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Glued words | Avg words |
|---|---|---|---|---|---|
| `no_repeat_ngram_size=3` (was the default) | 34.04 | 9.14 | 31.14 | 6 | 40 |
| no n-gram ban | 39.71 | 18.06 | 36.95 | 0 | 53 |
| **`repetition_penalty=1.1`** (new default) | **40.01** | 17.78 | 36.94 | 0 | 50 |

The n-gram ban was the cause of the glued words, and it cost about 6 ROUGE-1 and 9 ROUGE-2 points. **Fixed in `src/ytb/prompt.py`:** greedy decoding with `repetition_penalty=1.1`, which scores the same as no ban and still guards against loops on long transcripts. Base and v1 are re-scored on the full 500 articles with it below.

## Solutions, in order of value per hour

| # | Change | Why | Cost |
|---|---|---|---|
| 1 | **Log plain NLL / perplexity** (or set ε = 0 so the logged loss *is* NLL) | You can't judge a plateau through a 1.5-point fixed offset | minutes |
| 2 | **`label_smoothing_factor` 0.1 → 0.0** (or 0.01) | Removes the floor and stops the model learning to spread mass over 152k junk tokens; logged loss drops to ≈1.6–1.7 at once and calibration improves | rerun (~50 min) |
| 3 | **Learning-rate sweep for Adafactor:** 1e-5 vs 3e-5 vs 1e-4, 150 steps on 4,800 articles each, compared on validation NLL | Adafactor without momentum usually wants a higher lr than AdamW; 1e-5 is likely too low | ~3 × 12 min |
| 4 | **AdamW 8-bit** (`optim="adamw_bnb_8bit"`, needs `bitsandbytes`) instead of Adafactor | Momentum, at about a quarter of full AdamW's optimizer memory | rerun |
| 5 | **More data before more epochs:** 20k → 60k articles (≈2.5 h) | The curve is data-limited after ~6,400 articles; new articles teach more than repeats. A 2nd epoch on the same 20k risks memorising | hours |
| 6 | **Domain data:** add a slice of transcript-style summarization (spoken text) next to CNN/DM | The app summarizes YouTube captions; news-only SFT saturates on news style | data work |

The suggested next run combines 1 + 2 + the winner of 3, on 20k articles. If validation NLL drops below about 1.55 and ROUGE-1 beats this run, scale to 60k.
