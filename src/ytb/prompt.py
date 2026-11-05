"""Single source of truth for how text becomes model input (serving, evaluation and training)."""

INSTRUCTION = "Summarize the following text in 3 to 4 sentences."
# No n-gram ban: Qwen's BPE has separate tokens for " London" and "London", so no_repeat_ngram_size=3 pushed
# greedy decoding to the unspaced twin ("theLondon") and cost ~6 ROUGE-1 / ~9 ROUGE-2 on 100 test articles.
# A mild repetition penalty scores the same and still guards a 0.6B model against loops on long transcripts.
GENERATION = {"do_sample": False, "max_new_tokens": 200, "repetition_penalty": 1.1}


def build_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
