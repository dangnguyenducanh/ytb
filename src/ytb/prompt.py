"""Single source of truth for how text becomes model input (serving, evaluation and training)."""

INSTRUCTION = "Summarize the following text in 3 to 4 sentences."
GENERATION = {"do_sample": False, "max_new_tokens": 200, "no_repeat_ngram_size": 3}


def build_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
