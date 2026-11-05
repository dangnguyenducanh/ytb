"""Single source of truth for how text becomes model input (serving, evaluation and training)."""

# Training and ROUGE evaluation: the CNN/DailyMail format, so scores stay comparable across runs.
INSTRUCTION = "Summarize the following text in 3 to 4 sentences."
# No n-gram ban: Qwen's BPE has separate tokens for " London" and "London", so no_repeat_ngram_size=3 pushed
# greedy decoding to the unspaced twin ("theLondon") and cost ~6 ROUGE-1 / ~9 ROUGE-2 on 100 test articles.
# A mild repetition penalty scores the same and still guards a 0.6B model against loops on long transcripts.
GENERATION = {"do_sample": False, "max_new_tokens": 200, "repetition_penalty": 1.1}


def chat(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def build_prompt(tokenizer, text: str) -> str:
    return chat(tokenizer, f"{INSTRUCTION}\n\n{text}")


# The app: structured notes whose length grows with the video (key points per part, then overview + takeaway).
# Base Qwen3-0.6B follows these; the CNN/DM fine-tune ignores them and writes 3-4 sentences of prose.
# Instructions go after the text: a small model drops instructions that sit before thousands of tokens.
SERVE_GENERATION = {**GENERATION, "max_new_tokens": 512}
_RULES = "Use only what the speaker or author says. Keep names and numbers exactly as given. No bold, no headings."


def part_prompt(text: str, i: int, n: int) -> str:
    return (
        f"Here is part {i} of {n} of a text, often a video transcript that may lack punctuation and contain caption "
        f"errors.\n\n<text>\n{text}\n</text>\n\nList the main points of this part as 3 to 5 bullets, in the order they "
        f"come up, one idea per bullet. Output only the bullets, each starting with '- '. {_RULES}"
    )


def overview_prompt(points: str) -> str:
    return (
        f"Here are the key points of a text, in order:\n\n{points}\n\nUse exactly this format:\n\n"
        # "whole text ... not only the first ones": without it the overview described only the opening point
        "Overview: <2-3 sentences: the main topic of the whole text and its main message, based on all the points, "
        "not only the first ones>\n\n"
        f"Takeaway: <one sentence: the most useful lesson of the whole text>\n\n{_RULES}"
    )
