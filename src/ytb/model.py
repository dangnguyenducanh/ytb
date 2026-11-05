from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ytb.prompt import GENERATION, build_prompt


def load(model_id: str, device: str | None = None, prompt=build_prompt, generation: dict = GENERATION,
         ) -> tuple[Callable[[str], str], object]:
    """Load a causal LM (Hub id or local folder); return (generate(text) -> reply, tokenizer).

    Defaults are the CNN/DM summary format (training, ROUGE); the app passes prompt=chat and SERVE_GENERATION.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()

    def generate(text: str) -> str:
        inputs = tok(prompt(tok, text), return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, **generation, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return generate, tok
