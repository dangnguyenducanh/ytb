"""Full fine-tune of Qwen3-0.6B on CNN/DailyMail with label-smoothed loss on summary tokens only.

uv run python -m ytb.train --out DIR [--run-dir DIR] [--train-samples 20000] [--eval-samples 500]
    [--max-steps -1] [--batch 1] [--grad-accum 32] [--lr 1e-5] [--seed 0]
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ytb.data import load_cnndm, prepare

BASE_MODEL = "Qwen/Qwen3-0.6B"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--run-dir", help="checkpoints and tensorboard logs; default runs/<basename of --out>")
    p.add_argument("--train-samples", type=int, default=20000)
    p.add_argument("--eval-samples", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run_dir = args.run_dir or str(Path("runs") / Path(args.out).name)
    # shared GPU: variable-length batches fragment the cache; keep reserved memory close to allocated
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    # fp32 weights + bf16 autocast: optimizer states stay fp32 (pure-bf16 weights lose 1e-5 updates)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    model.config.use_cache = False

    train_ds = prepare(tok, load_cnndm("train", args.train_samples), augment=True)
    eval_ds = prepare(tok, load_cnndm("validation", args.eval_samples), augment=False)

    targs = TrainingArguments(
        output_dir=run_dir, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.03,  # float < 1 = ratio (v5)
        num_train_epochs=1, max_steps=args.max_steps, bf16=True, gradient_checkpointing=True,
        label_smoothing_factor=0.1, eval_strategy="steps", eval_steps=200, eval_on_start=True,
        save_steps=200, save_total_limit=2, logging_steps=10, logging_first_step=True,
        # ponytail: AdamW peaked at 11.36 GB (> 11 GB shared-GPU cap); Adafactor's factored states save ~4.4 GB.
        # Back to AdamW (drop this line) when the GPU has more room.
        optim="adafactor", report_to="tensorboard", seed=args.seed)
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True))
    trainer.train()

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    config = {**vars(args), "run_dir": run_dir, "base_model": BASE_MODEL, "label_smoothing_factor": 0.1,
              "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "num_train_epochs": 1, "optim": "adafactor"}
    (Path(args.out) / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"peak_gpu_gb={torch.cuda.max_memory_allocated() / 2**30:.2f}")


if __name__ == "__main__":
    main()
