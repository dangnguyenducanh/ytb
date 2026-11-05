"""autoresearch on the ytb summarizer: THE file the agent edits.

Baseline: keep fine-tuning the served v1 model on CNN/DailyMail rows it has never seen, with v1's own
recipe, for the fixed time budget; prepare.report() then scores it on the fixed dev set.

    uv run python autoresearch/train.py > autoresearch/run.log 2>&1
"""

import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import prepare  # noqa: E402  (autoresearch/prepare.py, fixed)
from ytb.data import prepare as build_dataset  # noqa: E402  (prompt-masked examples with the app's prompt)

T0 = time.time()

# ---- knobs (v1's recipe) ----
LR = 1e-5
LABEL_SMOOTHING = 0.1
BATCH, GRAD_ACCUM = 4, 8  # effective batch 32
WARMUP_STEPS = 10
NEW_ROWS = 8000  # unseen training rows on offer; ~5,000 fit in the 10-minute budget


class TimeBudget(TrainerCallback):
    """Stop training once prepare.TIME_BUDGET seconds of training have passed."""

    def __init__(self):
        self.start = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start >= prepare.TIME_BUDGET:
            control.should_training_stop = True


tok = AutoTokenizer.from_pretrained(prepare.INIT_MODEL)
model = AutoModelForCausalLM.from_pretrained(prepare.INIT_MODEL, dtype=torch.float32)
model.config.use_cache = False
train_ds = build_dataset(tok, prepare.unseen_train_rows(NEW_ROWS), augment=True)

args = TrainingArguments(
    output_dir=str(prepare.ROOT / "runs" / "autoresearch" / "trainer"),
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    label_smoothing_factor=LABEL_SMOOTHING,
    optim="adafactor",
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    disable_tqdm=True,
    seed=0,
)
budget = TimeBudget()
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
    callbacks=[budget],
)
trainer.train()
prepare.report(trainer.model, tok, training_seconds=time.time() - budget.start,
               num_steps=trainer.state.global_step, t_start=T0)
