"""Fine-tune gpt-4o-mini on the full training set (Phase 3 extension).

Few-shot prompting (llm_classify.py) only shows the model 15 labeled
examples total; the local models (TF-IDF, InLegalBERT) learn from all
1,288 training cases. Fine-tuning closes that gap by training on the full
set directly, at the cost of a one-time training job (see docs/MODEL_RESULTS.md
for the actual cost/result once run).

Unlike llm_classify.py's few-shot prompt, training examples here don't need
in-context exemplars or chain-of-thought -- the label pattern is learned
into the weights -- so each example is just (system + judgment excerpt) ->
single-word label, keeping training tokens (and therefore cost) down.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from evaluation.metrics import temporal_split
from training.llm_classify import LABEL_DEFS, WINDOW, _client, _tail
from training.train_baseline import DATA_PATH, DATE_COLUMN, LABEL_COLUMN, TEXT_COLUMN

ML_DIR = Path(__file__).resolve().parents[1]
REGISTRY = ML_DIR / "registry"
TRAIN_JSONL = REGISTRY / "_finetune_train.jsonl"

BASE_MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = (
    "You are an expert Singapore legal analyst classifying the disposition of "
    "court judgments into exactly one of these 5 categories:\n"
    + "\n".join(f"- {lbl}: {desc}" for lbl, desc in LABEL_DEFS.items())
    + "\n\nReply with ONLY the category name, nothing else."
)


def build_training_file(train_df: pd.DataFrame, out_path: Path = TRAIN_JSONL) -> Path:
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Judgment excerpt:\n{_tail(row[TEXT_COLUMN], WINDOW)}"},
                    {"role": "assistant", "content": row[LABEL_COLUMN]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def launch_job(client: OpenAI, train_path: Path) -> str:
    with train_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="fine-tune")
    job = client.fine_tuning.jobs.create(training_file=uploaded.id, model=BASE_MODEL)
    return job.id


def wait_for_job(client: OpenAI, job_id: str, poll_seconds: int = 30) -> str:
    seen_events = set()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        events = client.fine_tuning.jobs.list_events(job_id, limit=10)
        for e in reversed(events.data):
            if e.id not in seen_events:
                seen_events.add(e.id)
                print(f"  [{job.status}] {e.message}", flush=True)
        if job.status in ("succeeded", "failed", "cancelled"):
            if job.status != "succeeded":
                raise RuntimeError(f"Fine-tune job ended with status={job.status}")
            print(f"Done. Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        time.sleep(poll_seconds)


def main() -> str:
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    train_df, _ = temporal_split(df, DATE_COLUMN, test_fraction=0.2)

    print(f"Building training JSONL from {len(train_df)} cases...")
    train_path = build_training_file(train_df)
    print(f"Wrote {train_path}")

    client = _client()
    print("Uploading training file and launching fine-tune job...")
    job_id = launch_job(client, train_path)
    print(f"Job ID: {job_id}")

    model_name = wait_for_job(client, job_id)
    (REGISTRY / "latest_finetuned_llm.json").write_text(
        json.dumps({"model": model_name, "job_id": job_id, "base_model": BASE_MODEL}, indent=2)
    )
    return model_name


if __name__ == "__main__":
    main()
