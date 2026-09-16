"""LLM-as-judge scoring for the agent's English-sentence output quality.

This is a DIFFERENT axis of success from classification accuracy
(docs/MODEL_RESULTS.md Runs 1-15): it doesn't ask "did the model predict
the right label" (that's what accuracy already measures) -- it asks "is
the agent's English-language communication of whatever it predicted clear,
honest about uncertainty, and faithful to the underlying model output (no
overclaiming, no fabricated detail)". docs/OUTCOME_PREDICTION.md Level 3
identifies exactly this kind of rubric (clarity, honesty, faithfulness) as
the standard evaluation approach for generated legal-outcome text in the
research literature, complementary to -- not a replacement for --
classification metrics.

Rubric (each 1-5, GPT-4o as judge):
- clarity: is the outcome and its meaning stated in plain, understandable language?
- uncertainty_honesty: does it avoid overclaiming confidence the model doesn't have?
- faithfulness: does it only state things actually supported by the model's own
  output (label, probabilities, accuracy metrics) -- no fabricated detail?
- completeness: does it give enough context (alternatives, confidence, disclaimer)
  for the reader to form an informed view, not just a bare label?
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd

from agent.case_agent import build_report
from evaluation.metrics import temporal_split
from training.llm_classify import _client
from training.train_baseline import DATA_PATH, DATE_COLUMN, LABEL_COLUMN, TEXT_COLUMN

MODEL = "gpt-4o-mini"

JUDGE_SYSTEM_PROMPT = """You are evaluating the quality of an AI legal-outcome-prediction \
agent's English-language summary, NOT whether its prediction was correct (a separate metric \
already measures that). Rate the summary on 4 dimensions, each 1-5:

- clarity: is the predicted outcome and what it means stated in plain, understandable language?
- uncertainty_honesty: does it avoid overclaiming confidence, and clearly state the model's own \
limitations/accuracy rather than presenting a bare confident verdict?
- faithfulness: does it only state things actually grounded in the data given to you (the \
predicted label, probabilities, confidence score, model accuracy) -- no fabricated detail, no \
claims about the case beyond what was given?
- completeness: does it give the reader enough context (alternative outcomes considered, a \
confidence indication, an appropriate disclaimer) to form an informed view, not just a bare label?

Reply with ONLY a JSON object, no other text: {"clarity": <1-5>, "uncertainty_honesty": <1-5>, \
"faithfulness": <1-5>, "completeness": <1-5>, "note": "<one short sentence on the biggest issue, \
or 'none' if no issues>"}"""


def judge_one(client, summary: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Agent summary to evaluate:\n{summary}"},
        ],
        max_tokens=200,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    return json.loads(raw)


def run(sample_size: int = 30, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    _, test_df = temporal_split(df, DATE_COLUMN, test_fraction=0.2)
    sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=seed)

    client = _client()
    results = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        case = {"text": row[TEXT_COLUMN], "court": row.get("court", "unknown")}
        report = build_report(case)
        summary = report["english_summary"]
        try:
            scores = judge_one(client, summary)
        except Exception as e:
            print(f"  [{i}] JUDGE ERROR: {type(e).__name__}: {e}", flush=True)
            scores = {"clarity": None, "uncertainty_honesty": None, "faithfulness": None, "completeness": None, "note": str(e)}
        scores["predicted_label"] = report["label"]
        scores["true_label"] = row[LABEL_COLUMN]
        scores["correct"] = report["label"] == row[LABEL_COLUMN]
        results.append(scores)
        if i % 5 == 0 or i == len(sample):
            print(f"  {i}/{len(sample)}", flush=True)
        time.sleep(0.3)
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    results = run(sample_size=n)
    numeric_cols = ["clarity", "uncertainty_honesty", "faithfulness", "completeness"]
    means = results[numeric_cols].mean()
    overall = means.mean()
    print(f"\nn={len(results)}")
    print(means)
    print(f"overall mean (0-5 scale): {overall:.2f}  ({overall/5:.1%})")
    print(f"classification accuracy on same sample: {results['correct'].mean():.3f}")
    results.to_csv(f"registry/_llm_judge_sample_{n}.csv", index=False)
