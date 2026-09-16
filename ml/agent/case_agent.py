"""An interactive agent over the trained outcome-prediction model.

This is the "agent that is able to interact with the user directly, and
answer their questions about the outcomes of cases" piece of the project
goal. Two design choices, both deliberate and documented in
docs/OUTCOME_PREDICTION.md:

1. **Output is Level 1 structured fields (disposition + ranked
   probabilities + costs_amount_bucket where extractable), turned into an
   English-sentence summary by a fixed template -- not free-form LLM
   generation.** docs/OUTCOME_PREDICTION.md explicitly recommends against
   Level 4 (open-ended narrative generation) at this project's scale,
   citing hallucination risk and regulatory precedent (FTC v. DoNotPay).
   A templated summary grounded in the model's actual predict_proba output
   can't say anything the model didn't actually predict.

2. **Question-answering is a fixed set of intents matched by keyword, not
   an LLM chat loop.** An LLM (GPT-4o) *is* available and used elsewhere in
   this project -- it's one component of the higher-accuracy ensemble tier
   (docs/MODEL_RESULTS.md Run 14c) that `ml/inference/predict.py` can
   optionally call per-prediction (`USE_LLM_ENSEMBLE=1`). But it's
   deliberately not used here for open-ended answer *generation*: this
   means the agent can only answer the question types it's been built to
   answer, but every answer is directly traceable to the model's own
   output -- there is no space for it to fabricate an answer.

Every report includes the model's own accuracy/macro-F1 from its registry
metadata, so "how confident should I be in this?" is answered honestly
with the model's actual measured performance (docs/MODEL_RESULTS.md),
never a bare verdict -- matching ARCHITECTURE.md's responsible-use note.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.predict import predict  # noqa: E402
from labels import DISPOSITION_MEANINGS  # noqa: E402
from preprocessing.clean import extract_costs_amount_bucket  # noqa: E402

_CONFIDENCE_BANDS = [
    (0.5, "high (relative to the other candidate outcomes)"),
    (0.2, "moderate"),
    (0.0, "low -- the model sees several outcomes as roughly equally plausible"),
]


def _confidence_band(confidence: float) -> str:
    for threshold, label in _CONFIDENCE_BANDS:
        if confidence >= threshold:
            return label
    return _CONFIDENCE_BANDS[-1][1]


def build_report(case: dict) -> dict:
    """Run the trained model on `case` and build a structured outcome report.

    `case` needs at least a `text` field; metadata columns the model was
    trained on (see model_info()['metrics']) are optional and default to
    "unknown". Returns the raw prediction plus a templated English summary
    -- see module docstring for why the summary is templated, not generated.
    """
    result = predict(case)
    metrics = result.get("model_metrics", {})
    accuracy = metrics.get("accuracy")
    macro_f1 = metrics.get("f1")

    result["confidence_band"] = _confidence_band(result["confidence"])
    result["disposition_meaning"] = DISPOSITION_MEANINGS.get(result["label"], "unknown disposition")
    result["model_tier"] = (
        "local+LLM ensemble (USE_LLM_ENSEMBLE=1)" if os.environ.get("USE_LLM_ENSEMBLE") == "1" else "local ensemble"
    )
    # Only meaningful on already-decided judgment text (docs/MODEL.md) -- on
    # genuinely pre-decision text this correctly returns "none" most of the
    # time, since no costs order has been made yet to describe.
    result["costs_amount_bucket"] = extract_costs_amount_bucket(case.get("text", ""))
    result["english_summary"] = _render_summary(result, accuracy, macro_f1)
    return result


def _render_summary(result: dict, accuracy: float | None, macro_f1: float | None) -> str:
    top = result["ranked_probabilities"][0]
    alternatives = result["ranked_probabilities"][1:4]  # up to 3 runners-up

    lines = [
        f"Based on patterns in similar past cases, the model's single most likely "
        f"predicted outcome is {top['label'].upper()} (estimated probability "
        f"{top['probability']:.0%}) -- meaning {result['disposition_meaning']}.",
        f"Model confidence in this being the top outcome: {result['confidence_band']} "
        f"(raw confidence score {result['confidence']:.2f} out of 1.0).",
    ]
    if alternatives:
        alt_str = ", ".join(f"{a['label'].upper()} ({a['probability']:.0%})" for a in alternatives)
        lines.append(f"Other outcomes the model considered plausible: {alt_str}.")

    if accuracy is not None and macro_f1 is not None:
        lines.append(
            f"IMPORTANT: this model's own measured accuracy on held-out historical "
            f"cases is only about {accuracy:.0%} (macro-F1 {macro_f1:.2f}) across the "
            f"5 outcome categories. Treat this prediction as a rough statistical "
            f"signal, not a reliable forecast."
        )
    lines.append(
        "This is a statistical estimate from patterns in past cases, not legal "
        "advice, and not a guarantee of any outcome."
    )
    lines.append(f"[Model tier used: {result['model_tier']}]")
    return " ".join(lines)


# --- Question answering: fixed intents, keyword-matched --------------------
# Every branch below reads only from `report` (the model's own output) or
# static reference text (DISPOSITION_MEANINGS) -- nothing here can invent an
# answer the model didn't actually produce. See module docstring point 2.

_INTENTS: list[tuple[str, list[str]]] = [
    # Checked in order, first match wins -- more specific intents (e.g.
    # "other outcomes") must come before the generic intents they'd
    # otherwise be swallowed by (e.g. "outcome" alone), since a question
    # like "what other outcomes are possible?" contains both keywords.
    ("meaning", ["what does", "what is the meaning", "explain "]),
    ("alternatives", ["other outcome", "alternative", "else", "instead", "what other"]),
    ("confidence", ["confiden", "how sure", "certain", "reliab"]),
    ("costs", ["cost", "money", "pay", "damages", "award"]),
    ("accuracy", ["accura", "trust", "how good", "how well"]),
    ("outcome", ["outcome", "predict", "happen", "win", "lose", "result"]),
    ("help", ["help", "what can you", "what questions"]),
]

_HELP_TEXT = (
    "I can answer questions about: the predicted OUTCOME, my CONFIDENCE in it, "
    "ALTERNATIVE outcomes considered, any predicted COSTS figure, the model's "
    "own ACCURACY, or what a disposition category MEANS. Ask in plain English, "
    "e.g. 'what's the likely outcome?' or 'how confident are you?'."
)


def _match_intent(question: str) -> str | None:
    q = question.lower()
    for intent, keywords in _INTENTS:
        if any(kw in q for kw in keywords):
            return intent
    return None


def answer_question(report: dict, question: str) -> str:
    """Answer one question about `report` (from build_report()). Only
    answers the fixed intents in _INTENTS -- an unmatched question gets the
    help text, never a guessed or fabricated answer.
    """
    intent = _match_intent(question)

    if intent == "outcome":
        top = report["ranked_probabilities"][0]
        return (
            f"Most likely predicted outcome: {top['label'].upper()} "
            f"({top['probability']:.0%} estimated probability) -- "
            f"{report['disposition_meaning']}."
        )
    if intent == "confidence":
        return (
            f"Confidence is {report['confidence_band']} "
            f"(raw score {report['confidence']:.2f}/1.0, where 0 means no better "
            f"than guessing among the categories and 1 means certain)."
        )
    if intent == "alternatives":
        alts = report["ranked_probabilities"][1:]
        if not alts:
            return "No other outcomes were considered plausible for this case."
        alt_str = "; ".join(f"{a['label'].upper()} ({a['probability']:.0%})" for a in alts)
        return f"Other outcomes considered, ranked by probability: {alt_str}."
    if intent == "costs":
        bucket = report.get("costs_amount_bucket")
        if bucket is None:
            return (
                "I don't have a predicted costs figure for this case. Costs "
                "extraction currently only works on already-decided judgment "
                "text (docs/MODEL.md) -- it isn't a pre-decision forecast yet."
            )
        if bucket == "none":
            return "No costs figure was found in the text provided."
        return f"Extracted costs order falls in the '{bucket}' range."
    if intent == "accuracy":
        metrics = report.get("model_metrics", {})
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")
        if acc is None:
            return "Model accuracy metrics aren't available for this model version."
        return (
            f"This model's measured accuracy on held-out historical cases is "
            f"{acc:.0%} (macro-F1 {f1:.2f}) across 5 outcome categories -- treat "
            f"any single prediction as a rough signal, not a reliable forecast."
        )
    if intent == "meaning":
        for label, meaning in DISPOSITION_MEANINGS.items():
            if label in question.lower() or label.replace("_", " ") in question.lower():
                return f"{label.upper()}: {meaning}."
        return (
            "Disposition categories: "
            + "; ".join(f"{k.upper()} = {v}" for k, v in DISPOSITION_MEANINGS.items())
        )
    return _HELP_TEXT


def run_cli() -> None:  # pragma: no cover -- interactive, not unit tested
    import argparse
    parser = argparse.ArgumentParser(description="Interactive legal-outcome-prediction agent")
    parser.add_argument("text_file", help="Path to a text file with the case's facts/text")
    parser.add_argument("--court", default="unknown")
    parser.add_argument("--case-type", default="unknown")
    parser.add_argument("--represented", default="unknown")
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use the higher-accuracy local+GPT-4o ensemble tier (docs/MODEL_RESULTS.md Run 14c: "
             "0.568 accuracy vs 0.475 for the local-only default). Costs a real, metered OpenAI "
             "API call (~$0.05-0.06) and ~16s of extra latency per prediction.",
    )
    args = parser.parse_args()
    if args.use_llm:
        os.environ["USE_LLM_ENSEMBLE"] = "1"

    text = Path(args.text_file).read_text(encoding="utf-8")
    case = {
        "text": text,
        "court": args.court,
        "case_type": args.case_type,
        "represented": args.represented,
    }
    report = build_report(case)
    print(report["english_summary"])
    print()
    print(_HELP_TEXT)
    while True:
        try:
            question = input("\nAsk a question (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        print(answer_question(report, question))


if __name__ == "__main__":  # pragma: no cover
    run_cli()
