"""Phase 3 (docs/MODEL_IMPROVEMENTS.md): LLM few-shot classification.

Local methods (TF-IDF, fine-tuned InLegalBERT, and their ensemble --
docs/MODEL_RESULTS.md Runs 1-13) plateaued around 0.52 accuracy / 0.45
macro-F1 after extensive tuning. This tests the one architecturally
different approach flagged but not yet tried: prompting a large
general-purpose LLM with a handful of labeled examples per class, per
Phase 3's cited research (57.3% -> 73.2% accuracy from zero-shot to
few-shot in one legal-judgment-prediction study).

Cost control: documents run 10,000-50,000+ characters, so this truncates
each example (few-shot exemplars AND the query case) to the last WINDOW
characters -- consistent with this project's own extraction-heuristic
rationale (Singapore judgments state their disposition near the end).
Uses gpt-4o-mini, run first on a modest SAMPLE_SIZE subset before scaling
to the full test set, to get a cheap directional read.
"""
from __future__ import annotations

import os
import re
import time

import pandas as pd
from openai import OpenAI

from evaluation.metrics import temporal_split
from labels import DISPOSITION_MEANINGS as LABEL_DEFS
from training.train_baseline import DATA_PATH, DATE_COLUMN, LABEL_COLUMN, TEXT_COLUMN

WINDOW = 6000  # chars per example, few-shot exemplars and query alike -- matches
# extract_outcome_heuristic's own disposition-window convention (docs/MODEL.md §3)
EXAMPLES_PER_CLASS = 3
MODEL = os.environ.get("LLM_CLASSIFY_MODEL", "gpt-4o-mini")

LABELS = list(LABEL_DEFS)


def _client() -> OpenAI:
    # OPENAI_API_KEY in this environment is expired/invalid (confirmed via a
    # 401 test call); MATH_PROJECT_API_KEY_OPENAI is a working key the user
    # explicitly approved reusing for this unrelated project.
    key = os.environ.get("MATH_PROJECT_API_KEY_OPENAI") or os.environ.get("OPENAI_API_KEY")
    # Explicit timeout + capped retries -- a hung/retrying request with no
    # timeout previously stalled a 100-sample run for 40+ minutes with zero
    # output or CPU activity (confirmed via `Get-Process`), so don't rely on
    # the SDK's defaults here.
    return OpenAI(api_key=key, timeout=60.0, max_retries=2)


def _tail(text: str, window: int = WINDOW) -> str:
    return text[-window:] if isinstance(text, str) else ""


def build_few_shot_examples(train_df: pd.DataFrame, seed: int = 0) -> list[dict]:
    examples = []
    for label in LABELS:
        subset = train_df[train_df[LABEL_COLUMN] == label]
        sample = subset.sample(n=min(EXAMPLES_PER_CLASS, len(subset)), random_state=seed)
        for _, row in sample.iterrows():
            examples.append({"text": _tail(row[TEXT_COLUMN]), "label": label})
    return examples


class RetrievalFewShotSelector:
    """Per-query few-shot selection: for each class, picks the
    EXAMPLES_PER_CLASS training examples most textually similar (TF-IDF
    cosine) to the current query, instead of a fixed random set for every
    query. Standard retrieval-augmented in-context-learning technique --
    not yet tried in this project. Motivated directly by `withdrawn`'s
    persistent 0% recall across every fixed-few-shot and ensemble run so
    far (docs/MODEL_RESULTS.md): a random `withdrawn` exemplar may look
    nothing like the query case's actual withdrawal phrasing, whereas the
    single most similar training `withdrawn` case is far more likely to
    share it.
    """

    def __init__(self, train_df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.train_df = train_df.reset_index(drop=True)
        self.texts = [_tail(t) for t in self.train_df[TEXT_COLUMN]]
        # Smaller vocab than the main model's 50k -- this is just for
        # nearest-neighbor retrieval, not a classifier in its own right.
        self.vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.label_indices = {
            label: self.train_df.index[self.train_df[LABEL_COLUMN] == label].tolist() for label in LABELS
        }

    def select(self, query_text: str, examples_per_class: int = EXAMPLES_PER_CLASS) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([_tail(query_text)])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        examples = []
        for label in LABELS:
            idxs = self.label_indices[label]
            if not idxs:
                continue
            ranked = sorted(idxs, key=lambda i: -sims[i])[:examples_per_class]
            for i in ranked:
                examples.append({"text": self.texts[i], "label": label})
        return examples


def _build_prompt(few_shot: list[dict], query_text: str) -> list[dict]:
    label_list = "\n".join(f"- {lbl}: {desc}" for lbl, desc in LABEL_DEFS.items())
    system = (
        "You are an expert Singapore legal analyst classifying the disposition "
        "of court judgments into exactly one of these 5 categories:\n" + label_list + "\n\n"
        "You will see the final portion of several past judgments (where the "
        "disposition is stated), each labeled with its true category, followed "
        "by one new judgment excerpt to classify. Some categories are easy to "
        "conflate: `struck_out` is a procedural disposal (the pleading/action is "
        "struck out), not a merits ruling -- do not call it `dismissed` just "
        "because the moving party lost. `withdrawn` means the moving party "
        "pulled out before any ruling -- there is no winner on the merits.\n\n"
        "First reason briefly (2-3 sentences) about which disposition language "
        "the excerpt actually uses, then on a new final line write exactly:\n"
        "FINAL: <category>"
    )
    messages = [{"role": "system", "content": system}]
    for ex in few_shot:
        messages.append({"role": "user", "content": f"Judgment excerpt:\n{ex['text']}"})
        messages.append({"role": "assistant", "content": f"This excerpt's disposition language matches {ex['label']}.\nFINAL: {ex['label']}"})
    messages.append({"role": "user", "content": f"Judgment excerpt:\n{query_text}"})
    return messages


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s")


def classify_one(client: OpenAI, few_shot: list[dict], query_text: str, max_attempts: int = 4) -> str:
    from openai import RateLimitError

    messages = _build_prompt(few_shot, query_text)
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=200, temperature=0)
            break
        except RateLimitError as e:
            # This org's key is capped at 30,000 tokens/min for gpt-4o, and a
            # single call here runs ~20-25k tokens (a full few-shot prompt) --
            # close enough to the ceiling that back-to-back calls collide with
            # it routinely. Parse the API's own suggested wait, don't just
            # guess, and actually sleep for it rather than immediately
            # falling back to a wrong-answer placeholder.
            m = _RETRY_AFTER_RE.search(str(e))
            wait_s = float(m.group(1)) + 2.0 if m else 20.0
            if attempt == max_attempts - 1:
                raise
            time.sleep(wait_s)
    raw = resp.choices[0].message.content.strip().lower()
    final_line = raw.split("final:")[-1] if "final:" in raw else raw
    match = re.search(r"|".join(LABELS), final_line)
    return match.group(0) if match else "dismissed"  # fallback to majority class on unparseable output


def run(sample_size: int | None = None, seed: int = 0, retrieval: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    train_df, test_df = temporal_split(df, DATE_COLUMN, test_fraction=0.2)

    selector = RetrievalFewShotSelector(train_df) if retrieval else None
    fixed_few_shot = None if retrieval else build_few_shot_examples(train_df, seed=seed)

    if sample_size is not None and sample_size < len(test_df):
        test_sample = test_df.sample(n=sample_size, random_state=seed)
    else:
        test_sample = test_df

    client = _client()
    results = []
    n_errors = 0
    for i, (row_idx, row) in enumerate(test_sample.iterrows(), start=1):
        try:
            few_shot = selector.select(row[TEXT_COLUMN]) if retrieval else fixed_few_shot
            pred = classify_one(client, few_shot, _tail(row[TEXT_COLUMN]))
        except Exception as e:
            # Don't let one bad call (timeout, rate limit, etc.) stall or
            # abort the whole batch -- record it as an error/miss and move on.
            n_errors += 1
            print(f"  [{i}] ERROR: {type(e).__name__}: {e}", flush=True)
            pred = "dismissed"  # majority-class fallback, scored as a miss unless true label happens to match
        results.append({"row_idx": row_idx, "true": row[LABEL_COLUMN], "pred": pred})
        if i % 5 == 0 or i == len(test_sample):
            print(f"  {i}/{len(test_sample)} ({n_errors} errors so far)", flush=True)
        # gpt-4o on this key is rate-limited to 30k tokens/min and a single
        # call here runs ~20-25k tokens, so proactively pace it instead of
        # relying on reactive retry-after waits for every single call.
        time.sleep(16.0 if "gpt-4o" == MODEL else 0.2)
    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    retrieval = "--retrieval" in sys.argv
    results = run(sample_size=n, retrieval=retrieval)
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    acc = accuracy_score(results["true"], results["pred"])
    f1 = f1_score(results["true"], results["pred"], average="macro")
    print(f"\nn={len(results)} retrieval={retrieval} accuracy={acc:.3f} macro-f1={f1:.3f}")
    print(classification_report(results["true"], results["pred"]))
    suffix = "_retrieval" if retrieval else ""
    results.to_csv(f"registry/_llm_classify_sample_{n}{suffix}.csv", index=False)
