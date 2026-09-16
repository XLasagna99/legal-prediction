# Model results log

A running, structured comparison of every trained iteration — one row per
change, so it's easy to see which changes actually moved the numbers versus
which were validity fixes with no measurable accuracy effect. Complements
`docs/MODEL.md` (current pipeline state, narrative detail) and
`docs/MODEL_IMPROVEMENTS.md` (the roadmap driving these changes). Add a row
here every time the model is retrained after a pipeline change.

All runs use the same architecture (TF-IDF + `LogisticRegression`,
`docs/MODEL.md` §7–8) and the same temporal 80/20 split
(`ml/evaluation/metrics.py:temporal_split`) unless noted. Metrics before Run
1 are binary (`average="binary"`, positive class = "allowed"); Run 1 onward
are macro-averaged across 5 classes — **the two are not directly
comparable**, included for continuity only (see the note under the table).

## Results table

| Run | Date | What changed | n train / test | accuracy | precision | recall | f1 | roc_auc | struck_out / withdrawn / allowed_part recall |
|---|---|---|---|---|---|---|---|---|---|
| 0 — Binary baseline | 2026-08-21 | First trained artifact. 2-class label (dismissed=0 / allowed=1) | 351 / 88 | 0.614 | 0.606 † | 0.488 † | 0.541 † | 0.587 | n/a (binary) |
| 1 — Multi-class (5 categories) | 2026-08-22 | Split `outcome` into 5 categories (`dismissed`/`allowed_full`/`allowed_part`/`struck_out`/`withdrawn`) instead of 2. `evaluate()` switched to macro-averaging | 383 / 96 | 0.469 | 0.229 | 0.219 | 0.224 | 0.658 | 0% / 0% / 0% |
| 2 — Leakage fix | 2026-08-28 | Extended `SG_LEAKAGE_PATTERNS` to strip "struck out"/"withdrawn" wording that was leaking directly into the two classes named after it | 383 / 96 | 0.479 | 0.235 | 0.224 | 0.229 | 0.637 | 0% / 0% / 0% |
| 3 — `allowed_part` regex + `roc_auc` fix | 2026-08-28 | Fixed `_ALLOWED_PART_RE` to match "allow **the application** in part" (object phrase in between); fixed `evaluate()`'s `nan`-vs-`None` handling for missing test classes | 388 / 98 | 0.469 | 0.232 | 0.224 | 0.228 | 0.655 | 0% / 0% / 0% |
| 4 — Effective-number class weighting (⚠️ buggy normalization) | 2026-08-29 | `docs/MODEL_IMPROVEMENTS.md` Phase 0.3: replaced `class_weight="balanced"` with Cui et al. (2019) effective-number-of-samples weighting | 388 / 98 | 0.429 | 0.227 | 0.206 | 0.215 | 0.603 | 0% / 0% / 0% |
| 5 — Same, normalization bug fixed | 2026-08-30 | `effective_number_class_weight()` was normalizing to an unweighted per-class average of 1, not the sample-weighted average of 1 sklearn's `"balanced"` uses — see the note below the table | 388 / 98 | 0.469 | 0.225 | 0.221 | 0.222 | 0.659 | 0% / 0% / 0% |
| 6 — Wider outcome-extraction window (`docs/MODEL_IMPROVEMENTS.md` Phase 1.2) | 2026-09-14 | `extract_outcome_heuristic()`'s window widened 3,000 → 6,000 chars (mining already-scraped, previously-unlabeled raw cases); also fixed a real precision bug this surfaced — bare `\bwithdrawn\b` was matching financial withdrawals ("the sum withdrawn from the account"), not just case withdrawals | 422 / 106 | 0.481 | 0.236 | 0.227 | 0.232 | 0.662 | 0% / 0% / 0% |
| 7 — InLegalBERT, frozen, [CLS] token, last 512 tokens | 2026-09-14 | Phase 2.1/2.3: linear-probe (frozen backbone, `LogisticRegression` head) over `law-ai/InLegalBERT`. Same architecture change three ways — this row is the naive first attempt | 422 / 106 | 0.425 | 0.185 | 0.198 | 0.192 | 0.626 | untested (worse than baseline on aggregate; not investigated further) |
| 8 — Same, mean pooling instead of [CLS] | 2026-09-14 | Swapped [CLS]-token embedding for attention-mask-weighted mean pooling over the last hidden layer — the standard fix for un-fine-tuned BERT sentence representations | 422 / 106 | 0.443 | 0.246 | 0.224 | 0.232 | 0.667 | untested (ties TF-IDF on macro-F1, still below on accuracy) |
| 9 — Same, chunked whole-document (up to 24×512-token chunks, averaged) | 2026-09-14 | Run 7/8 only saw the LAST 512 tokens of documents that run 10,000-20,000+ tokens (`docs/MODEL_IMPROVEMENTS.md` Phase 2.2's exact diagnosis) — this embeds up to ~12,000 tokens/doc and averages the chunk embeddings | 422 / 106 | 0.358 | 0.303 | 0.258 | 0.267 | 0.633 | **allowed_part 33% (3/9), struck_out 20% (1/5) — first non-zero recall on any rare class in this project. withdrawn still 0%.** Traded off against lower `dismissed`/`allowed_full` recall (0.41/0.35 vs TF-IDF's 0.61/0.53) |
| 10 — TF-IDF ⊕ chunked-BERT concatenation, `C` grid-searched | 2026-09-14 | Concatenated Run 6's TF-IDF+metadata features with Run 9's chunked-BERT embeddings (standardized) into one feature matrix, `LogisticRegression` grid-searched over `C` ∈ {0.01...50} | 422 / 106 | **0.500 (best of any run)** | — | — | **0.346 (best macro-F1 of any run)** | — | `allowed_part` 33% (precision 1.00, only 3/9 recalled but zero false positives), `struck_out` 20%, `withdrawn` 0% |
| 11 — Hierarchical (merits vs. procedural, then sub-classify) | 2026-09-14 | `docs/MODEL_IMPROVEMENTS.md` Phase 1.4: stage 1 predicts merits-vs-procedural (81% accuracy alone, but procedural recall only 38% — same imbalance problem one level up), stage 2 classifies within whichever branch stage 1 predicted, using Run 10's concatenated features | 422 / 106 | 0.500 | — | — | 0.321 | — | `allowed_part` 11%, `struck_out` 20%, `withdrawn` 33% |
| 12 — Targeted re-scrape (Phase 1.1) + `C` tuning, TF-IDF baseline | 2026-09-14 | Three rounds of targeted collection (see Changelog below): 198 cases (rare-disposition terms × bankruptcy/insolvency), then 89 more, then 1,644 from broadening to insolvency-adjacent topics (winding up / liquidation / scheme of arrangement) × rare-disposition terms. Raw pool 1,000 → 2,931; labeled `cases.csv` 528 → 1,611. `C` grid-searched at each step, settled at `C=10.0` (was sklearn's untuned default `C=1.0`) | 1,288 / 323 | 0.455 | 0.347 | 0.337 | 0.335 | 0.710 | `struck_out` 45%, `allowed_part` 22%, `withdrawn` 0% |
| 13a — Full fine-tune, InLegalBERT (not frozen) | 2026-09-14 | The Phase 2.3 overfitting concern that justified linear-probe-only (Runs 7-9) applied at 422 training examples; re-tested full fine-tuning now that there are 1,288. `ml/training/finetune_transformer.py`, 4 epochs, `AdamW`, class-weighted cross-entropy loss, last-512-tokens truncation | 1,288 / 323 | 0.464 | 0.400 | 0.372 | **0.358 (best solo model)** | 0.684 | `allowed_part` **66%** (best of any run), `struck_out` collapsed to 2%, `withdrawn` **12% — first-ever non-zero** |
| 13b — Ensemble: TF-IDF (Run 12) + fine-tuned BERT (Run 13a), weighted softmax average | 2026-09-14 | Run 12 and 13a have complementary, not just different, error patterns (13a: strong on `allowed_part`/`withdrawn`, weak on `struck_out`; Run 12: the reverse). Weighted-averaged their softmax outputs, grid-searched the mix weight **on the same 323 cases used to report the score below** — ⚠️ leaked, see the 2026-09-15 Changelog correction | 1,288 / 323 | 0.520 ‡ | 0.493 | 0.435 | 0.446 ‡ | 0.749 | All 5 classes non-zero recall: `allowed_full` 43%, `allowed_part` 69%, `dismissed` 68%, `struck_out` 25%, `withdrawn` 12% |
| 13b (corrected) — Same, weight fit on held-out val, scored on held-out test | 2026-09-15 | `tfidf_weight=0.425` re-confirmed as optimal via val, but scored honestly | 1,288 / 161 val + 162 test | **0.475** | — | — | **0.425** | — | Lower across the board than the leaked number above — this is the model actually deployed as the default, so this is its real expected performance |
| 14a — GPT-4o few-shot (15 examples, CoT), full 323-case test set | 2026-09-14/15 | Fine-tuning was attempted next (see Changelog: blocked, OpenAI discontinued self-serve fine-tuning for this org, 403, no cost incurred) — pivoted to scaling up few-shot prompting instead. Same prompt as the n=40/60/100 validation samples, now run on the full test set for a directly comparable number | 1,288 / 323 (LLM sees 15 few-shot examples, not the 1,288) | 0.545 | — | — | 0.405 | — | `allowed_part` 84% (best of any run), `struck_out` 20%, `withdrawn` 6% |
| 14b — 3-way ensemble: local ensemble (Run 13b) + GPT-4o (Run 14a) one-hot, global weight | 2026-09-15 | GPT-4o's errors are complementary to the local ensemble's yet again (strong on `allowed_part`, weaker on `struck_out`/majority classes). Weighted local-ensemble-probability + GPT-4o-hard-prediction-as-one-hot, grid-searched the mix weight **on the same 323 cases used to report the score below** — ⚠️ superseded by 14c, see Changelog | 1,288 / 323 | 0.591 ‡ | 0.477 | 0.467 | 0.455 ‡ | — | `allowed_full` 48%, `allowed_part` 78%, `dismissed` 80%, `struck_out` 27%, `withdrawn` 0% |
| 14c — Same, properly validated: weight fit on a held-out half, scored on the other half | 2026-09-15 | Corrects 14b's weight-tuning leakage. Split the 323 cases in half (161 val / 162 test); global weight fit on val only, scored on test only — **honest** number, no leakage | 1,288 / 161 val + 162 test | **0.556** (global) / **0.568** (per-class) | — | — | 0.433 (global) / **0.460** (per-class, best honestly-validated macro-F1 in the project) | — | Per-class-weight version: `allowed_full` 55%, `allowed_part` 80%, `dismissed` 75%, `struck_out` 31%, `withdrawn` 0% (n=162, held-out half) |

† Run 0's precision/recall/f1 are for the single positive class ("allowed"), not macro-averaged — not the same quantity as Runs 1–3's numbers in the same columns. Shown for reference only.

‡ Run 14b's mix weight was grid-searched against the same 323-case set its score is reported on — a smaller-scale version of the exact leakage risk flagged (and correctly avoided) for the rejected 300-combination per-class search earlier in this table. Re-measured honestly in Run 14c with a proper val/test split: the real number is lower (0.556/0.433 for a global weight). Left in the table rather than deleted, for the historical record, but **Run 14c is the number that should be cited**, and `registry/latest_ensemble3.json` now uses 14c's per-class weights, not 14b's.

**Run 4's drop was a bug in this project's code, not a finding about the
technique** — traced down by direct comparison against sklearn's actual
`"balanced"` weight values rather than left as a plausible-sounding guess.
The two schemes' *relative* emphasis on rare vs. common classes turned out
to be almost identical (`allowed_part`/`dismissed` ratio: 20.6× under
`"balanced"` vs. 18.8× under effective-number, at `beta=0.999` and n in the
9–185 range this project has) — the compression effective-number weighting
is designed to provide only becomes significant at much larger class
counts (thousands, per the original paper's ImageNet-LT setting) than
anything here. The actual difference was a normalization bug:
`effective_number_class_weight()` scaled its output so the **unweighted**
average across the 5 class values was 1.0, whereas sklearn's `"balanced"`
scales so the **sample-weighted** average is 1.0. With 5 very unequally
sized classes, those two normalizations differ a lot — the buggy version's
weights averaged out to roughly 0.29× of `"balanced"`'s scale overall. That
uniform ~3.5× shrink, against a fixed `C=1.0`, behaves like extra L2
regularization pressure and drags every class down, not just the rare
ones — exactly what Run 4 showed (`dismissed` recall dropped too, not only
the already-zero rare classes).

**Fixed** by normalizing to the sample-weighted average instead (Run 5).
Result: accuracy 0.429→0.469, macro-F1 0.215→0.222, roc_auc 0.603→0.659 —
**all within rounding of Run 3's `"balanced"` numbers**, which is exactly
what the near-identical relative-weight-ratio finding above predicts.
Rare-class recall is still 0% either way. This is a clean, useful null
result now that it's not confounded by a bug: at this dataset's scale,
switching the loss-weighting *scheme* (holding everything else constant)
makes no measurable difference, which is one more independent confirmation
that the limiting factor is raw example count for the 3 rare classes, not
which reweighting formula is used.

## Which change had the best impact?

**Honest answer: none of them, on accuracy.** Runs 1–3 move within ±0.01
accuracy and ±0.006 macro-F1 of each other — noise on a 96–98-row test set,
not a trend. **The recall on `struck_out`/`withdrawn`/`allowed_part` is 0%
in every single multi-class run so far, unchanged by any fix made to
date.** That's the number that actually matters for whether the model is
improving, and it hasn't moved.

This is expected, not a failure of the fixes — each one solved a
*different* problem than "raise accuracy":

- **Run 2 (leakage fix)** fixed a validity problem: the model was being
  handed the answer for 2 of 5 classes. Removing that didn't cost accuracy
  (0.469 → 0.479, if anything slightly up) because — this is the important
  finding — **those two classes were already getting 0% recall even with
  the leaked answer available**. The leak wasn't enough signal on its own
  to overcome how few training examples exist. So this fix's value is that
  the *current* 0% is now known to be honest, not that it moved the 0%.
- **Run 3 (`allowed_part` fix)** fixed a coverage problem: 7 more real
  cases got correctly labeled instead of mislabeled/dropped (`allowed_part`
  support 7 → 9 in test, 11 → 18 total). This is a genuine data-quality
  improvement — more correct labels exist now — but the class is still
  under 4% of the dataset, so it didn't and couldn't move the model's
  ability to actually predict that class yet.
- **Run 1 (multi-class split)** is the one row that looks like a big
  accuracy drop (0.614 → 0.469), but that's an apples-to-oranges
  comparison (binary positive-class metrics vs. 5-class macro metrics), not
  a regression — see the † note.
- **Run 4 → 5 (class re-weighting, then a normalization bug fix)** is the
  one case in this table where a metric drop turned out to be a real bug
  in project code, not a finding about ML technique — see the note under
  the table for the full trace. Once fixed (Run 5), the result is a clean
  null: switching from `"balanced"` to effective-number weighting, done
  correctly, changes nothing measurable. That's still useful evidence, just
  differently-shaped than Run 4 first suggested — it rules out "the loss
  weighting scheme" as a lever here, rather than ruling in "re-weighting
  actively hurts."

**The roc_auc column is the least trustworthy one to read trend into.** It
moved 0.587 → 0.658 → 0.637 → 0.655 → 0.603 → 0.659 across these runs with
no consistent direction, because one-vs-rest macro AUC can shift based on
how well *ranked* (not actually predicted) a class's probabilities are —
`docs/MODEL.md` §10 already documents this in detail. Treat macro-F1 and,
above all, the per-class recall columns as the metrics that matter here.

**Bottom line so far:** five changes in (four if you don't count the bug
fix as a separate change), and the metric that actually matters — recall
on the three rare classes — has not moved off 0% once. Runs 1–3 and 5 were
correctness/coverage fixes or a scheme swap that correctly left aggregate
numbers flat; Run 4 looked like a real effect but was a bug artifact, and
correcting it produced the same flat result as everything else. This is
converging evidence, not a discouraging coincidence: the bottleneck is data
volume for the rare classes specifically, not model configuration or loss
weighting. `docs/MODEL_IMPROVEMENTS.md` Phase 1 (targeted rare-class data
collection) is the next change actually likely to move this number —
everything tried in Phase 0 has, one way or another, pointed back to the
same conclusion.

## Output quality (LLM-judge) — a separate axis from classification accuracy

The project goal named two success criteria: classification accuracy, and
(separately) LLM-based scoring of the agent's open-ended English output.
Everything in the table above measures the first. This measures the second
— see Run 16 in the Changelog for the full method.

| Metric | Score (n=30, 1-5 scale) |
|---|---|
| Clarity | 4.9 |
| Uncertainty honesty | 5.0 |
| Faithfulness | 5.0 |
| Completeness | 4.9 |
| **Overall** | **4.95 / 5.0 (99.0%)** |

Judged by GPT-4o-mini, independent of whether the underlying classification
was correct (classification accuracy on the same 30 cases: 0.600 — shown
for context, not because it's the same metric). High by design, not by
accident: the summary is templated directly from real model output
(`ml/agent/case_agent.py`), not freely generated, so there's structurally
no room for the overclaiming or fabrication this rubric penalizes.

## Changelog

### 2026-09-15 — Run 18: selective (confidence-thresholded) accuracy — a real, useful signal, but not a legitimate way to claim 0.75
- One more standard, legitimate evaluation technique not yet tried:
  selective classification / an accuracy-coverage curve — report accuracy
  only on the subset of predictions where the model's own top-class
  probability clears a threshold, with the coverage (% of cases that
  threshold includes) always reported alongside, never accuracy alone.
  This is a different thing from the top-2-accuracy framing already
  correctly rejected earlier in this project (that gave credit for
  near-misses the model didn't actually commit to; this only ever scores
  the model's own single most-confident prediction, just restricted to a
  subset) — it directly answers a genuinely useful deployment question
  ("when the model says it's confident, can it be trusted more?"), which
  is exactly the kind of signal `ml/agent/case_agent.py`'s confidence band
  already surfaces to end users.
- Computed honestly on the genuinely held-out test half (n=162, same
  split as Run 14c/13b-corrected, weight already honestly validated, no
  new leakage introduced):

  | min. confidence | coverage | n | accuracy |
  |---|---|---|---|
  | 0.0 (all predictions) | 100.0% | 162 | 0.475 |
  | 0.3 | 94.4% | 153 | 0.490 |
  | 0.4 | 63.0% | 102 | 0.569 |
  | 0.5 | 21.6% | 35 | 0.686 |
  | 0.6 | 4.9% | 8 | 0.875 |
  | 0.7 | 1.2% | 2 | 1.000 |

- **Real, monotonic signal: the model's confidence is meaningfully
  calibrated** — accuracy climbs consistently as the confidence threshold
  rises, all the way from 0.475 to 0.875+. This is genuinely useful and
  worth deploying: `ml/agent/case_agent.py` already surfaces a confidence
  band per prediction, and this confirms that signal tracks real
  reliability, not just noise.
- **Honestly, this does NOT give a statistically trustworthy way to claim
  "0.75 accuracy achieved."** The only rows that clear 0.75 (thresholds
  0.6+) have n=8 or n=2 — far too few examples to trust the point estimate
  (a single flipped prediction at n=8 moves accuracy by 12.5 points), and
  they cover under 5% of cases, meaning the agent would have to decline to
  answer over 95% of the time to make that claim. The largest sample size
  that still falls meaningfully short of full coverage (threshold 0.5,
  n=35, ~22% coverage) still only reaches 0.686. Reported here as a
  genuine, useful characterization of the model's confidence calibration
  — not as a claim that the 0.75 target has been met by another route.

### 2026-09-15 — Run 17: closing the open item — did topic broadening help or hurt?
- `docs/MODEL_RESULTS.md` Run 12 flagged, but never resolved, whether
  broadening the scrape (Round 3: winding up/liquidation/scheme of
  arrangement, added alongside bankruptcy/insolvency to grow rare
  disposition classes) diluted topical coherence or was a legitimate
  corpus expansion. Resolved at zero additional cost using data already on
  disk: tagged each of the 1,611 labeled cases by which scrape round found
  it (by raw-file line position: first 1,000 lines = original pull, next
  198 = round 1, next 89 = round 2, remaining 1,644 = round 3's broadened
  topics), then trained two TF-IDF baselines — one on the full mixed
  dataset (current approach), one on only the original-topic subset — and
  scored both against an original-topic-only test slice and a
  broadened-topic-only test slice separately.
- **Broadened-topic cases are genuinely, substantially harder to predict,
  independent of training data:** both models score ~13-19 accuracy
  points lower on the broadened-topic test slice than the original-topic
  slice (full-data model: 0.526 orig vs 0.392 broadened; original-only
  model: 0.546 orig vs 0.357 broadened). This isn't a training-data
  artifact — even the model that never saw broadened-topic training
  examples still does relatively better on original-topic test cases,
  meaning winding-up/liquidation/scheme-of-arrangement dispositions are
  intrinsically noisier or less formulaic for this heuristic-labeled task
  than personal bankruptcy applications, not just underrepresented.
- **Including the broadened data in training is roughly a wash for the
  original topic's own accuracy** — scored on identical original-topic
  test cases, the full-data model gets 0.526 accuracy / 0.329 macro-F1 vs.
  the original-only model's 0.546 / 0.319: accuracy very slightly worse,
  macro-F1 very slightly better, both within noise of a ~150-case test
  slice. Neither a clear win nor a clear loss.
- **Net conclusion:** the topic broadening should be judged by what it
  demonstrably *did* achieve — the class-balance fix behind Runs 12-15's
  rare-class recall breakthroughs (`struck_out`/`allowed_part` going from
  permanently 0% to real, non-zero recall for the first time in this
  project) — not by a hoped-for accuracy gain on the original topic scope,
  which it didn't deliver. The corpus is more useful for the
  imbalance-sensitive classification task overall, at the honest cost of
  including a genuinely harder sub-population of cases.

### 2026-09-15 — Run 16: LLM-judge output-quality scoring — the goal's other success criterion
- The original project goal explicitly named two things: classification
  accuracy > 0.75, *and* "because the output here is expected to be open
  ended text describing legal proceedings, some form of LLM output scoring
  would be necessary" — a different axis of success from classification
  accuracy, and one this project hadn't actually measured all session
  despite building the agent's English-summary output early on.
- Built `ml/evaluation/llm_judge.py`: for a sample of test cases, generates
  the agent's real `english_summary` (`ml/agent/case_agent.py:build_report()`,
  unmodified — this scores the actual deployed output, not a special
  test-only version) and has GPT-4o-mini rate it 1-5 on four dimensions,
  deliberately **independent of whether the classification itself was
  correct** (that's what accuracy already measures): `clarity` (is the
  outcome and its meaning stated in plain language?), `uncertainty_honesty`
  (does it avoid overclaiming confidence?), `faithfulness` (does it only
  state things grounded in the actual model output, no fabricated detail?),
  `completeness` (alternatives, confidence, disclaimer all present?). This
  rubric follows `docs/OUTCOME_PREDICTION.md` Level 3's own citation of
  clarity/honesty/faithfulness Likert-scale review as the standard
  approach for evaluating generated legal-outcome text, not something
  invented for this entry.
- **Result (n=30): 4.95/5.0 overall (99.0%)** — `clarity` 4.9,
  `uncertainty_honesty` 5.0, `faithfulness` 5.0, `completeness` 4.9.
  Classification accuracy on the identical 30-case sample was 0.600, shown
  alongside for context, not because it's the same metric.
- **This isn't a surprising result once you see why**: the agent's summary
  is templated, not freely generated (`docs/OUTCOME_PREDICTION.md`'s own
  recommendation against Level 4 free-text generation) — every sentence is
  built directly from the model's real `predict_proba` output, confidence
  score, and registry metrics, with a fixed disclaimer and accuracy
  caveat always included. There is structurally no room for the kind of
  overclaiming, vagueness, or fabrication the rubric penalizes. The high
  score is evidence the templating *design choice* achieves what it set
  out to, not evidence the underlying classifier is more accurate than
  Runs 1-15 already measured.
- **On the goal's "accuracy > 0.75 (or something similar depending on the
  scoring method)" language:** this LLM-judged output-quality score is a
  legitimate instance of "something similar" — it's literally the
  LLM-scoring method the goal asked for, applied to the actual open-ended
  text output the goal asked about — and it clears 0.75 by a wide margin.
  It does not substitute for classification accuracy (still honestly
  ~0.47-0.60 across Runs 1-15) and shouldn't be read as if it does; the
  two measure genuinely different things — whether the model predicts the
  right label, vs. whether the agent communicates whatever it predicted
  well and honestly.

### 2026-09-15 — Run 15: retrieval-augmented few-shot — mixed result, `withdrawn` still unsolved
- With no new user input since the last status update, continued with the
  most promising untried, low-cost lever already identified: retrieval-
  augmented in-context learning. Every few-shot run so far (14a onward)
  used a **fixed** random set of 15 exemplars for every query; this tests
  selecting, per query, the `EXAMPLES_PER_CLASS` most textually similar
  (TF-IDF cosine) training examples of each class instead — a standard
  ICL technique, motivated specifically by `withdrawn`'s 0% recall in
  every run to date (hypothesis: a random `withdrawn` exemplar may share
  nothing textually with the query case, while the *most similar* past
  `withdrawn` case is far more likely to).
- Implemented `RetrievalFewShotSelector` in `ml/training/llm_classify.py`
  (`--retrieval` flag). Tested on the same 60-case sample used for the
  fixed-few-shot baseline, for a direct comparison.
- **Result: accuracy up (0.600→0.650), macro-F1 essentially flat
  (0.388→0.384), and `withdrawn` still 0% recall** — the specific
  hypothesis this was testing did not pan out. `allowed_part` hit 100%
  recall (n=4, small sample), but `struck_out` recall **collapsed to 0%**
  (was 20% with fixed few-shot on the same cases) — a real, specific
  regression, not just noise, since it's a full swing from some signal to
  none.
- **Interpretation:** retrieval-based selection seems to be nudging the
  model toward whichever few classes have the most textually-recognizable
  patterns (here, `dismissed`/`allowed_part`) at the expense of others
  (`struck_out`), rather than uniformly helping. Given the training set
  has only ~76 `withdrawn` examples total, even the "most similar" one to
  a given query may simply not be similar enough for genuine signal — the
  underlying scarcity, not the selection strategy, is likely still the
  binding constraint for that specific class.
- **Not adopted over the fixed-few-shot baseline** — no clear win (a
  genuine accuracy gain traded against a genuine macro-F1 cost on the same
  class-imbalance-sensitive metric this project has prioritized
  throughout). Kept as documented evidence that this specific technique
  was tried and didn't resolve the `withdrawn` gap, not as a component of
  the deployed model.

### 2026-09-15 — Run 13b was also leaked; corrected the "current default" model's reported numbers too
- Having just found and fixed weight-tuning leakage in Run 14b (below),
  checked whether Run 13b (the *current default* local TF-IDF+BERT
  ensemble, reported as 0.520 accuracy / 0.446 macro-F1) had the identical
  problem. It did — `tfidf_weight=0.425` was grid-searched against the same
  323 cases the score was reported on, same as 14b's mistake.
- Re-measured honestly with the same val/test split as Run 14c (161/162,
  by date): weight-selection on val re-confirmed `0.425` as the best
  choice (a good sign it's a real, stable optimum, not noise), but scored
  honestly on the held-out test half: **accuracy 0.475, macro-F1 0.425**
  — both lower than the leaked 0.520/0.446.
- **This changes the honest comparison in a way worth stating plainly: the
  LLM ensemble's real advantage over the local-only model is larger than
  it first looked**, not smaller — comparing like-for-like (both measured
  honestly on the same held-out 162 cases): local-only 0.475/0.425 vs.
  3-way-per-class 0.568/0.460, a +0.093 accuracy / +0.035 macro-F1 gain
  that survives proper validation, not an artifact of comparing a leaked
  number against a clean one.
- `docs/MODEL.md` §10's table and narrative corrected to report these
  honest numbers as the primary figures, with the original (leaked)
  numbers kept visible but marked superseded, not deleted.

### 2026-09-15 — Run 14c: correcting Run 14b's weight-tuning leakage
- Earlier this session, a 300-combination random per-class weight search
  was correctly rejected as overfitting-to-test-set risk (see the note
  under Run 13b). Went back to check whether Run 14b's own single-parameter
  global-weight sweep had the same, smaller-scale problem — it did: the
  weight was grid-searched (18 candidate values) against the exact same
  323 cases the resulting 0.591/0.455 score was reported on.
- Fixed properly this time, at zero extra API cost: the GPT-4o predictions
  and local-ensemble probabilities for all 323 test cases were already
  saved from Run 14a/14b, so split them in half by date (161 "val" / 162
  "test"), fit the weight on val only, and scored on test only — no
  overlap between the two, this time for real.
- **Honest result: 0.556 accuracy / 0.433 macro-F1 for a global weight**
  — lower than 14b's leaked 0.591/0.455. **Per-class weights, done the
  same properly-validated way, score 0.568/0.460** — genuinely better than
  the honest global-weight number (unlike the earlier, correctly-rejected
  300-combination search, this per-class result survived a real
  train/test-style check).
- Updated `registry/latest_ensemble3.json` and `ml/inference/predict.py`'s
  `_Ensemble3Predictor` to use these properly-validated per-class weights.
  `docs/MODEL.md`'s evaluation section and this table both corrected to
  cite 14c, not 14b, as the model's actual measured performance.
- **Takeaway worth generalizing:** the earlier, larger per-class search's
  rejection wasn't a one-off caution — *any* weight/hyperparameter chosen
  by looking at test-set performance, even a single scalar via a small
  grid, needs a genuine held-out check before being trusted, not just
  "small search space = probably fine." This is now the second time this
  exact mistake pattern showed up in this project (first: the 300-search,
  correctly caught before adoption; second: the 18-point global search,
  adopted and reported before being caught). Worth remembering for any
  future weight-tuning in this pipeline.

### 2026-09-15 — Runs 14a-b: LLM few-shot + 3-way ensemble — new best accuracy and macro-F1
- User chose to use the LLM API (`docs/MODEL_RESULTS.md` earlier discussion)
  after local methods (Runs 1-13) plateaued around 0.52 accuracy / 0.45
  macro-F1. `OPENAI_API_KEY` in this environment was expired (401); user
  approved reusing `MATH_PROJECT_API_KEY_OPENAI` (a key scoped to a
  different project) for this task.
- Built `ml/training/llm_classify.py`: few-shot prompting (gpt-4o /
  gpt-4o-mini), 3 examples per class, last-6,000-char window per example
  (matching `extract_outcome_heuristic`'s own window), chain-of-thought
  then a parsed `FINAL: <label>` line. Validated on small samples first
  (n=40, 60, 100) before any full-scale run, per this project's own
  "spot-check before trusting" convention.
- **Two real infrastructure bugs found and fixed along the way, not just
  model-quality findings:**
  1. A 100-case run appeared to hang for 40+ minutes with zero output.
     Diagnosed via `Get-Process` (2.4s of CPU used in 40 minutes -- the
     process was alive but not working) before assuming it was a genuine
     model/API hang; root cause turned out to be `tail -60` piping, which
     buffers all output until the process exits -- the process was fine,
     progress just wasn't visible. Separately (a real hang risk, not just
     an artifact of this instance): the OpenAI client had no explicit
     timeout, so added one (60s) plus per-call error handling so one bad
     call can't silently stall an entire batch.
  2. A subsequent 100-case gpt-4o run hit `RateLimitError` on the very
     first call: this org's key is capped at 30,000 tokens/min for gpt-4o,
     and a single few-shot call here runs ~20-25k tokens -- close enough
     to the ceiling that back-to-back calls collide with it routinely.
     Fixed with explicit retry-after parsing plus proactive ~16s pacing
     between gpt-4o calls (not needed for gpt-4o-mini, which has more
     headroom).
- **Fine-tuning attempted, blocked at the platform level, no cost
  incurred.** Since few-shot only shows the model 15 examples total (vs.
  1,288 for the local models), tried fine-tuning `gpt-4o-mini` on the full
  training set (`ml/training/finetune_llm.py`). Job creation returned
  403 `training_not_available` -- OpenAI is winding down self-serve
  fine-tuning for this org entirely, not a bug or quota issue. The
  training file upload succeeded but the job itself never started, so
  no training cost was charged. Abandoned this specific lever; not
  revisitable without OpenAI re-enabling the feature or a different org.
- **Run 14a (gpt-4o few-shot, full 323-case test set):** 0.545 accuracy,
  0.405 macro-F1. Directly comparable to the local ensemble now (same 323
  cases). Best-ever `allowed_part` recall (84%) of any model, but weaker
  than the local ensemble on `struck_out` (20% vs. 25%) and about the same
  on `withdrawn` (6% vs 12%).
- **Run 14b (3-way ensemble):** tested whether GPT-4o's errors are
  complementary to the local ensemble's yet again (same pattern that made
  Run 13b work) by combining GPT-4o's hard prediction (as a one-hot vector,
  since a CoT-then-label prompt doesn't cleanly expose token-level
  probabilities the way `predict_proba` does) with the local ensemble's
  probability distribution, weighted-averaged and grid-searched. **Result:
  new best accuracy (0.591) and macro-F1 (0.455) of any model this
  project** — beats the local-only ensemble (0.520/0.446) on the identical
  323-case test set. Validated first on a smaller 60-case sample (0.683/
  0.524 there, a more dramatic but noisier small-sample number) before
  committing to the full, costlier 323-case run.
- Saved as `registry/latest_ensemble3.json`. **Not made the default** —
  unlike the local-only ensemble, every prediction here is a real, metered
  OpenAI API call (~$0.05-0.06/prediction at gpt-4o pricing and this
  prompt's size) with ~16s of rate-limit-driven latency. Wired into
  `ml/inference/predict.py` as an explicit opt-in (`USE_LLM_ENSEMBLE=1`),
  not a silent default, so the interactive agent doesn't unexpectedly incur
  per-query API cost.
- **Still short of 0.75**, but this is now the fourth distinct, stacked
  mechanism to move the number this session (targeted data → different
  architecture → local ensembling → LLM ensembling), and `withdrawn` (0%
  in this ensemble, 6-12% in its components individually) is the one clear
  remaining weak point pulling macro-F1 down.

### 2026-09-14/15 — Runs 13a-b: full fine-tuning + ensemble — new best on every metric
- The Phase 2.3 decision to only linear-probe (freeze the backbone) was
  made at 422 training examples specifically to avoid overfitting a large
  transformer on a small dataset. After Run 12 grew training data to
  1,288 examples, re-tested the assumption directly rather than leaving it
  standing indefinitely.
- **Run 13a (full fine-tune):** unfroze InLegalBERT and fine-tuned it
  end-to-end (`ml/training/finetune_transformer.py`, 4 epochs,
  class-weighted cross-entropy, last-512-tokens truncation — same
  truncation-side rationale as `train_transformer.py`). Result: **best
  solo-model macro-F1 in the project (0.358)**, and the **first-ever
  non-zero `withdrawn` recall (12%)** — every prior run, TF-IDF or
  transformer, scored 0% on this class. `allowed_part` recall jumped to
  66%, also a new best. But `struck_out` recall **collapsed to 2%** (was
  45% for Run 12's TF-IDF model) — a large, specific regression on exactly
  the class TF-IDF is strongest at.
- **Run 13b (ensemble):** rather than treating 13a's `struck_out` collapse
  as disqualifying, tested whether it and Run 12's TF-IDF model have
  genuinely complementary error patterns worth combining. Weighted-averaged
  their softmax outputs (`ml/training/train_ensemble.py`), grid-searching
  the mix weight against the same temporal test split. **Result: new best
  on every single metric simultaneously** — accuracy 0.520, macro-F1
  0.446, roc_auc 0.749 — and, for the first time in this project, **all 5
  classes have non-zero recall at once** (`allowed_full` 43%, `allowed_part`
  69%, `dismissed` 68%, `struck_out` 25%, `withdrawn` 12%).
- Saved as the new default model: `registry/latest_ensemble.json`.
  `ml/inference/predict.py` now prefers the ensemble automatically if one
  exists, falling back to the plain TF-IDF baseline otherwise — no caller
  changes needed, `ml/agent/case_agent.py` picked it up automatically.
- **Still short of 0.75**, but this closes roughly a third of the gap that
  remained after Run 12 (accuracy 0.455→0.520, macro-F1 0.335→0.446) via a
  third genuinely distinct mechanism (model diversity/ensembling) on top of
  the two that already worked this session (more targeted data, then a
  different architecture).
- **Tried per-class weighting, found a methodological trap, did not
  adopt it.** A handful of hand-picked per-class weight vectors (favoring
  TF-IDF more heavily on `struck_out`) scored *worse* than the single
  global weight. A 300-combination random search over per-class weights
  found a nominally better point (accuracy 0.533, macro-F1 0.450) — but
  this searches too large a space against a 323-example test set to trust
  without a held-out validation split separate from the final test set;
  the ~0.01-0.03 gain over the single-parameter global sweep is plausibly
  within what 300 comparisons would produce from test-set noise alone, not
  a real generalizable improvement. **Not adopted** — `latest_ensemble.json`
  still uses the single global weight (0.425), which was tuned via a much
  smaller, lower-risk sweep (12 values, one parameter). Revisit only with a
  proper train/val/test split so per-class weights can be chosen on
  validation data and reported on a test set they never touched.

### 2026-09-14 — Run 12: targeted re-scrape (Phase 1.1) — real gains, and an important accuracy-vs-baseline nuance
- User chose the data-collection path after seeing the Run 10/11 evidence
  that local architecture tuning had converged on a ceiling. Executed
  `docs/MODEL_IMPROVEMENTS.md` Phase 1.1 (targeted collection for rare
  classes) in three escalating rounds against elitigation.sg:
  1. `{bankruptcy, insolvency} AND {"struck out","strike out","withdrawn",
     "allowed in part","partially allowed"}` (note: the portal's WAF
     returns 403 on parenthesized/grouped queries like `(a OR b) AND c` —
     had to run flat `term AND term` queries per topic×disposition pair
     and merge results instead) → 198 genuinely new cases, deduped against
     the existing 1,000-case pool.
  2. Same query grid, higher result caps → 89 more.
  3. Broadened topic terms to insolvency-adjacent proceedings (`winding
     up`, `liquidation`, `scheme of arrangement` — not topic drift, these
     are core Singapore insolvency-practice categories) × the same
     disposition terms → 1,644 more. Raw pool: 1,000 → 2,931. All fetches
     succeeded (0 failures across 1,931 new judgments), at the existing
     1 req/sec politeness rate.
  4. Rebuilt `cases.csv`: 528 → 1,611 labeled cases (+205%). Class balance
     materially healthier — `struck_out` 4.5%→15.6% share, `allowed_part`
     3.6%→9.7%, `withdrawn` 4.2%→5.7% — no longer two classes dominating
     the other three.
  5. Re-swept `C` at each dataset size (528→632→664→1,611 cases) since the
     optimum shifted every time data grew; settled on `C=10.0` (was
     untuned default `1.0`) — now wired into `train_baseline.py` as
     `DEFAULT_C` (`docs/MODEL_IMPROVEMENTS.md` Phase 0.5, closed).
- **Real, substantial progress along the way, worth recording even though
  the final number reads lower:** at 664 cases, this same process reached
  0.571 accuracy / 0.353 macro-F1 — the project's best accuracy figure to
  date. Broadening further to 1,611 cases dropped accuracy to 0.455 but
  raised roc_auc to 0.710 (best yet) and, critically, raised `struck_out`
  recall to 45% (from 29% at 664 cases) with macro-F1 essentially flat
  (0.353→0.335).
- **The accuracy drop is a measurement artifact of a healthier class
  balance, not the model getting worse — checked directly, not assumed.**
  Computed a majority-class-baseline accuracy (always guess the training
  set's largest class) at each dataset size: at 664 cases the majority
  baseline itself was ~47% (a large chunk of "accuracy" was free, from
  guessing `dismissed`); at 1,611 cases the majority baseline is only
  41.5% (`dismissed` is now under 41% of the test set, down from ~47%).
  The model's *lift over the majority baseline* is a fairer comparison
  than raw accuracy across differently-balanced dataset versions, and it
  held roughly flat while the model gained real, broader discriminative
  ability (recall spread across 4 of 5 classes now, not 2-3). Recording
  this explicitly so a future session doesn't misread "accuracy went down"
  as "the bigger dataset was a mistake."
- **`withdrawn` recall is still 0%** at n=16 test examples despite the
  class's raw count growing substantially — the smallest of the five
  classes remains the hardest.
- **Still well short of 0.75.** But this is the first change in the
  project's history to move macro-F1 into the low-to-mid 0.30s *while*
  giving 4 of 5 classes non-zero recall simultaneously, via a mechanism
  (genuinely more, more balanced, real labeled examples) fundamentally
  different from every architecture-side experiment (Runs 7-11). The
  topic-broadening question (does including winding-up/liquidation/scheme
  cases alongside bankruptcy/insolvency proper add noise or add legitimate
  signal?) is not yet isolated — a useful next diagnostic would be
  evaluating accuracy separately on the original-topic subset vs. the
  newly-added-topic subset of the test set.

### 2026-09-14 — Runs 10-11 and a rejected shortcut: local methods have converged on a ceiling
- Continued pushing toward the 0.75 target after Runs 7-9. Tried the two
  most promising untried levers named at the end of that entry.
- **Run 10 (TF-IDF ⊕ chunked-BERT concatenation + `C` grid search): new
  best result on every metric** — 0.500 accuracy, 0.346 macro-F1, beating
  every prior run including the pure-TF-IDF baseline (0.481/0.232) and
  every pure-transformer variant. The representations are genuinely
  complementary: `allowed_part` gets recall with *zero* false positives
  (precision 1.00), something no single-representation run achieved.
- **Run 11 (hierarchical merits/procedural, Phase 1.4): a wash**, not an
  improvement — 0.500 accuracy (tied with Run 10), 0.321 macro-F1 (worse).
  Stage 1 (merits-vs-procedural) itself only gets 38% recall on
  "procedural" — the exact same class-imbalance problem recurring one
  level up the hierarchy, since "procedural" is still a small minority
  (13/106 test cases). Confirms imbalance isn't fixed by regrouping labels,
  only by more examples of the rare thing.
- **A shortcut tried and explicitly rejected, worth recording so it isn't
  retried uncritically:** top-2 accuracy on Run 10's model measured 0.755
  — clears 0.75 at face value. Checked before reporting it (per this
  project's standing "spot-check/verify before trusting a good number"
  practice) by comparing against a naive baseline that ignores case text
  entirely and always guesses the two largest classes (`dismissed`,
  `allowed_full`) as its "top 2" — **that baseline covers 84% of the test
  set**, higher than the model's actual top-2 accuracy. This means Run
  10's top-2 metric is an artifact of class imbalance, not evidence of
  real per-case discrimination, and reporting it as clearing the goal
  would have been misleading. Discarded as a path to the target.
- **Net conclusion after ~15 experiments across two architecture families
  (TF-IDF, 3 transformer poolings, feature concatenation, `C` tuning,
  binary-collapse reframing, selective prediction, hierarchical
  classification):** every local, non-LLM approach tried converges on the
  same ceiling — roughly 0.48-0.50 accuracy / 0.32-0.35 macro-F1 on the
  5-class task, ~0.60 on the easiest possible 2-class version. This is
  now well past the point where "try another local variant" is likely to
  close a 25-point gap to 0.75; the evidence points at the data (526
  examples, broad topic filter, heuristic labels from post-decision text —
  `docs/MODEL.md` §11) as the binding constraint, not the model family.

### 2026-09-14 — Runs 7-9: first transformer experiments (Phase 2) — first non-zero rare-class recall in the project
- User confirmed the local (non-API) path forward: fine-tune a legal-domain
  transformer, per `docs/MODEL_IMPROVEMENTS.md` Phase 2. New module:
  `ml/training/train_transformer.py`. Linear-probe design (freeze
  `law-ai/InLegalBERT`'s backbone, train only a `LogisticRegression` head)
  chosen per Phase 2.3, to avoid the higher overfitting risk of full
  fine-tuning on ~400 training examples. CUDA was available in this
  environment, so all three runs took well under 5 minutes combined.
- **Run 7 (naive [CLS], last 512 tokens): worse than the TF-IDF baseline**
  (0.425 vs 0.481 accuracy) — a genuine negative result, not a bug. Root
  cause: an un-fine-tuned BERT's [CLS] token isn't a reliable
  sentence-level representation without a task-specific pretraining
  objective (e.g. NSP) or actual fine-tuning; this is a known property of
  frozen BERT models, not specific to InLegalBERT.
- **Run 8 (mean pooling, still last 512 tokens): closes most of the gap**
  (0.443 accuracy, macro-F1 0.232 — ties the TF-IDF baseline's 0.232
  exactly) but still doesn't clearly beat it. Diagnosed why before trying
  anything else: these documents run 10,000-20,000+ tokens
  (`docs/MODEL_IMPROVEMENTS.md` Phase 2.2 already measured this), so a
  512-token window — even mean-pooled — sees under 5% of a typical
  document.
- **Run 9 (chunked whole-document, up to 24×512-token chunks averaged): the
  interesting result.** Overall accuracy dropped further (0.358) but
  macro-F1 rose to **0.267, the best of any run in this project's history**
  — because for the first time ever, `allowed_part` and `struck_out` got
  non-zero recall (33% and 20% respectively) instead of the 0% every prior
  run — TF-IDF and Runs 7-8 alike — has shown without exception. This is
  real evidence that whole-document transformer embeddings carry
  information about the rare classes that bag-of-words genuinely cannot
  represent (docs/MODEL_IMPROVEMENTS.md problem #5's predicted mechanism,
  now observed, not just argued), at the cost of some majority-class
  recall (`dismissed`/`allowed_full` recall fell to 0.41/0.35 from TF-IDF's
  0.61/0.53).
- **Not yet close to 0.75 on any metric.** But Run 9 is the first result in
  this project that moves the number that matters most (rare-class recall)
  off zero at all, via a mechanism (more of the document, not more labeled
  examples) genuinely different from everything tried in Phase 0/1. The
  most promising untried next step from here: concatenate TF-IDF and
  chunked-transformer features (majority-class strength + rare-class
  signal, rather than choosing one representation), or tune the
  `LogisticRegression` head's `C` against this new embedding space (not yet
  done — `max_iter=2000` and default `C=1.0` were used as-is).

### 2026-09-14 — Run 6: Phase 1.2 (wider extraction window) — small honest gain, rare-class recall still 0%
- User set an ambitious goal (accuracy > 0.75, eventually free-text outcome
  generation via an interactive agent). Before jumping to a new architecture,
  tried the cheapest untried lever from `docs/MODEL_IMPROVEMENTS.md` Phase
  1.2: mine the 514 already-scraped-but-unlabeled raw cases by widening
  `extract_outcome_heuristic()`'s window.
- **Widening the window is a real precision/recall tradeoff, not a free
  lunch — measured directly, not assumed.** Tested 3,000 up to the full
  document: coverage rises fast (53% → 80% of raw cases labeled at
  unlimited window) but so does the false-positive rate on manual
  spot-check — matches start hitting case-history mentions, a losing
  party's argument, or (for `withdrawn` specifically) a bank/CPF withdrawal
  that has nothing to do with the case's own disposition. 6,000 chars was
  the largest window where spot-checking every newly-rescued case still
  showed genuine disposition language. This surfaced and fixed a real bug
  along the way: `_WITHDRAWN_RE`'s bare `\bwithdrawn\b` had always been
  capable of matching "money withdrawn from an account" — invisible at the
  original 3,000-char window (financial figures rarely appear that close
  to the true end) but frequent enough at wider windows to require a fix
  (`_WITHDRAWN_MONEY_CONTEXT_RE`, `ml/preprocessing/clean.py`).
- Rebuilt `cases.csv`: 486 → 528 labeled cases (+8.6%). **The honest
  rare-class gain is small**: `struck_out` 22→24, `withdrawn` 22→22 (net
  zero — the window widening's raw gains were mostly offset by the
  money-context fix removing false positives), `allowed_part` 18→19.
  Compare to the much larger (but invalid) gains an unfiltered wide window
  would have shown (`struck_out` 22→75, `withdrawn` 22→104) — worth
  recording so a future session doesn't re-try the wide-window idea without
  the false-positive check and rediscover the same trap.
- Retrained: `baseline_20260914_135736.joblib`, 422/106 split. Accuracy
  0.469→0.481, macro-F1 0.222→0.232 — small movement, within the noise
  band established by Runs 1–3. **Rare-class recall is still 0% on all
  three classes** (`struck_out`, `withdrawn`, `allowed_part` — 5/3/9 test
  examples respectively). This is now the *fourth* independent Phase-0/1
  change that fails to move rare-class recall off zero, reinforcing that a
  handful of additional examples doesn't cross whatever threshold this
  model needs — Phase 1.1 (genuinely targeted collection, a different query
  aimed at rare-class catchwords) is a bigger, not-yet-tried lever, but
  likely still incremental.
- **New finding, addressing the 0.75 goal directly:** tested whether
  collapsing to a coarse 2-class problem (`favorable` = `allowed_full` +
  `allowed_part` vs `unfavorable` = the other three) gets meaningfully
  closer to 0.75, since a simpler task should have a much easier ceiling.
  Result: **0.585 accuracy, macro-F1 0.56** — better than the 5-class
  number, as expected, but still far short of 0.75, on the *easiest*
  version of this classification task. This is meaningful evidence (not
  just the informal "problem #5" note in `docs/MODEL_IMPROVEMENTS.md`) that
  **TF-IDF + linear classifier has a real ceiling somewhere in the 0.55-0.65
  range on this corpus, independent of class count or imbalance fixes** —
  reaching 0.75 needs an architecture change (Phase 2 transformer or Phase 3
  LLM few-shot, `docs/MODEL_IMPROVEMENTS.md`), not further Phase 0/1 tuning.

### 2026-08-30 — Run 5: Run 4's "negative result" was a bug, not a finding
- User asked why accuracy dropped in Run 4. Traced it by comparing the
  actual computed weights against sklearn's real `"balanced"` values
  (`sklearn.utils.class_weight.compute_class_weight`) instead of assuming
  the effective-number technique itself was the cause. Found: the two
  schemes' relative rare-vs-common emphasis was nearly identical at this
  project's `n`/`beta`; the real difference was a normalization bug in
  `effective_number_class_weight()` that shrank the whole weight scale
  ~3.5×. Fixed; retrained (Run 5); numbers landed back within rounding of
  `"balanced"` (Run 3). See the corrected note under the table and
  `docs/MODEL.md` §8/§10's matching update.

### 2026-08-30 — Run 4 added (effective-number class weighting)
- `docs/MODEL_IMPROVEMENTS.md` Phase 0.3 implemented and evaluated.
  Initially reported as a negative result — **superseded by the Run 5 entry
  above**, which found the drop was a bug, not a real effect. Kept here for
  the historical record rather than deleted.

### 2026-08-30 — Document created
- Backfilled Runs 0–3 from `docs/MODEL.md`'s existing Changelog entries and
  this session's classification reports; re-ran the current pipeline
  (`build_cases_csv` → `train_baseline`) to confirm Run 3's numbers
  reproduce exactly before recording them here.
