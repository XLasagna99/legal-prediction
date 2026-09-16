# Model improvement roadmap

This document diagnoses why the current model (`docs/MODEL.md`, version
`20260822_154053`) underperforms, and lays out a phased plan to fix it,
grounded in actual evidence from the fitted model plus current legal-NLP and
imbalanced-learning research. `docs/MODEL.md` documents the pipeline as it
exists; this document proposes changes to it. When an item below gets
implemented, update `docs/MODEL.md` (data/metrics/architecture sections +
Changelog) to match — this doc stays a plan, that one stays ground truth.
Also add a row to `docs/MODEL_RESULTS.md`, which tabulates every trained
iteration's metrics side by side so it's obvious which changes actually
moved the numbers.

**Methodology note:** the "top 5 problems" below are not guesses. Each is
backed by a specific diagnostic run against the actual fitted artifact and
`data/processed/cases.csv` — inspecting `LogisticRegression.coef_` for the
top-weighted feature per class, cross-tabulating `court` × `outcome`, and
retraining at 25/50/75/100% of the training set to plot a learning curve.
The commands are reproducible; see each finding for what was run.

---

## Part 1 — Top 5 problems, with evidence

### 1. Class imbalance is the dominant bottleneck — and more data alone won't fix it

**Evidence:** retraining at increasing fractions of the (chronologically
ordered) training set, holding the same fixed test tail:

| train size | accuracy | macro-F1 | macro-recall | roc_auc (macro-ovr) |
|---|---|---|---|---|
| 95 (25%) | 0.406 | 0.197 | 0.191 | 0.553 |
| 191 (50%) | 0.490 | 0.218 | 0.229 | 0.511 |
| 287 (75%) | 0.479 | 0.224 | 0.227 | 0.601 |
| 383 (100%) | 0.469 | 0.224 | 0.219 | 0.658 |

Macro-F1 is **flat** — 0.197 → 0.224 — across a 4x increase in training
data. That's the signature of a problem that scaling the *same
distribution* doesn't fix: `allowed_part`, `struck_out`, and `withdrawn`
stay at roughly 2–5% of the data no matter how large the pull gets, because
they're that rare in the underlying query's results (`docs/MODEL.md` §3).
The per-class breakdown (`docs/MODEL.md` §10) confirms it directly — 0%
recall on all three rare classes at every dataset size tested.

**Why this matters more than it looks:** it means "scrape more cases with
the same `"bankruptcy" OR "insolvency"` query" is not a lever that will
move the needle. The fix has to change the *composition* of new data
(targeted collection of rare classes) or the *learning approach* (methods
designed for scarcity), not just the *volume*.

### 2. Leakage: `struck_out` and `withdrawn` literally leak their own disposition word

**Evidence:** inspecting `clf.coef_` for the fitted `LogisticRegression`,
the single highest-weighted feature for the `struck_out` class is the
literal bigram **`struck out`** (weight 0.516), followed by `striking out`
(0.480), `strike out` (0.469), `struck` (0.438). For `withdrawn`, the top
non-metadata feature is the literal word **`withdrawn`** (0.428), with
`had withdrawn` (0.221) and `withdrawn by` (0.207) also in the top 20.

**Root cause:** `SG_LEAKAGE_PATTERNS` in `ml/preprocessing/clean.py` only
covers `dismiss/allow/grant/refus(e/ed)` phrasing (see `docs/MODEL.md` §4).
It was never extended to cover "struck out"/"strike out"/"withdrawn"
wording — an oversight from when the heuristic was binary and those words
didn't exist as separate output classes yet (`docs/MODEL.md` Changelog,
2026-08-22 entry).

**Important nuance:** this leakage is real and must be fixed, but it is
**not** the reason recall on those classes is 0% — the model has direct
access to the literal answer for those two classes and *still* can't
predict them reliably in the 5-class setting, because problem #1 (only
~18–22 total examples each) dominates even a leakage shortcut. Fixing #2
without also addressing #1 will not, by itself, fix recall — but leaving it
unfixed means any future improvement's reported numbers for those two
classes are not trustworthy, since the model would be graded on a question
it's allowed to see the answer to.

### 3. Overfitting to judge/party names in a 50,005-feature / 383-example regime

**Evidence:** top positive coefficients include specific people's names:
`goh` / `yihan` / `goh yihan` (a specific High Court judge, Goh Yihan) for
`allowed_full`; `mr chan`, `wang`, `sudhir`, `chadwick` scattered across
other classes. These are proper nouns tied to specific cases in the
training set, not generalizable legal concepts.

**Why it happens:** `p >> n` — 50,005 features against 383 training rows is
a classically overfitting-prone regime for a linear model, even with
`class_weight="balanced"` and sklearn's default L2 regularization
(`C=1.0`, never tuned — `docs/MODEL.md` §8). A name that happens to
correlate with an outcome in a handful of training cases gets a real
coefficient, and there's nothing in the current pipeline stopping it.

**Consequence:** the model partly predicts *whose case this is* / *which
judge heard it*, not *what the legal merits suggest* — this won't
generalize to new judges, new parties, or (worse) could encode something
close to individual-judge profiling, which `ARCHITECTURE.md`'s
responsible-use notes specifically flag as a jurisdiction-sensitive concern
("some jurisdictions restrict profiling individual judges").

### 4. The model leans on `court` (a weak structural proxy) more than almost any lexical feature

**Evidence:** one-hot `court` coefficients are among the largest in the
model: `court_SGCA` = **1.079** for `withdrawn` (the single largest
coefficient found across all 5 classes), `court_SGHCR` = 0.463 for
`struck_out`, `court_SGHC` = 0.755 for `allowed_part`. Cross-tabulating
`court` × `outcome` shows why: 7 of the 22 `withdrawn` cases (32%) are
`SGCA`, even though `SGCA` is only 26/479 (5.4%) of the whole dataset — a
real skew, but built on just 7 examples.

**Why it's a problem even though the correlation is real:** appeals
plausibly do get withdrawn at a different rate than first-instance
applications (a defensible domain intuition — settling after an appeal is
filed but before it's heard is common), but a coefficient this large,
learned from 7 data points, is fragile. It will not survive a larger or
differently-composed pull of SGCA cases, and it's doing the job that
genuine textual reasoning about the case should be doing. This is the same
underlying cause as #3 (data scarcity forcing the model onto whatever
correlate is available), applied to structural metadata instead of proper
nouns.

### 5. Bag-of-words / TF-IDF has a representational ceiling this task will hit regardless of data fixes

This one isn't from inspecting the artifact — it's a known, literature-
documented limitation of the architecture itself, and worth stating plainly
so it doesn't get missed while fixing #1–#4. TF-IDF + linear classifier has
no way to represent:
- **Negation and argument-vs-holding structure** — "the defendant argued
  the debt was time-barred, but the court rejected this" contains the
  token `argued` and `debt` and `time-barred` with no signal distinguishing
  the losing argument from the court's actual holding. A model reading
  this as a bag of words cannot tell which side's contention became the
  outcome.
- **Long-range dependency across a 50,000+ character document** — the
  actual operative reasoning may depend on facts established in paragraph
  10 and applied in paragraph 90; TF-IDF has no sequence or position
  information at all.
- **Multi-issue judgments** — `docs/MODEL.md` §3 already documents the
  `2020_SGHC_5` case (claim dismissed, counterclaim allowed) as a
  single-label limitation; a bag-of-words model has no structural way to
  even represent "this document contains two separate rulings," let alone
  predict both.

Fixing #1–#4 will raise the ceiling this architecture can reach, but won't
remove the ceiling. Genuinely resolving #5 requires representational
capacity TF-IDF doesn't have — see Phase 2 below.

---

## Part 2 — Roadmap

Ordered by dependency, not just importance: several later phases would
produce misleading results if run before earlier ones (e.g., benchmarking
a transformer against leakage-inflated struck_out/withdrawn numbers), so
**do not skip ahead** even if a later phase looks more appealing. This
follows standard data-centric-AI practice — fix data and label quality
before increasing model complexity, or added complexity just overfits the
same defects harder.

### Phase 0 — Fix what's broken now (days, no new data needed)

| Item | What | Why |
|---|---|---|
| 0.1 | ✅ **Done 2026-08-28.** Extend `SG_LEAKAGE_PATTERNS` (`ml/preprocessing/clean.py`) to strip "struck out"/"strike out"/"withdrawn"/"had withdrawn" phrasing, matching how `dismiss`/`allow` are already covered | Fixes problem #2 directly. Rebuild `cases.csv`, retrain, and diff the `struck_out`/`withdrawn` coefficients before/after to confirm the literal-answer leakage is gone |
| 0.2 | ✅ **Done 2026-08-28.** Re-ran the coefficient inspection after 0.1 — confirmed 0/479 rows still contain the literal words, and they're gone from the top coefficients. **Correction to the prediction made when this row was written:** it predicted recall would "drop further, possibly to a still-0% floor" — in fact `struck_out`/`withdrawn` recall was *already* 0% before the fix and stayed 0% after (see `docs/MODEL.md` §10, 2026-08-28 Changelog entry). The leaked words weren't, on their own, enough signal to move the argmax given how few training examples exist — problem #1 (imbalance) is the binding constraint, not problem #2 (leakage) | Establishes a clean baseline before any other change, so later improvements are measured against a trustworthy number — this is now true; problems #3/#4 (name overfitting, court over-reliance) are visible in the current top coefficients and are what Phase 0.3-0.5 target next |
| 0.3 | ✅ **Done 2026-08-30 — result: null (once a self-inflicted bug was fixed).** Swapped `class_weight="balanced"` for effective-number-of-samples re-weighting ([Cui et al., CVPR 2019](https://arxiv.org/abs/1901.05555): weight ∝ `(1-β)/(1-β^n)` per class). **First measurement was wrong, and the correction is itself the interesting part:** initially every aggregate metric dropped (accuracy 0.469→0.429), which looked like evidence the technique hurt. Investigating *why* (comparing this project's computed weights against sklearn's actual `"balanced"` output) found a normalization bug — this project's weights were scaled to a plain per-class average of 1.0 instead of sklearn's sample-weighted average of 1.0, silently shrinking the whole weight scale ~3.5× and acting like extra regularization. Fixed; retrained; numbers landed back within rounding of `"balanced"`. Rare-class recall: 0% throughout, both before and after the bug fix. Full trace: `docs/MODEL_RESULTS.md` Runs 4→5, `docs/MODEL.md` §8/§10 | `balanced` uses naive inverse-frequency, which over-weights a 10-example class to the point of instability; effective-number weighting is specifically designed for the "very few examples, near-duplicate content" regime `allowed_part` is in — **turns out, at this project's `beta`/class-size range, the two schemes' relative treatment of rare vs. common classes is nearly identical anyway (ratio 18.8× vs 20.6×), so a real, bug-free comparison was always going to be a near-null result.** This is now a third independent data point (after 0.1/0.2) that the binding constraint is problem #1 (raw data volume for 3 classes), not the loss function or the leakage — reinforces going straight to Phase 1 next rather than trying 0.4/0.5 first |
| 0.4 | Named-entity stripping/masking pass before TF-IDF: replace judge names, party names, and law firm names with placeholder tokens (`[JUDGE]`, `[PARTY]`, `[FIRM]`) using a simple NER pass (spaCy's `en_core_web_sm` or similar) | Directly targets problem #3. Cheap to add, doesn't require new data, and is good practice regardless — a model that generalizes shouldn't need to know it's "Goh Yihan J" to predict correctly |
| 0.5 | ✅ **Done 2026-09-14** (as part of Phase 1.1's re-scrape work, not standalone). Grid-searched `C` at each dataset size as data grew; settled on `C=10.0` (weaker regularization than the default, not stronger) — wired into `train_baseline.py` as `DEFAULT_C`. Re-sweep if the dataset changes materially | `ROADMAP.md` Phase 2 already flags this as unstarted. **Correction to this row's original hypothesis:** it predicted *lower* `C` (stronger regularization) would help by shrinking spurious coefficients (#3/#4); the actual sweep favored progressively *higher* `C` as the dataset grew — more data needs less shrinkage to generalize, the opposite direction from what #3/#4's overfitting-at-small-n framing implied |

### Phase 1 — Fix the data, not just the model (1–3 weeks)

| Item | What | Why |
|---|---|---|
| 1.1 | ✅ **Done 2026-09-14 — this is the change that finally worked.** Re-queried `elitigation.sg` with flat `{topic} AND {disposition term}` queries (grouped/parenthesized queries 403 from the portal's WAF) targeting `struck_out`/`withdrawn`/`allowed_part`, in 3 escalating rounds — 1,931 new judgments fetched, `cases.csv` 528 → 1,611. `struck_out` recall 0%→45%, `allowed_part` 0%→22%, macro-F1 0.232→0.335. Full trace: `docs/MODEL_RESULTS.md` Run 12 | Active-learning research on rare-class collection reports needing up to 8x fewer newly-labeled examples than random/undifferentiated collection to move rare-class performance ([TALISMAN, arXiv:2112.00166](https://arxiv.org/pdf/2112.00166)) — confirmed directly: this is the first change of any kind (Phase 0, 1, or 2) to move rare-class recall substantially, after ~15 architecture-side experiments failed to |
| 1.2 | ✅ **Done 2026-09-14 — small honest gain, does not change the Phase 1 diagnosis.** Widened `extract_outcome_heuristic()`'s `window` 3,000 → 6,000 chars (chosen by testing up to the full document and spot-checking every newly-rescued case at each width — wider windows raise coverage fast but also a real false-positive rate). `cases.csv` grew 486 → 528; rare classes grew only marginally (`struck_out` 22→24, `withdrawn` 22→22 net after a real precision bug fix, `allowed_part` 18→19) since most rescued cases were `dismissed`/`allowed_full`. Retrained: accuracy 0.469→0.481, **rare-class recall still 0% on all three** — full trace in `docs/MODEL_RESULTS.md` Run 6 | Free — this data is already scraped, just unlabeled. Confirms (a fourth time) that the binding constraint is genuinely rare-class example count, not extraction coverage — a *targeted* re-query (1.1) or a different architecture (Phase 2/3) is what's actually needed next |
| 1.3 | **Weak supervision (Snorkel-style)** ([Ratner et al., "Data Programming"](https://snorkel.ai/data-centric-ai/programmatic-labeling/)): write several independent labeling heuristics (the existing regex heuristic, a court-level prior, a proximity-to-"Conclusion"-heading rule, a costs-language rule) and combine them via a generative label model that learns each heuristic's reliability, rather than trusting one regex as ground truth | Both expands usable labeled data (cases where heuristics agree even if the primary regex misses) and gives an explicit, inspectable reliability estimate for the current heuristic — directly useful for auditing problem #2-style bugs before they happen again |
| 1.4 | **Hierarchical (coarse-to-fine) classification**: stage 1 predicts a 2-way split — resolved-on-merits (`dismissed`/`allowed_full`/`allowed_part`) vs. resolved-procedurally (`struck_out`/`withdrawn`) — stage 2 classifies within whichever branch | Reduces the sparse 5-way problem to two easier ones; `struck_out` and `withdrawn` are procedurally similar to each other and dissimilar to the merits outcomes, so this grouping should be easier to learn than 5 flat, imbalanced classes ([hierarchical LJP framing, ACL 2021](https://aclanthology.org/2021.emnlp-main.46.pdf)) |
| 1.5 | Downsampling experiment: retrain with `dismissed`/`allowed_full` capped to match a smaller multiple of the rare classes, to see whether reducing majority-class dominance (at the cost of discarding majority data) helps macro metrics — the original ECHR outcome-prediction study ([Aletras et al. 2016](https://arxiv.org/pdf/2110.00976)) used a deliberately balanced 584-case dataset rather than an imbalance-correction algorithm | Cheap experiment (no new code beyond a sampling script) that tells you whether the imbalance techniques in 0.3 are enough, or whether the model needs less majority-class noise, before investing in heavier machinery |

### Phase 2 — Move past bag-of-words (weeks)

| Item | What | Why |
|---|---|---|
| 2.1 | ✅ **Done 2026-09-14.** Linear-probe over **InLegalBERT** ([law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)) implemented in `ml/training/train_transformer.py`. Naive [CLS]-token version underperformed TF-IDF (0.425 vs 0.481 accuracy); mean-pooling closed most of the gap (0.443, tied on macro-F1) — see `docs/MODEL_RESULTS.md` Runs 7-8 | Directly addresses problem #5 — transformer attention can represent negation and long-range dependency that TF-IDF cannot |
| 2.2 | ✅ **Done (approximated) 2026-09-14 — real signal found.** Not a true Longformer (no cross-chunk attention), but a cheap approximation: split each document into up to 24×512-token chunks and average the chunk embeddings, so the model sees up to ~12,000 tokens instead of 512. Result: macro-F1 rose to 0.267 (best in the project) because `allowed_part`/`struck_out` got their first-ever non-zero recall (33%/20%) — real evidence for this row's hypothesis, traded off against majority-class recall. Full Longformer (true long-range attention, not just chunk-averaging) remains untried | Truncating to 512 tokens would cut off most of a Singapore judgment (~10,000-20,000+ tokens per case) — confirmed directly: chunk-averaging recovered rare-class signal that both TF-IDF and the 512-token-only transformer runs missed entirely |
| 2.3 | ✅ **Done 2026-09-14.** Used linear-probe fine-tuning (frozen backbone + `LogisticRegression` head) for all three Phase 2.1/2.2 runs, not full fine-tuning, exactly per this row's recommendation | Directly avoids repeating problem #3/#4 (overfitting to spurious correlates) at a larger parameter count, which would make things worse, not better |
| 2.4 | Only after 2.1–2.3 show a real gain over the Phase-0/1 baseline, consider full fine-tuning or a bigger model | Sequencing discipline — a bigger model on the same broken-then-fixed data should be judged against the honest Phase-0/1 numbers, not against the original leakage-inflated ones |

### Phase 3 — Benchmark against LLM few-shot classification (parallel to Phase 2)

| Item | What | Why |
|---|---|---|
| 3.1 | Prompt a large model (Claude/GPT-4-class) with a handful of labeled examples per class (weighted toward the rare ones) and the 5-class task definition, evaluate on the same held-out test set | Reported results show few-shot in-context learning substantially beating zero-shot (57.3% → 73.2% accuracy in one LJP study) and in some settings comparable to fine-tuned transformers — cheap to test as a baseline before investing in 2.1–2.4's fine-tuning effort |
| 3.2 | **Do not trust the raw probability/confidence an LLM reports.** Research on legal-domain LLM calibration finds "inertia of confidence" — near-maximal certainty regardless of actual correctness, worse on harder/lower-signal cases (directly analogous to the rare classes here) | Any LLM-based classifier's output needs the same post-hoc calibration treatment as Phase 4, not a bypass of it |
| 3.3 | If an LLM approach is pursued for production (not just benchmarking), treat it as a **feature extractor** (e.g., extract structured facts from the judgment, then classify) rather than an end-to-end verdict generator | Legal-domain hallucination base rates are reported as high (58%+ in some studies); constraining the LLM to extraction rather than open verdict generation reduces surface area for fabrication |

### Phase 4 — Calibration and rigorous evaluation (after Phases 0–1 land)

| Item | What | Why |
|---|---|---|
| 4.1 | Wrap the final classifier in `sklearn.calibration.CalibratedClassifierCV` (Platt or isotonic) | `ROADMAP.md` Phase 3 already names this as planned. Given how few test examples exist per rare class, this is the realistic choice — Dirichlet calibration ([Kull et al., NeurIPS 2019](https://papers.neurips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf)) is more accurate in principle but needs more held-out data per class than this dataset has |
| 4.2 | Report **bootstrap confidence intervals**, not point estimates, for macro-F1/recall/precision, given a 96-row test set with as few as 2 examples of a class | A single-number macro-F1 of 0.224 implies more precision than a 96-row test set can actually support; a confidence interval makes that honest |
| 4.3 | ✅ **Done 2026-09-17.** Added `ml/evaluation/bias_audit.py`, which loads the latest registry model, reproduces its exact temporal test split, and runs `group_metrics()` by `court`. Result on `baseline_20260914_154554` (n=323 test rows): accuracy ranges from 0.30 (`SGCA`, n=20) to 0.49 (`SGHC`, n=223); `SGHCR` (n=44) and `unknown` (n=36) sit in between at 0.39/0.42. Full table below | Given #4's finding that `court` coefficients are doing outsized work, checking whether error rates differ meaningfully by court is now a bias-audit question, not just a data-quality one |
| 4.4 | Explicitly log, per `docs/MODEL.md`'s Changelog convention, whether each phase's change was measured on the Phase-0-corrected baseline or an earlier one | Prevents accidentally comparing a Phase-2 transformer's numbers against the leakage-inflated Phase "-1" numbers from before this document existed |

**4.3 full results** (`python -m evaluation.bias_audit` from `ml/`, `baseline_20260914_154554`):

| court | accuracy | precision | recall | f1 | n |
|---|---|---|---|---|---|
| SGHC | 0.489 | 0.367 | 0.284 | 0.286 | 223 |
| SGHCR | 0.386 | 0.297 | 0.350 | 0.259 | 44 |
| unknown | 0.417 | 0.247 | 0.298 | 0.270 | 36 |
| SGCA | 0.300 | 0.255 | 0.240 | 0.217 | 20 |

`roc_auc` is `None` for every group here (not a bug) — `evaluate()` is called
without `y_score` in `bias_audit.py`'s current form, so it's skipped
entirely rather than computed and discarded.

`SGCA` (appeals) is both the smallest group (n=20) and the worst-performing
(accuracy 0.30, roughly 19 points below `SGHC`). This is consistent with,
not independent of, problem #4 above: `SGCA` is also where the model
learned its single largest coefficient (`court_SGCA` = 1.079 for
`withdrawn`) from just 7 training examples. A gap this size on 20 test rows
is not yet strong evidence of systematic bias by itself (the confidence
interval at this n is wide — see 4.2, still unimplemented), but it's the
same court the coefficient inspection already flagged, which raises the
prior that it's a real, not spurious, weak spot. Re-run this audit once
Phase 1 (more `SGCA`-specific data) or Phase 4.2 (bootstrap CIs) lands,
rather than treating this one snapshot as conclusive.

---

## Part 3 — What this roadmap deliberately does *not* recommend yet

- **Don't jump straight to Phase 2 (transformers) or Phase 3 (LLMs) before Phase 0.** The learning curve (Part 1, #1) already shows that adding capacity without fixing labels/leakage doesn't help — a bigger model trained on data where two classes leak their own answer, and where proper nouns are uncontrolled, will learn the same shortcuts faster and more confidently, not better legal reasoning.
- **Don't treat ROC-AUC improvements in isolation as evidence of progress.** `docs/MODEL.md` §10 already flags that macro one-vs-rest AUC rose (0.587 → 0.658) between the binary and multi-class runs while macro-F1/recall stayed flat — this metric can look better even when the model's actual predictions (the argmax) haven't improved. Track the full metric set, and especially per-class recall, at every phase.
- **Don't scale up the raw case pull under the current query before Phase 1.1–1.2.** Per #1's evidence, this specific lever is empirically shown not to work here.

---

## Sources

- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019 — [arxiv.org/abs/1901.05555](https://arxiv.org/abs/1901.05555)
- Lin et al. (Focal Loss), summarized — [towardsdatascience.com](https://towardsdatascience.com/a-loss-function-suitable-for-class-imbalanced-data-focal-loss-af1702d75d75/)
- Two-Stage Fine-Tuning for imbalanced classification — [arxiv.org/abs/2207.10858](https://arxiv.org/abs/2207.10858)
- Hierarchical legal judgment prediction — [aclanthology.org/2021.emnlp-main.46](https://aclanthology.org/2021.emnlp-main.46.pdf)
- Active learning for rare classes (TALISMAN) — [arxiv.org/pdf/2112.00166](https://arxiv.org/pdf/2112.00166); overview — [statisticalhorizons.com](https://statisticalhorizons.com/supercharge-your-classifier-development-with-active-learning/)
- Legal-BERT — [huggingface.co/nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
- InLegalBERT — [huggingface.co/law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
- Longformer for legal documents — [arxiv.org/abs/2211.00974](https://arxiv.org/abs/2211.00974)
- LexGLUE benchmark — [arxiv.org/pdf/2110.00976](https://arxiv.org/pdf/2110.00976)
- LLM few-shot legal judgment prediction (Athena RAG-LJP) — [arxiv.org/html/2410.11195](https://arxiv.org/html/2410.11195)
- LLM legal-domain calibration / overconfidence — [academic.oup.com/jla](https://academic.oup.com/jla/article/16/1/64/7699227), [dho.stanford.edu](https://dho.stanford.edu/wp-content/uploads/Hallucinations_JLA.pdf)
- ILDC / CJPE (Indian LJP benchmark) — [aclanthology.org/2021.acl-long.313](https://aclanthology.org/2021.acl-long.313/)
- Medvedeva, Wieling & Vols, *Rethinking the field of automatic prediction of court decisions* — [link.springer.com/10.1007/s10506-021-09306-3](https://link.springer.com/article/10.1007/s10506-021-09306-3)
- Medvedeva & McBride, *Legal Judgment Prediction: If You Are Going to Do It, Do It Right*, NLLP 2023 — [aclanthology.org/2023.nllp-1.9](https://aclanthology.org/2023.nllp-1.9/)
- Fine-tuning smaller transformers on limited data — [towardsdatascience.com](https://towardsdatascience.com/fine-tune-smaller-transformer-models-text-classification-77cbbd3bf02b/)
- Snorkel / data programming — [snorkel.ai/data-centric-ai/programmatic-labeling](https://snorkel.ai/data-centric-ai/programmatic-labeling/)
- `CalibratedClassifierCV` — [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- Dirichlet calibration — [NeurIPS 2019](https://papers.neurips.cc/paper/9397-beyond-temperature-scaling-obtaining-well-calibrated-multi-class-probabilities-with-dirichlet-calibration.pdf)

---

## Changelog

### 2026-09-17 — Phase 4.3 implemented — bias audit by court
- Added `ml/evaluation/bias_audit.py` (+ `ml/tests/test_bias_audit.py`),
  wiring the existing `group_metrics()` up to the latest registry model and
  its exact temporal test split, grouped by `court`. `ROADMAP.md` Phase 2
  and this doc's Phase 4.3 both flagged this as the last open item in that
  section.
- Result: `SGCA` (appeals) is both the smallest test group (n=20) and the
  worst-performing (0.30 accuracy vs. 0.49 for `SGHC`, n=223) — see the
  full table above. Consistent with problem #4's finding that `court_SGCA`
  carries the single largest coefficient in the model, learned from only 7
  training examples; not yet strong evidence on its own given the small n,
  but corroborating, not new/independent, evidence.
- Does not change the Phase 1 diagnosis — this is a symptom of the same
  data-scarcity root cause as problems #1/#4, not a new independent issue
  requiring its own fix track.

### 2026-09-14 — Phase 2 implemented (3 variants) — first non-zero rare-class recall in the project
- User confirmed direction: stay local (no LLM API for now), and pursue the
  transformer architecture change (Phase 2) next, having just measured that
  Phase 0/1 tuning of TF-IDF+LogReg is exhausted (previous changelog entry).
- Implemented `ml/training/train_transformer.py`: linear-probe fine-tuning
  (frozen `law-ai/InLegalBERT` backbone + `LogisticRegression` head, per
  2.3) in three variants, each a direct response to the previous one's
  diagnosed weakness:
  1. [CLS] token, last 512 tokens → worse than TF-IDF (0.425 vs 0.481 acc)
  2. Mean pooling, last 512 tokens → ties TF-IDF on macro-F1 (0.232) but
     not accuracy (0.443) — diagnosed cause: 512 tokens covers <5% of a
     typical 10,000-20,000+ token judgment (2.2's exact prediction)
  3. Mean pooling, chunked whole-document (up to 24×512-token chunks
     averaged) → macro-F1 0.267 (best ever), and **allowed_part/struck_out
     get non-zero recall for the first time in this project's history**
     (33%/20%), at the cost of majority-class recall
- Full trace, numbers, and per-class breakdowns: `docs/MODEL_RESULTS.md`
  Runs 7-9.
- **Still far from 0.75**, but this is the first change of any kind (Phase
  0, 1, or 2) to move rare-class recall off zero — a qualitatively
  different result from every TF-IDF-based run. Most promising untried
  next step: concatenate TF-IDF and chunked-transformer features rather
  than choosing one representation, since they show complementary
  strengths (majority-class precision vs. rare-class recall); `C` tuning
  for the new embedding space also untried.

### 2026-09-14 — Phase 1.2 implemented — small honest gain, rare-class recall unmoved; new evidence on the 0.75 target
- User set a new goal: accuracy > 0.75, plus eventual free-text outcome
  generation via an interactive agent. Ran Phase 1.2 (the cheapest untried
  Phase 0/1 item) before considering an architecture change.
- Full detail in `docs/MODEL.md`'s matching Changelog entry and
  `docs/MODEL_RESULTS.md` Run 6. Net finding: `cases.csv` grew 486 → 528,
  but rare-class recall is still 0% on `struck_out`/`withdrawn`/
  `allowed_part` — the fourth Phase 0/1 change in a row not to move it.
- **New, directly relevant to the 0.75 goal:** tested the easiest possible
  collapse of this task (2-class favorable/unfavorable) and got only 0.585
  accuracy. This is now measured evidence, not just the informal problem #5
  argument, that TF-IDF+LogReg's ceiling on this corpus is well below 0.75
  regardless of further Phase 0/1 tuning — Phase 2 (transformer) or Phase 3
  (LLM few-shot) is the next lever actually capable of reaching the new
  target, not more imbalance/leakage/coverage work on the current model.

### 2026-08-30 — Phase 0.3's "negative result" was our own bug
- User asked why accuracy dropped after 0.3. Root-caused it by diffing this
  project's computed class weights against sklearn's real `"balanced"`
  output rather than trusting the plausible-sounding narrative — found a
  normalization bug in `effective_number_class_weight()` (see `docs/
  MODEL.md`'s matching Changelog entry for the full mechanism). Fixed;
  retrained; numbers landed back at the `"balanced"` baseline.
- Updated conclusion: it's still a null result on rare-class recall (0%
  before and after, bug or no bug), but the *reason* is now correctly
  understood as "the two weighting schemes barely differ at this project's
  scale," not "effective-number weighting actively hurts." Same practical
  recommendation as before this correction — Phase 1 (targeted data
  collection) next — but for the right reason now.

### 2026-08-30 — Phase 0.3 implemented — see correction above

### 2026-08-28 — Test suite added; caught 2 more bugs
- Added `ml/tests/` (pytest) to guard the code touched by this roadmap's
  fixes against regressions — see `docs/MODEL.md` §12. Run it before every
  rebuild/retrain from here on.
- Writing tests for `extract_outcome_heuristic` and `evaluate()` found two
  real bugs (an `allowed_part` regex phrasing gap, and a `roc_auc` `nan`-vs-
  `None` leak) independent of this roadmap's planned Phase 0 items — both
  fixed. Full detail in `docs/MODEL.md`'s matching Changelog entry. Neither
  changes the Phase 1 diagnosis: rare-class recall is still 0%, still a
  data-scarcity problem, not a code-correctness one.

### 2026-08-28 — Phase 0.1/0.2 implemented
- Extended `SG_LEAKAGE_PATTERNS` to fix the struck_out/withdrawn leakage
  from problem #2. See `docs/MODEL.md` §4 and its 2026-08-28 Changelog
  entry for full detail and numbers.
- Prediction in 0.2's original text didn't hold: expected recall to *drop*
  after the fix; it was already at 0% and stayed there. Row updated above
  with the correction. Net effect: confirms problem #1 (imbalance), not
  problem #2 (leakage), is the binding constraint on the rare classes.

### 2026-08-22 — Document created
- Diagnosed top 5 problems via direct inspection of `baseline_20260822_154053.joblib` coefficients, a `court`×`outcome` cross-tab, and a learning-curve retrain at 25/50/75/100% of training data.
- Roadmap phases written against current research (see Sources). No pipeline code changed by this document — see `docs/MODEL.md` for what's actually implemented, and update both together as phases land.
