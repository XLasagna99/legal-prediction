# From a disposition label to "the full outcome and proceedings"

This document lays out what it would actually take to abstract the model
from predicting a single disposition category (`docs/MODEL.md`'s current
5-class `dismissed`/`allowed_full`/`allowed_part`/`struck_out`/`withdrawn`)
toward something closer to "the exact predicted outcome and proceedings of
the case" — relief granted, monetary amounts, costs orders, and possibly a
narrative description of what will happen. Read this alongside
`docs/MODEL_IMPROVEMENTS.md`, which fixes the *existing* classifier;
this document is about *changing what the model is asked to output at all*.

**The central intricacy, stated up front:** "exact predicted outcome and
proceedings" is not one target, it's several, of sharply increasing
difficulty, data requirement, and risk. This document breaks it into four
levels and is explicit about which ones are realistic now, which need
substantially more data, and which carry regulatory/ethical exposure that
should be resolved deliberately, not backed into.

**The interactive agent (2026-09-14): built at `ml/agent/case_agent.py`.**
Rather than free-form generation, it turns the model's actual
`predict_proba` output into an English summary via a fixed template, and
answers a defined set of question intents (outcome, confidence,
alternatives, costs, model accuracy, disposition meaning) by reading only
from that same structured output — nothing it says can go beyond what the
model actually predicted. This is the practical implementation of "Level 1
output → English sentences → an agent that answers questions about it"
without taking on Level 3/4's generation risk. See the module docstring for
the two design choices and their rationale, and `docs/MODEL.md`'s matching
Changelog entry for how it's tested.

---

## The four levels

| Level | Target | Example output | Feasible now? |
|---|---|---|---|
| 0 (current) | Single disposition category | `dismissed` | Yes — built, see `docs/MODEL.md` |
| 1 | Structured multi-field outcome | `{disposition: allowed_part, costs_to: respondent, costs_amount_bucket: "$10k-$50k", relief_type: monetary}` | Partially — needs new labeled fields, but no new architecture paradigm |
| 2 | Quantum/damages regression | `costs_awarded: $34,500` (a number, not a bucket) | Hard — noisier problem, more data-hungry, may not be well-suited to this case type at all (see below) |
| 3 | Rationale/explanation generation | Label + a paragraph explaining *why*, grounded in retrieved similar cases | Research-stage even in well-funded academic settings; not recommended as a near-term production target |
| 4 | Full generative "predicted proceedings" narrative | A free-text paragraph describing what will happen in the case, unconstrained | **Not recommended.** See [Responsible-use intricacies](#responsible-use-and-regulatory-intricacies) below — this is where real regulatory precedent exists against exactly this framing |

Each level is described below with what it needs and what it costs.

---

## Level 1 — Structured multi-field outcome

**Status (2026-09-14): in progress, chosen as the project's target output
level.** `costs_amount_bucket` is implemented and in `cases.csv` (528
rows, `ml/preprocessing/clean.py:extract_costs_amount_bucket()` —
docs/MODEL.md's matching Changelog entry has the spot-check results).
`costs_to` and `relief_type` are not yet implemented. None of the three
fields are wired into `train_baseline.py` as a prediction target yet —
that's deliberately deferred past a Phase 2/3 architecture change (see
`docs/MODEL_IMPROVEMENTS.md`), since TF-IDF+LogisticRegression's measured
ceiling on the existing disposition task (`docs/MODEL_RESULTS.md` Run 6:
0.585 accuracy even on the easiest 2-class collapse) means predicting more
fields with the same architecture would just be more outputs stuck at the
same low ceiling, not genuinely richer prediction.

**What changes:** instead of one `outcome` column, `cases.csv` gains
several target columns predicted jointly — e.g. `disposition` (the
existing 5-class label), `costs_to` (which party costs were awarded
against), `costs_amount_bucket` (a binned range, not a raw number — see
Level 2 for why binning matters), `relief_type` (monetary / injunctive /
declaratory / none).

**Data requirements:**
- Each new field needs its own extraction heuristic, similar in spirit to
  `extract_outcome_heuristic()` but pattern-matching different language —
  e.g. costs orders in Singapore judgments follow fairly formulaic
  phrasing (`"...costs fixed at $X to be paid by [party] to [party]"`,
  visible in the raw judgment text already scraped — see the
  `2026_SGHC_167` example in `docs/MODEL.md`'s conversation history:
  *"That sum shall be paid by the claimants to the defendants"*), so
  `costs_to`/`costs_amount_bucket` are plausibly extractable from the
  **already-scraped** `data/raw/singapore_bankruptcy_cases.jsonl` without a
  new scrape.
- `relief_type` is harder — it requires distinguishing what kind of order
  was actually made (an injunction vs. a monetary judgment vs. a
  declaration), which isn't announced in one consistent phrase the way
  disposition is. This likely needs either a larger heuristic pattern set
  per relief type, or a small hand-labeled validation set to check
  heuristic accuracy before trusting it at scale (same caution as
  `docs/MODEL.md` §3's "spot-check before trusting" principle, applied to
  a new field).

**Architecture:** multi-task learning, not four separate models. The
established precedent is **TOPJUDGE** ([Zhong et al., 2018](https://aclanthology.org/D18-1390.pdf)),
which models law-article, charge, and prison-term prediction as a DAG of
dependent subtasks rather than independent labels, and **CAIL2018**
([arxiv.org/pdf/1807.02478](https://arxiv.org/pdf/1807.02478)), the
reference large-scale dataset that formalized this as jointly predicting
multiple structured fields from one case. Concretely: one shared text
encoder (whatever survives `docs/MODEL_IMPROVEMENTS.md` Phase 0–2 — TF-IDF
or a fine-tuned transformer), with a separate classification head per
target field, trained jointly so the shared representation benefits from
all the signal, not just one field's.

**Important caveat found in research:** the multi-task literature is
concentrated on Chinese **criminal** LJP (charge + law article + sentence
term). No strong civil/commercial analogue was found for jointly predicting
liability + relief-type + damages + costs — this is genuinely less
developed territory, so treat Level 1 as adapting a technique from a
different case-type domain, not applying an off-the-shelf civil-case
system.

**Verdict:** the most realistic next step if richer output is wanted. No
new fundamental architecture paradigm, extraction work is tractable for
at least the `costs_*` fields from data already on hand, and it produces
genuinely more useful output (a cost-outcome prediction is actionable in a
way a bare disposition label isn't) without the generation risks of
Levels 3–4.

---

## Level 2 — Quantum/damages regression

**What it would mean:** predicting an actual dollar figure — "costs of
$34,500" — rather than a bucket.

**What the research says about this specific sub-problem:**
- It's treated as a recognized but **harder and noisier** problem than
  categorical disposition prediction. A PeerJ study on predicting
  compensation for immaterial damage ([peerj.com/articles/cs-1225](https://peerj.com/articles/cs-1225/))
  and a comparison of LASSO/OLS/random forest/XGBoost/BERT for
  damages-amount prediction ([ResearchGate](https://www.researchgate.net/publication/380885719_Predicting_the_Amount_of_Compensation_for_Harm_Awarded_by_Courts_Using_Machine-Learning_Algorithms))
  both found this is driven by many continuous, case-specific factors
  (injury type, treatment specifics, claimant relationship in those
  studies' domains) that don't reduce cleanly to text features the way a
  disposition category does.
- **Binned/discretized classification is the common fallback to raw
  regression when data is small** — exactly the situation here. Given
  ~479 examples total and costs/damages figures that would be sparser
  still (not every case has an extractable dollar figure), Level 1's
  `costs_amount_bucket` (a handful of ranges: none / <$10k / $10k-$50k /
  $50k+) is the realistic version of this; raw-number regression should
  wait until there's meaningfully more data specifically with clean
  extracted dollar figures.
- **Bankruptcy/insolvency specifically may not be the best case type for
  this.** The current corpus is about applications, bankruptcy orders, and
  statutory demands (`docs/MODEL.md` §7's top TF-IDF features:
  `"bankruptcy order"`, `"statutory demand"`) rather than damages-heavy
  litigation — costs figures exist, but substantive damages awards are
  comparatively rare in this case type. A different case-type pull
  (contract/tort claims) might be a better source of quantum-prediction
  training data than the current bankruptcy-focused corpus.

**Verdict:** worth doing only for `costs` (extractable, present in most
judgments) and only as a bucket. Full damages-amount regression should be
deferred — it's a different, harder problem best tackled with a
purpose-collected dataset, not bolted onto the existing bankruptcy corpus.

---

## Level 3 — Rationale/explanation generation

**What it would mean:** alongside the predicted label, generate a
paragraph explaining *why* — e.g. "the application is likely to be
dismissed because the statutory demand was not set aside within the
required period, similar to [cited precedent]."

**What exists in research:** this is an active academic area, concentrated
in Indian legal NLP — **NyayaRAG** ([arxiv.org/html/2508.00709v2](https://arxiv.org/html/2508.00709v2))
generates both a decision and explanation using retrieval-augmented
generation; **PredEx** ([arxiv.org/pdf/2406.04136](https://arxiv.org/pdf/2406.04136))
is a judgment-prediction-plus-explanation dataset/model; **NyayaAnumana &
INLegalLlama** ([arxiv.org/pdf/2412.08385](https://arxiv.org/pdf/2412.08385))
pairs the largest current Indian LJP dataset with a specialized
explanation-generating LLM.

**The hallucination problem, specifically for this level:** these systems
consistently report that generation is "especially prominent when input
facts are sparse/ambiguous" — producing fabricated legal principles or
precedent citations that mimic the *style* of real legal reasoning without
the *substance*. With ~479 examples (versus the tens of thousands these
research systems train on), sparse/ambiguous input facts would be the
**normal** case here, not an edge case — this is a serious, not
theoretical, risk at the current data scale.

**Mitigation used in research, and its limits:** retrieval-augmented
generation — grounding the explanation in retrieved similar past cases
rather than generating free-form — reduces but does not eliminate this. A
2026 critique specifically of legal RAG ([arxiv.org/pdf/2606.09724](https://arxiv.org/pdf/2606.09724))
argues that similarity-based retrieval alone lacks the structural/temporal/
causal grounding needed for legal relevance, and legal RAG systems are
documented to "cite real documents in ways that are anachronistic,
structurally incomplete, or lacking institutional grounding." One promising
mitigation is producing **structured, auditable "explanation traces"**
(a frontiers-in-AI 2026 paper: [frontiersin.org/.../frai.2026.1905145](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1905145/full))
rather than free text — closer to Level 1's structured fields than to a
narrative paragraph.

**Evaluation:** lexical-overlap metrics (ROUGE-1/2/L, BLEU, METEOR) and
semantic-similarity metrics (BERTScore) are used but treated as
**insufficient on their own** in the research reviewed — legal-expert
Likert-scale review (rating accuracy, relevance, completeness, and
"clarity + linking to outcome" on a 1–5 scale) is described as the
qualitative gold standard. This means Level 3 isn't just an ML engineering
problem — it needs a legal-expert evaluation loop, which is a process
change, not just a modeling change.

**Verdict:** not recommended as a near-term target. The corpus is too
small relative to what research systems use, hallucination risk is
correspondingly higher, and proper evaluation requires legal-expert review
infrastructure that doesn't exist in this project. If pursued at all,
pursue the structured-explanation-trace variant, not free-text generation.

---

## Level 4 — Full generative "predicted proceedings" narrative

**What it would mean:** the literal reading of "exact predicted outcome
and proceedings" — a free-text passage describing what will happen in a
case, written the way a judgment describes what did happen.

**Why this is different in kind, not just degree, from Levels 1–3:**
Levels 1–2 are still classification/regression (bounded, checkable
outputs). Level 3 is generation but grounded and structured. Level 4 is
open-ended natural-language generation presented as a **prediction of real
future events in someone's actual legal matter** — the highest-stakes
framing of everything on this list, and the one explicitly warned against
by `ARCHITECTURE.md`'s existing responsible-use note: *"Surface
uncertainty everywhere; never present a bare verdict."* A fluent narrative
paragraph, even with a disclaimer attached, reads as far more authoritative
than a probability score — that's a framing risk independent of model
accuracy.

### Responsible-use and regulatory intricacies

This is the part of "abstracting to full outcome prediction" that isn't a
modeling problem at all, and is worth taking as seriously as any technical
item above:

- **FTC v. DoNotPay** (finalized January 2025) is the marquee documented
  enforcement action in this space. The FTC alleged an "AI lawyer" product
  never tested whether its output matched a real lawyer's competence and
  never employed attorneys to verify it; the final order bans DoNotPay from
  claiming lawyer-equivalent performance without substantiating evidence,
  imposes $193,000 in monetary relief, and requires notifying past
  subscribers. ([FTC press release](https://www.ftc.gov/news-events/news/press-releases/2025/02/ftc-finalizes-order-donotpay-prohibits-deceptive-ai-lawyer-claims-imposes-monetary-relief-requires))
  This is directly on point: a system whose *output framing* implies more
  certainty/competence than it can substantiate is where the enforcement
  risk lives, independent of whether the underlying technology is
  reasonable.
- **An active (2026) federal lawsuit against OpenAI** alleges ChatGPT
  engaged in unauthorized practice of law after a pro se litigant filed 44
  ChatGPT-drafted filings, including one citing a fabricated case; OpenAI's
  defense is that it's "merely a tool" incapable of practicing law — an
  unresolved test of where liability falls for confident-sounding generated
  legal content. ([Thomson Reuters Institute coverage](https://www.thomsonreuters.com/en-us/posts/ai-in-courts/scaling-justice-unauthorized-practice-of-law/))
- The regulatory tension is genuinely unresolved, not just restrictive:
  unauthorized-practice-of-law rules are contested precisely because ~92%
  of low-income people get no or insufficient legal help, so "protect
  against overclaiming" and "don't chill access-to-justice tools" pull in
  opposite directions ([Thomson Reuters Institute](https://www.thomsonreuters.com/en-us/posts/government/ai-impacts-unauthorized-practice-of-law/)).
  This project sits in exactly that tension if it moves toward
  narrative-style output.

**What this means concretely, if Level 4 is ever pursued:**
1. Never ship free-form narrative generation ungrounded — at minimum, pair
   any generated text with the specific retrieved past cases it's based on
   (Level 3's RAG approach), so a user can check the generation against
   real source material rather than trusting it as-is.
2. Never drop the disclaimer/uncertainty framing that `ml/inference/
   predict.py` already attaches to Level 0 output (`docs/MODEL.md` §10) —
   if anything, Level 4 needs a *stronger* disclaimer than a bare
   probability, since a paragraph reads as more authoritative by default.
3. Treat this as a legal/compliance review item, not just an ML roadmap
   item, before any production exposure — the DoNotPay order shows the
   FTC scrutinizes marketing/framing claims independent of the underlying
   model's actual accuracy.

**Verdict: not recommended given current project scale.** Revisit only
after Levels 1–2 are stable, the dataset is substantially larger, and
there's an explicit decision (with legal/compliance input) about the
product's risk posture — this shouldn't be a decision made by expanding
the model architecture alone.

---

## A different axis of intricacy: what "outcome" and "prediction" even mean here

Independent of which level is pursued, there's a framing issue that
applies to all of them, surfaced by methodological research on this exact
question:

**Medvedeva, Wieling & Vols** ([*Rethinking the field of automatic
prediction of court decisions*](https://link.springer.com/article/10.1007/s10506-021-09306-3))
formally distinguish:
- **Outcome identification** — extracting the already-stated verdict from
  post-decision text. This is what the current Level 0 model actually
  does: `docs/MODEL.md` §4 already documents that `text` is the full
  post-decision judgment, and `extract_outcome_heuristic()` (§3) derives
  the label from that same document's own concluding language.
- **Outcome forecasting** — genuinely predicting an undecided case from
  pre-decision information only (a complaint, an application, arguments
  made before the ruling).

**Medvedeva & McBride** ([*Legal Judgment Prediction: If You Are Going to
Do It, Do It Right*](https://aclanthology.org/2023.nllp-1.9/), NLLP 2023
Best Presentation) found only ~7% of self-styled "LJP" papers actually do
the latter — most, like this project's current model, do the former while
describing it with forecasting language.

**Why this matters for every level above:** all four levels, as currently
scoped, would still train and predict on the full post-decision judgment
text — meaning Levels 1–4 would all still be *outcome identification*
dressed up as richer output, not genuine *forecasting*, unless the input
data pipeline changes too. A structured-field model, a damages regressor,
and a narrative generator are all equally capable of "predicting" facts
that are already stated in their own input if that input is the decided
judgment.

**If genuine forecasting is the actual goal** (predicting a case's outcome
*before* it's decided, from a complaint/application/pre-hearing
submissions), that requires a parallel change independent of which output
level is chosen: **sourcing pre-decision documents** for the input side,
not just richer labels for the output side. `ml/ingestion/courtlistener.py`
already carries this exact warning in its docstring for the US source;
`elitigation.sg`, the current Singapore source, does not expose
pre-decision filings the way it does published judgments (`docs/MODEL.md`
§2 — no "State Courts" filter was found either, meaning even in-progress
proceedings would be unreachable through this source). This is arguably a
bigger, more fundamental intricacy than any single output level above —
it's a data-sourcing problem, not a modeling problem, and no model
architecture change fixes it.

---

## Recommended path

1. **Do not pursue Level 4.** The regulatory precedent and hallucination
   risk are real and not offset by anything this project's current scale
   can mitigate.
2. **Level 1 (structured `costs_to`/`costs_amount_bucket`/`relief_type`)
   is the realistic near-term target**, and should happen *after*
   `docs/MODEL_IMPROVEMENTS.md`'s Phase 0–1 (fix leakage, fix imbalance) —
   adding more prediction targets to a model with known leakage and
   overfitting problems compounds those problems across more outputs
   rather than fixing anything.
3. **Level 2 (quantum) limited to `costs`, binned, not full damages
   regression** — and consider whether a different, damages-heavier case
   type is a better source for this specific target than the current
   bankruptcy/insolvency corpus.
4. **Level 3 (explanation generation), if pursued at all, only in the
   structured/RAG-grounded form**, with a legal-expert review loop, and
   only once Levels 1–2 are stable — this is a multi-month research
   undertaking, not an incremental feature.
5. **Separately and in parallel, investigate whether genuine pre-decision
   forecasting is even possible from available Singapore data sources** —
   this determines whether the project is honestly building outcome
   forecasting or outcome identification, which should be stated plainly
   in `docs/MODEL.md` regardless of which output level is implemented.

---

## Sources

- TOPJUDGE (multi-task LJP) — [aclanthology.org/D18-1390](https://aclanthology.org/D18-1390.pdf)
- CAIL2018 — [arxiv.org/pdf/1807.02478](https://arxiv.org/pdf/1807.02478)
- Damages/compensation regression (PeerJ) — [peerj.com/articles/cs-1225](https://peerj.com/articles/cs-1225/)
- Damages-amount ML comparison — [researchgate.net](https://www.researchgate.net/publication/380885719_Predicting_the_Amount_of_Compensation_for_Harm_Awarded_by_Courts_Using_Machine-Learning_Algorithms)
- NyayaRAG — [arxiv.org/html/2508.00709v2](https://arxiv.org/html/2508.00709v2)
- PredEx — [arxiv.org/pdf/2406.04136](https://arxiv.org/pdf/2406.04136)
- NyayaAnumana & INLegalLlama — [arxiv.org/pdf/2412.08385](https://arxiv.org/pdf/2412.08385)
- RAG limitations in legal domain — [arxiv.org/pdf/2606.09724](https://arxiv.org/pdf/2606.09724)
- Structured "explanation trace" generation — [frontiersin.org](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1905145/full)
- FTC v. DoNotPay — [ftc.gov press release](https://www.ftc.gov/news-events/news/press-releases/2025/02/ftc-finalizes-order-donotpay-prohibits-deceptive-ai-lawyer-claims-imposes-monetary-relief-requires)
- OpenAI UPL lawsuit coverage — [thomsonreuters.com](https://www.thomsonreuters.com/en-us/posts/ai-in-courts/scaling-justice-unauthorized-practice-of-law/)
- UPL regulatory tension — [thomsonreuters.com](https://www.thomsonreuters.com/en-us/posts/government/ai-impacts-unauthorized-practice-of-law/)
- Medvedeva, Wieling & Vols — [link.springer.com/10.1007/s10506-021-09306-3](https://link.springer.com/article/10.1007/s10506-021-09306-3)
- Medvedeva & McBride — [aclanthology.org/2023.nllp-1.9](https://aclanthology.org/2023.nllp-1.9/)

---

## Changelog

### 2026-08-22 — Document created
- Defined four target levels between the current single-label classifier
  and a fully generative "predicted proceedings" output, with data,
  architecture, evaluation, and regulatory considerations for each.
- No pipeline code changed by this document.
