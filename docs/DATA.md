# Data sourcing

The model is only as good — and as fair — as the corpus behind it. This is the
hardest part of the project, so treat it with more care than the code.

## Where to get cases

- **CourtListener / RECAP** (Free Law Project) — US opinions and federal
  dockets, with an API and bulk data. A starter ingester lives at
  `ml/ingestion/courtlistener.py`. Get a free API token and respect rate limits.
- **Caselaw Access Project** (Harvard) — millions of US cases, now fully open.
  Good for bulk historical text.
- **Supreme Court Database** (Spaeth) — richly hand-coded SCOTUS metadata. The
  easiest place to get a clean, labeled dataset for a first model.
- Jurisdiction-specific court portals if you target a particular country.

Always read and follow each source's terms of use.

## The processed schema

`ml/training/train_baseline.py` expects `data/processed/cases.csv` with:

| Column | Meaning |
|--------|---------|
| `text` | Pre-decision text only (complaint, brief, alleged facts) |
| `filed_date` | ISO date used for the chronological train/test split |
| `outcome` | Binary label (0/1) |
| `court`, `case_type`, `represented` | Optional metadata (configurable) |

## The one rule that matters most

Court **opinions are written after the decision** and broadcast the outcome.
If opinion text reaches the `text` column, the model is reading the answer, not
predicting it — and your metrics will look great while the model is useless.
Use opinions only to derive `outcome`; source `text` from genuinely
pre-decision documents. `ml/preprocessing/clean.py` includes a crude leakage
filter, but preferring real pre-decision documents beats scrubbing opinions.

## Labeling

Decide your target precisely before you start: affirm/reverse, plaintiff/
defendant prevails, motion granted/denied, etc. A narrow, consistently codeable
label beats a vague one every time.
