# Legal Judgment Prediction

A starter monorepo for predicting court case outcomes from pre-decision
information, with a clean separation between the ML pipeline, a FastAPI
backend that serves predictions, and a React frontend.

> **This is a scaffold, not a finished product.** The model shipped here is a
> deliberately simple baseline. Read the methodology notes below before you
> trust any number it produces.

## Architecture

```
ml/        -> produces a versioned model artifact (data -> features -> train -> evaluate)
backend/   -> loads the artifact and exposes it over HTTP (FastAPI)
frontend/  -> talks only to the backend (React + TypeScript)
```

Training and serving are decoupled on purpose: you can retrain without
touching the API, and the API never imports training code.

## Quickstart

### 1. Train a baseline model

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Drop a processed dataset at data/processed/cases.csv first (see docs/DATA.md)
python -m training.train_baseline
```

This writes a model artifact + metadata to `ml/registry/`.

### 2. Run the backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# Docs at http://localhost:8000/docs
```

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
# App at http://localhost:5173
```

### Or run everything with Docker

```bash
cp .env.example .env   # then fill in values
docker compose up --build
```

## Methodology — read this before trusting the model

1. **Avoid data leakage.** Court *opinions* are written after the decision and
   are full of outcome signals. To honestly predict a case *before* it is
   conducted, features must use only pre-decision information (filings, facts
   as alleged, metadata). The preprocessing step is where you enforce this.
2. **Validate by time, never randomly.** Split train/test chronologically so
   you simulate real prediction. A random split leaks the future. See
   `ml/evaluation/metrics.py:temporal_split`.
3. **Expect metadata to dominate.** Much predictive power in this domain comes
   from who (judge, court, representation), not legal substance. Track this.
4. **Audit for bias.** Historical rulings encode historical bias. Report
   group-wise metrics, not just aggregate accuracy. See
   `ml/evaluation/metrics.py:group_metrics`.
5. **Know the legal terrain.** Some jurisdictions restrict analyzing or
   profiling individual judges. Check what is permissible where you operate
   before building judge-specific features.

## Repository layout

| Path | Purpose |
|------|---------|
| `ml/ingestion/` | Pull raw cases from sources (CourtListener stub included) |
| `ml/preprocessing/` | Clean text, dedup, strip post-decision leakage |
| `ml/features/` | Build TF-IDF + metadata feature matrices |
| `ml/training/` | Train scripts and configs |
| `ml/evaluation/` | Metrics, temporal CV, bias audits |
| `ml/inference/` | Load artifact + `predict()` |
| `ml/registry/` | Versioned model artifacts (gitignored) |
| `backend/app/` | FastAPI app, routes, schemas, services |
| `frontend/src/` | React UI |
| `docs/` | Data sourcing and design notes |

## License

Add your own. Note that case data sources have their own terms — respect them.
