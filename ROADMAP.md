# Legal Prediction — Build Roadmap

A prioritized checklist from raw data to production deployment.

---

## Phase 1 — Get Real Data

- [ ] Get a CourtListener API token at courtlistener.com and add it to `.env` as `COURTLISTENER_API_TOKEN`
- [ ] Flesh out the ingestion stub in `ml/ingestion/courtlistener.py` — complete API pagination and field mapping
- [ ] Pull your first dataset — start narrow (e.g., one circuit, one case type) to get ~1,000+ labeled cases into `data/raw/`
- [ ] Run the preprocessing pipeline — `clean_record()` in `ml/preprocessing/clean.py` maps raw API JSON to the processed schema; verify `strip_leakage()` catches outcome-announcing phrases for your chosen case type
- [ ] Validate your `cases.csv` — confirm `text` is pre-decision only, `outcome` is binary, and `filed_date` is populated (required for temporal split)

---

## Phase 2 — Train and Evaluate the Baseline

- [ ] Train the model: `python -m training.train_baseline` from the `ml/` directory — produces `ml/registry/latest.json` and a `.joblib` artifact
- [ ] Check for leakage — if `roc_auc > 0.95` in `latest.json`, suspect leakage; re-examine `strip_leakage()` patterns in `ml/preprocessing/clean.py`
- [ ] Run bias audit — call `group_metrics()` from `ml/evaluation/metrics.py` on the test set, grouping by `court`, `case_type`, and `represented`
- [ ] Tune the baseline — adjust TF-IDF params (`max_features`, `ngram_range`) and `C` in LogisticRegression in `ml/training/train_baseline.py`; check if text-length features from `add_text_length()` help

---

## Phase 3 — Improve the Model

- [ ] Add cross-validation — replace single `temporal_split()` with an expanding-window time-series CV loop for stable metric estimates
- [ ] Experiment with better models — swap in `GradientBoostingClassifier` or `XGBClassifier` in `ml/training/train_baseline.py` while keeping the same sklearn Pipeline contract
- [ ] Try LEGAL-BERT — replace TF-IDF with `nlpaueb/legal-bert-base-uncased` embeddings; update `backend/app/services/inference_service.py` for the new serialization format
- [ ] Add calibration — wrap the final classifier with `sklearn.calibration.CalibratedClassifierCV` so probability outputs are well-calibrated

---

## Phase 4 — Wire Up the Backend

- [ ] Test the prediction endpoint locally: `uvicorn app.main:app --reload` in `backend/`, then `POST /predict` with a sample case JSON
- [ ] Verify the health check: `GET /health` should return `model_ready: true` once the artifact exists at `MODEL_DIR`
- [ ] Wire up Postgres — the `db` service is in `docker-compose.yml` but nothing writes to it; add audit logging of predictions (case hash, probability, timestamp) using SQLAlchemy in `backend/app/`
- [ ] Add input validation — `backend/app/schemas/prediction.py` accepts any string; add min/max length constraints on `text` and reject empty submissions

---

## Phase 5 — Run the Full Stack

- [ ] Copy and fill `.env`: `cp .env.example .env` — set `MODEL_DIR`, `ALLOWED_ORIGINS`, and `DATABASE_URL`
- [ ] Build and run with Docker: `docker compose up --build` — frontend on `:5173`, backend on `:8000`, Postgres on `:5432`
- [ ] Test the UI end-to-end — paste a real case brief into the form, confirm the probability meter and confidence display render correctly, check the browser console for API errors
- [ ] Fix the frontend proxy — verify the Vite proxy in `frontend/vite.config.ts` works in Docker (the static build may need nginx to handle `/api/*` proxying in production)

---

## Phase 6 — CI/CD and Production Readiness

- [ ] Add backend tests — `backend/tests/` is mostly empty; write a test that loads the artifact and checks `POST /predict` returns a valid response
- [ ] Pin dependencies — run `pip freeze` in `ml/` and `backend/` and lock versions for reproducible training
- [ ] Add model versioning to the UI — the `/model` endpoint already returns `version` and `trained_at`; surface these in the frontend so users can see which model scored their case
- [ ] Set up monitoring — add structured JSON logging in `inference_service.run_prediction()` to track prediction distributions over time and detect model drift
