# Legal Prediction — Build Roadmap

A prioritized checklist from raw data to production deployment.

**Status note (2026-08-30):** Phase 1–2 were completed via a different path
than originally written below — see the inline notes on each item. The
data source pivoted from CourtListener (no Singapore coverage — see
`docs/MODEL.md` §2) to Singapore Courts' `elitigation.sg` portal, and the
ML work has outgrown this checklist's detail: `docs/MODEL.md` (current
pipeline state), `docs/MODEL_IMPROVEMENTS.md` (the detailed successor
roadmap to Phase 2/3 below, evidence-based and phased), and
`docs/MODEL_RESULTS.md` (tabulated results per change) are now the sources
of truth for ML work — this file stays the high-level checklist.

---

## Phase 1 — Get Real Data

- [ ] ~~Get a CourtListener API token at courtlistener.com and add it to `.env` as `COURTLISTENER_API_TOKEN`~~ (superseded — CourtListener has no Singapore coverage)
- [ ] ~~Flesh out the ingestion stub in `ml/ingestion/courtlistener.py` — complete API pagination and field mapping~~ (superseded — the pagination bug was fixed, then this source was abandoned in favor of Singapore Courts)
- [x] Pull your first dataset — done via `ml/ingestion/singapore_courts.py` instead (elitigation.sg, no API/token needed): 1,000 raw Singapore bankruptcy/insolvency judgments in `data/raw/singapore_bankruptcy_cases.jsonl`
- [x] Run the preprocessing pipeline — `clean_singapore_record()` + `extract_outcome_heuristic()` in `ml/preprocessing/clean.py` (a source-specific variant of `clean_record()`/`strip_leakage()`, since the source's field names and disposition phrasing differ), run via `ml/preprocessing/build_cases_csv.py` — see `docs/MODEL.md` §3–4
- [~] Validate your `cases.csv` — `filed_date` is populated ✅. `text` is **not** pre-decision-only (it's the full post-decision judgment — a known, documented deviation, not an oversight: `docs/MODEL.md` §4, §11 item 4) and `outcome` is **not** binary (5 categories, not 2 — a deliberate scope expansion: `docs/MODEL.md` §3, `docs/OUTCOME_PREDICTION.md`). Don't check this box as "passing" the original binary/pre-decision bar — it deliberately doesn't meet it; the deviations are tracked, not hidden

---

## Phase 2 — Train and Evaluate the Baseline

- [x] Train the model: `python -m training.train_baseline` from the `ml/` directory — done repeatedly; latest results in `docs/MODEL_RESULTS.md`
- [x] Check for leakage — `roc_auc` has never exceeded ~0.66, well under the 0.95 red flag. Went further than the checklist's threshold check: found and fixed two *real* leakage bugs by directly inspecting fitted model coefficients (struck_out/withdrawn disposition words leaking into their own classes — `docs/MODEL.md` §4, 2026-08-28 Changelog) rather than waiting for the AUC symptom
- [x] Run bias audit — `ml/evaluation/bias_audit.py` (new) wraps `group_metrics()` around the latest registry model's exact test split, grouped by `court` (`case_type`/`represented` skipped — still constant `"unknown"` for every row, `docs/MODEL.md` §7). Results: `docs/MODEL_IMPROVEMENTS.md` Phase 4.3
- [ ] Tune the baseline — TF-IDF params, `C` in LogisticRegression, `add_text_length()` — **not yet done**. This is now `docs/MODEL_IMPROVEMENTS.md` Phase 0.3–0.5 (in progress) and Phase 2 (transformer swap), which supersede this checklist item with a researched, evidence-based plan

---

## Phase 3 — Improve the Model

See `docs/MODEL_IMPROVEMENTS.md` for the current, more detailed version of
this phase (phased 0–4, each item backed by a specific diagnosed problem
and cited research) and `docs/OUTCOME_PREDICTION.md` for what it would take
to go beyond a single disposition label. Original items kept below for
reference; none are done yet.

- [ ] Add cross-validation — replace single `temporal_split()` with an expanding-window time-series CV loop for stable metric estimates
- [ ] Experiment with better models — swap in `GradientBoostingClassifier` or `XGBClassifier` in `ml/training/train_baseline.py` while keeping the same sklearn Pipeline contract
- [ ] Try LEGAL-BERT — replace TF-IDF with `nlpaueb/legal-bert-base-uncased` embeddings; update `backend/app/services/inference_service.py` for the new serialization format (`docs/MODEL_IMPROVEMENTS.md` Phase 2 recommends InLegalBERT instead — closer to Singapore's Commonwealth common-law system than US-trained Legal-BERT)
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
