import { useState } from "react";
import { predictCase, type PredictionResponse } from "../api/client";

export function PredictionForm() {
  const [text, setText] = useState("");
  const [court, setCourt] = useState("");
  const [caseType, setCaseType] = useState("");
  const [represented, setRepresented] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleEstimate() {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await predictCase({
        text,
        court: court || undefined,
        case_type: caseType || undefined,
        represented: represented || undefined,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const pct = result ? Math.round(result.probability * 100) : 0;

  return (
    <div className="card">
      <label className="field">
        <span>Case facts (pre-decision only)</span>
        <textarea
          rows={8}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste the complaint, brief, or alleged facts. Do not include the court's opinion."
        />
      </label>

      <div className="grid">
        <label className="field">
          <span>Court</span>
          <input value={court} onChange={(e) => setCourt(e.target.value)} placeholder="e.g. ca9" />
        </label>
        <label className="field">
          <span>Case type</span>
          <input value={caseType} onChange={(e) => setCaseType(e.target.value)} placeholder="e.g. civil" />
        </label>
        <label className="field">
          <span>Represented</span>
          <select value={represented} onChange={(e) => setRepresented(e.target.value)}>
            <option value="">unknown</option>
            <option value="yes">yes</option>
            <option value="no">no</option>
          </select>
        </label>
      </div>

      <button className="primary" onClick={handleEstimate} disabled={loading || text.trim().length === 0}>
        {loading ? "Estimating\u2026" : "Estimate outcome"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <div className="meter" role="img" aria-label={`Estimated probability ${pct} percent`}>
            <div className="meter-fill" style={{ width: `${pct}%` }} />
            <div className="meter-mark" />
          </div>
          <p className="readout">
            <strong>{pct}%</strong> estimated probability of outcome class{" "}
            <strong>{result.label}</strong>
            <span className="confidence"> \u00b7 confidence {Math.round(result.confidence * 100)}%</span>
          </p>
          <p className="disclaimer">{result.disclaimer}</p>
          <p className="version">model {result.model_version}</p>
        </div>
      )}
    </div>
  );
}
