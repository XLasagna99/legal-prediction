// Typed client for the prediction backend.
// In dev, Vite proxies /api/* to http://localhost:8000 (see vite.config.ts).

export interface CaseInput {
  text: string;
  court?: string;
  case_type?: string;
  represented?: string;
}

export interface PredictionResponse {
  label: number;
  probability: number;
  confidence: number;
  model_version: string;
  disclaimer: string;
}

const BASE = "/api";

export async function predictCase(input: CaseInput): Promise<PredictionResponse> {
  const resp = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });

  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed (${resp.status})`);
  }
  return resp.json();
}
