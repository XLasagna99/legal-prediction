import { PredictionForm } from "./components/PredictionForm";

export function App() {
  return (
    <main className="shell">
      <header className="masthead">
        <p className="eyebrow">Legal Judgment Prediction</p>
        <h1>Case Outcome Estimator</h1>
        <p className="lede">
          A statistical estimate of how a case might resolve, based on patterns in
          past decisions. It is a research tool, not a verdict and not legal advice.
        </p>
      </header>
      <PredictionForm />
      <footer className="foot">
        Estimates reflect historical patterns, which can carry historical bias.
        Treat every number as a starting point for human judgment.
      </footer>
    </main>
  );
}
