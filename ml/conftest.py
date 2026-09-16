"""Make `ml/` importable as the package root regardless of where pytest is
invoked from, matching the `python -m training.train_baseline` convention
documented in ROADMAP.md (absolute imports like `from evaluation.metrics
import ...` assume `ml/` itself is on sys.path).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
