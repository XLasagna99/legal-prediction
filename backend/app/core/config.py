"""Application settings, read from environment / .env."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Default to the ml/registry dir for local dev; override with MODEL_DIR in prod.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_DIR = _REPO_ROOT / "ml" / "registry"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "legal-judgment-prediction"
    model_dir: Path = _DEFAULT_MODEL_DIR
    allowed_origins: str = "http://localhost:5173"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
