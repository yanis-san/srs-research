"""Settings read from environment variables (prefix SRS_) or a local .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SRS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_path: Path = PROJECT_ROOT / "srs_biometrics.db"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Used by the Streamlit client.
    api_base_url: str = "http://127.0.0.1:8000"

    @property
    def database_url(self) -> str:
        # as_posix() avoids backslashes in the URL on Windows.
        return f"sqlite:///{self.database_path.as_posix()}"


settings = Settings()
