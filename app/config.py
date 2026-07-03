import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def load_project_env(env_file: str | Path | None = None) -> None:
    dotenv_path = Path(env_file) if env_file else Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)


@dataclass(frozen=True)
class EmbeddingSettings:
    model_path: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_PATH", "/models/default"))
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", ""))
    backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "auto"))
    device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 8))
    max_batch: int = field(default_factory=lambda: _env_int("EMBEDDING_MAX_BATCH", 128))
    normalize: bool = field(default_factory=lambda: _env_bool("EMBEDDING_NORMALIZE", False))
    max_length: int = field(default_factory=lambda: _env_int("EMBEDDING_MAX_LENGTH", 512))

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or self.model_path


def get_settings(env_file: str | Path | None = None) -> EmbeddingSettings:
    load_project_env(env_file)
    return EmbeddingSettings()
