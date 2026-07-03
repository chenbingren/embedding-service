import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class EmbeddingSettings:
    model_path: str = os.getenv("EMBEDDING_MODEL_PATH", "/models/default")
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "")
    backend: str = os.getenv("EMBEDDING_BACKEND", "auto")
    device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    max_batch: int = int(os.getenv("EMBEDDING_MAX_BATCH", "128"))
    normalize: bool = _env_bool("EMBEDDING_NORMALIZE", False)
    max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or self.model_path


def get_settings() -> EmbeddingSettings:
    return EmbeddingSettings()
