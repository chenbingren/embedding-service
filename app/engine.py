from __future__ import annotations

import logging
from typing import Iterable, List

from app.config import EmbeddingSettings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self, settings: EmbeddingSettings | None = None):
        self.settings = settings or get_settings()
        self.model_name = self.settings.resolved_model_name
        self._backend = None
        self._backend_type = ""

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        self._load()
        embeddings: List[List[float]] = []
        for batch in self._batches(texts, self.settings.batch_size):
            if self._backend_type == "sentence-transformers":
                embeddings.extend(self._embed_with_sentence_transformers(batch))
            else:
                embeddings.extend(self._embed_with_transformers(batch))
        return embeddings

    def _load(self) -> None:
        if self._backend is not None:
            return

        backend = resolve_backend(self.settings)

        if backend in {"sentence-transformers", "sentence_transformers"}:
            from sentence_transformers import SentenceTransformer

            logger.info("loading sentence-transformers model: %s", self.settings.model_path)
            self._backend = SentenceTransformer(self.settings.model_path, device=self.settings.device)
            self._backend_type = "sentence-transformers"
            return

        if backend in {"transformers", "piccolo"}:
            from transformers import AutoModel, AutoTokenizer

            logger.info("loading transformers embedding model: %s", self.settings.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.settings.model_path)
            model = AutoModel.from_pretrained(self.settings.model_path)
            model.eval()
            if self.settings.device != "cpu":
                model.to(self.settings.device)
            self._backend = (tokenizer, model)
            self._backend_type = "transformers"
            return

        raise ValueError(f"unsupported EMBEDDING_BACKEND: {self.settings.backend}")

    def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        vectors = self._backend.encode(
            texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize,
            show_progress_bar=False,
        )
        return vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)

    def _embed_with_transformers(self, texts: List[str]) -> List[List[float]]:
        import torch

        tokenizer, model = self._backend
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.settings.max_length,
                return_tensors="pt",
            )
            if self.settings.device != "cpu":
                inputs = {key: value.to(self.settings.device) for key, value in inputs.items()}
            outputs = model(**inputs)
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            if self.settings.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().tolist()

    @staticmethod
    def _mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

    @staticmethod
    def _batches(values: List[str], batch_size: int) -> Iterable[List[str]]:
        safe_batch_size = max(1, batch_size)
        for index in range(0, len(values), safe_batch_size):
            yield values[index : index + safe_batch_size]


def resolve_backend(settings: EmbeddingSettings) -> str:
    backend = settings.backend.lower().strip()
    if backend != "auto":
        return backend

    model_hint = f"{settings.model_path} {settings.resolved_model_name}".lower()
    return "transformers" if "piccolo" in model_hint else "sentence-transformers"
