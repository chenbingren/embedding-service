from __future__ import annotations

import logging
from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.engine import EmbeddingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

app = FastAPI(title="Mehup Embedding Service", version="1.0.0")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text or text list to embed")
    model: str | None = None


def _get_engine() -> EmbeddingEngine:
    engine = getattr(app.state, "embedding_engine", None)
    if engine is None:
        engine = EmbeddingEngine(get_settings())
        app.state.embedding_engine = engine
    return engine


def _normalize_input(value: str | List[str]) -> List[str]:
    texts = [value] if isinstance(value, str) else value
    return [text if text is not None else "" for text in texts]


@app.get("/health")
def health():
    engine = _get_engine()
    return {"status": "ok", "model": engine.model_name}


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    texts = _normalize_input(request.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input cannot be empty")

    settings = get_settings()
    if len(texts) > settings.max_batch:
        raise HTTPException(status_code=400, detail=f"input batch too large, max={settings.max_batch}")

    engine = _get_engine()
    vectors = engine.embed(texts)
    model_name = request.model or engine.model_name
    return {
        "object": "list",
        "model": model_name,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
