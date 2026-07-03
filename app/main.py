from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Union

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.engine import EmbeddingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

app = FastAPI(title="Mehup Embedding Service", version="1.0.0")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(..., description="Text or text list to embed")
    model: str | None = None


class TestEmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to embed")
    sampleSize: int = Field(8, ge=0, le=32, description="Number of vector values to return")


def _get_engine() -> EmbeddingEngine:
    engine = getattr(app.state, "embedding_engine", None)
    if engine is None:
        engine = EmbeddingEngine(get_settings())
        app.state.embedding_engine = engine
    return engine


def _normalize_input(value: str | List[str]) -> List[str]:
    texts = [value] if isinstance(value, str) else value
    return [text if text is not None else "" for text in texts]


def _embed_texts(engine, texts: List[str]) -> List[List[float]]:
    try:
        return engine.embed(texts)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"embedding model is not ready: {exc}. "
                "Set EMBEDDING_MODEL_PATH to a valid local model directory "
                "or a supported SentenceTransformer model name."
            ),
        ) from exc


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
    vectors = _embed_texts(engine, texts)
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


def _run_test_embedding(text: str, sample_size: int):
    engine = _get_engine()
    vector = _embed_texts(engine, [text])[0]
    sample_size = min(sample_size, len(vector))
    return {
        "success": True,
        "model": engine.model_name,
        "text": text,
        "dim": len(vector),
        "vectorSample": vector[:sample_size],
    }


@app.post("/test/embedding")
def test_embedding(request: TestEmbeddingRequest):
    return _run_test_embedding(request.text, request.sampleSize)


@app.get("/test/embedding")
def test_embedding_get(
    text: str = Query(..., min_length=1),
    sampleSize: int = Query(8, ge=0, le=32),
):
    return _run_test_embedding(text, sampleSize)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("EMBEDDING_HOST", "0.0.0.0"),
        port=int(os.getenv("EMBEDDING_PORT", "8002")),
    )
