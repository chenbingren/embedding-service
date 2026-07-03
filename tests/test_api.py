import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import EmbeddingSettings, get_settings
from app.engine import resolve_backend
from app.main import app


class FakeEmbeddingEngine:
    model_name = "fake-embedding"

    def embed(self, texts):
        return [[float(index), float(index + 1), float(index + 2)] for index, _ in enumerate(texts)]


class BrokenEmbeddingEngine:
    model_name = "/models/default"

    def embed(self, texts):
        raise ValueError("Path /models/default not found")


class EmbeddingApiTest(unittest.TestCase):
    def setUp(self):
        app.state.embedding_engine = FakeEmbeddingEngine()
        self.client = TestClient(app)

    def tearDown(self):
        if hasattr(app.state, "embedding_engine"):
            delattr(app.state, "embedding_engine")

    def test_openai_compatible_embeddings_endpoint(self):
        response = self.client.post(
            "/v1/embeddings",
            json={"model": "fake-embedding", "input": ["hello", "world"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["model"], "fake-embedding")
        self.assertEqual(len(payload["data"]), 2)
        self.assertEqual(payload["data"][0]["object"], "embedding")
        self.assertEqual(payload["data"][0]["index"], 0)
        self.assertEqual(payload["data"][0]["embedding"], [0.0, 1.0, 2.0])
        self.assertEqual(payload["usage"]["prompt_tokens"], 0)

    def test_health_uses_injected_engine(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "fake-embedding")

    def test_standalone_test_embedding_endpoint_returns_dimension_and_sample(self):
        response = self.client.post(
            "/test/embedding",
            json={"text": "hello", "sampleSize": 2},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["model"], "fake-embedding")
        self.assertEqual(payload["dim"], 3)
        self.assertEqual(payload["vectorSample"], [0.0, 1.0])
        self.assertEqual(payload["text"], "hello")

    def test_standalone_test_embedding_get_endpoint_is_easy_to_curl(self):
        response = self.client.get("/test/embedding", params={"text": "hello", "sampleSize": 1})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["dim"], 3)
        self.assertEqual(payload["vectorSample"], [0.0])

    def test_standalone_test_embedding_returns_clear_error_when_model_is_missing(self):
        app.state.embedding_engine = BrokenEmbeddingEngine()

        response = self.client.get("/test/embedding", params={"text": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("EMBEDDING_MODEL_PATH", response.json()["detail"])
        self.assertIn("/models/default", response.json()["detail"])

    def test_openai_compatible_endpoint_returns_clear_error_when_model_is_missing(self):
        app.state.embedding_engine = BrokenEmbeddingEngine()

        response = self.client.post("/v1/embeddings", json={"input": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("EMBEDDING_MODEL_PATH", response.json()["detail"])

    def test_auto_backend_uses_model_name_for_piccolo_mounts(self):
        settings = EmbeddingSettings(
            model_path="/models/default",
            model_name="local-piccolo-large-zh",
            backend="auto",
        )

        self.assertEqual(resolve_backend(settings), "transformers")

    def test_get_settings_loads_env_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "EMBEDDING_MODEL_PATH=D:\\models\\piccolo-large-zh",
                        "EMBEDDING_MODEL_NAME=local-piccolo-large-zh",
                        "EMBEDDING_BACKEND=transformers",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = get_settings(env_file)

        self.assertEqual(settings.model_path, "D:\\models\\piccolo-large-zh")
        self.assertEqual(settings.model_name, "local-piccolo-large-zh")
        self.assertEqual(settings.backend, "transformers")


if __name__ == "__main__":
    unittest.main()
