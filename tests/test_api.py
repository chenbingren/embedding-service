import unittest

from fastapi.testclient import TestClient

from app.config import EmbeddingSettings
from app.engine import resolve_backend
from app.main import app


class FakeEmbeddingEngine:
    model_name = "fake-embedding"

    def embed(self, texts):
        return [[float(index), float(index + 1), float(index + 2)] for index, _ in enumerate(texts)]


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

    def test_auto_backend_uses_model_name_for_piccolo_mounts(self):
        settings = EmbeddingSettings(
            model_path="/models/default",
            model_name="local-piccolo-large-zh",
            backend="auto",
        )

        self.assertEqual(resolve_backend(settings), "transformers")


if __name__ == "__main__":
    unittest.main()
