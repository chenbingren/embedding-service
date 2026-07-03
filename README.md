# embedding-service

独立向量化服务，对外提供 OpenAI 兼容的 `/v1/embeddings` 接口。

## Local run

```bash
pip install -r requirements.txt
set EMBEDDING_MODEL_PATH=D:\models\piccolo-large-zh
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

## Docker

```bash
bash build.sh
MODEL_DIR=/data/models/piccolo-large-zh EMBEDDING_MODEL_NAME=local-piccolo-large-zh bash run.sh
```

服务内访问地址：

```text
http://embedding-service:8002/v1/embeddings
```
