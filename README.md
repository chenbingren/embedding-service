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


curl.exe -X POST "http://127.0.0.1:8002/v1/embeddings" ` -H "Content-Type: application/json" ` -d '{"input":"这是一个向量化测试文本","model":"test"}'

curl.exe -G "http://127.0.0.1:8002/test/embedding" `
  --data-urlencode "text=这是一个向量化测试文本" `
  --data-urlencode "sampleSize=8"

curl.exe -G "http://127.0.0.1:8002/test/embedding" `
  --data-urlencode "text=hello" `
  --data-urlencode "sampleSize=8"