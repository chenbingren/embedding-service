# embedding-service

独立向量化服务，对外提供 OpenAI 兼容的 `/v1/embeddings` 接口。

## Local run

```bash
pip install -r requirements.txt
set EMBEDDING_MODEL_PATH=models/piccolo-large-zh
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

## Docker

Default model directory: `./models/piccolo-large-zh`.

```bash
# build image and start container
EMBEDDING_MODEL_NAME=local-piccolo-large-zh bash build.sh

# build image only
RUN_CONTAINER=false bash build.sh

# restart container with an existing image
BUILD_IMAGE=false bash build.sh

# optional: join a Docker network only when container-name DNS is required
DOCKER_NETWORK=mehup-ai bash build.sh
```

服务内访问地址：

```text
http://<server-ip>:8002/v1/embeddings
```


curl.exe -X POST "http://127.0.0.1:8002/v1/embeddings" ` -H "Content-Type: application/json" ` -d '{"input":"这是一个向量化测试文本","model":"test"}'

curl.exe -G "http://127.0.0.1:8002/test/embedding" `
  --data-urlencode "text=这是一个向量化测试文本" `
  --data-urlencode "sampleSize=8"

curl -G "http://127.0.0.1:8002/test/embedding" --data-urlencode "text=这是一个向量化测试文本" --data-urlencode "sampleSize=8"