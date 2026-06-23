# SAM GEO

一个用于在服务器端快速搭建 SAM3 分割测试服务的最小项目骨架。本地只保存代码；模型权重、GPU 环境和推理服务部署在服务器上。

## 结构

```text
SAM_GEO/
  client/                 # 本地调用服务端 API 的测试客户端
  server/                 # FastAPI 推理服务
  scripts/                # 部署辅助脚本
  docker-compose.yml      # 服务器端启动入口
  Dockerfile.server       # 服务端镜像
```

## 快速开始

### 1. 本地准备代码

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-client.txt
```

### 2. 上传并启动服务器

编辑 `.env.example` 中的配置，或者在服务器上创建 `.env`。

```powershell
.\scripts\deploy_server.ps1 -Host user@your-server -RemotePath /opt/sam-geo
```

在服务器上：

```bash
cd /opt/sam-geo
cp .env.example .env
docker compose up -d --build
```

默认服务地址：

```text
http://SERVER_IP:8080
```

打开这个地址会进入前端测试页面，可以上传图片、输入文本 prompt，并查看 mask 叠加结果。

### 3. 命令行测试分割

```powershell
python .\client\segment_image.py `
  --server http://SERVER_IP:8080 `
  --image .\examples\test.jpg `
  --prompt "building" `
  --out .\outputs\mask.png
```

## SAM3 接入点

服务端通过 `server/app/adapters/` 做模型适配：

- `mock.py`：无模型时用于验证 API、部署和前后端链路。
- `sam3.py`：SAM3 真实模型接入点，服务器上安装官方依赖和权重后在这里加载。

环境变量：

```env
SAM_GEO_BACKEND=mock
SAM_GEO_MODEL_HOST_PATH=./models/sam3
SAM_GEO_MODEL_DIR=/models/sam3
SAM_GEO_CHECKPOINT_PATH=/models/sam3/sam3.1_multiplex.pt
SAM_GEO_CODE_HOST_PATH=
SAM_GEO_CODE_PATH=
SAM_GEO_SAM3_ENTRYPOINT=
SAM_GEO_DEVICE=6,7
SAM_GEO_INFERENCE_DTYPE=bfloat16
SAM_GEO_HOST=0.0.0.0
 SAM_GEO_PORT=8000
 SAM_GEO_HOST_PORT=8080
```

Docker build 会从项目根目录的 `sam3-main.zip` 离线安装官方 SAM3 Python 包，并以
zip 内的 `pyproject.toml` 和 `notebooks` extra 作为依赖来源，避免构建时再去 GitHub
clone 或手写官方依赖。SAM3 image API 需要的 `einops` 来自这个官方 extra。
PyTorch 按官方要求在镜像中安装：
`torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128`。

服务启动时会预加载 SAM3 模型；默认 `SAM_GEO_DEVICE=6,7` 会设置
`CUDA_VISIBLE_DEVICES=6,7`，容器内模型和图片预处理会使用逻辑设备 `cuda:0`。
模型加载使用本地 `SAM_GEO_CHECKPOINT_PATH`，不会在启动时从 Hugging Face 下载权重。
推理默认使用 `SAM_GEO_INFERENCE_DTYPE=bfloat16`，与官方 SAM3 CUDA 推理路径保持一致。

当服务器端 SAM3 环境准备好后，把 `SAM_GEO_BACKEND` 改成 `sam3`，并在 `server/app/adapters/sam3.py` 中完成官方模型 API 的具体调用。

如果需要把 SAM3 模型文件一并打进 Docker 镜像，使用 `Dockerfile.server.baked-model` 和
`docker-compose.baked-model.yml`。详细步骤见 `docs/model_in_docker.md`。

真实 SAM3 推理默认使用官方 image API：
`build_sam3_image_model()` + `Sam3Processor`。如果服务器有自定义封装，也可以通过
`SAM_GEO_SAM3_ENTRYPOINT` 接入桥接模块。详细约定见 `docs/sam3_bridge.md`。

## API

### `GET /health`

返回服务状态和当前 backend。

### `POST /segment`

multipart/form-data:

- `image`: 图像文件。
- `prompt`: 文本概念提示，例如 `building`、`road`、`tree`。
- `box`: 可选，`x1,y1,x2,y2`。
- `points`: 可选，JSON 字符串，例如 `[[100,120,1],[240,260,0]]`，第三列为 label。

返回：

```json
{
  "backend": "mock",
  "width": 1024,
  "height": 768,
  "masks": [
    {
      "score": 1.0,
      "bbox": [256, 192, 768, 576],
      "png_base64": "..."
    }
  ]
}
```
