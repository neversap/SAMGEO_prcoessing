# Inhouse Inference

独立版 Inhouse 耕地分割推理模块。

该模块刻意与现有的 `server/app` 服务和 `training` 训练工程解耦。只要目标环境具备 Python 依赖、模型配置文件和 checkpoint，就可以将 `inhouse_inference/` 单独拷贝到正式产品服务中运行。

## 输入与输出

- 输入支持任意尺寸 RGB 图片。
- 默认会将图片 resize 到 `512x512` 后送入模型推理。
- 输出 mask 会使用最近邻插值映射回原图尺寸。
- polygon 与 bbox 坐标会同时返回两套：
  - `points_512` / `bbox_512`：模型输入尺寸坐标。
  - `points` / `bbox`：原图尺寸坐标。

## 快速检验

可以使用单张图片快速验证模型、配置和后处理流程是否可用：

```bash
python -m inhouse_inference.quick_check \
  --config /app/configs/finetune/inhouse_unet_effb3_finetune_prue_logcosh.yaml \
  --checkpoint /path/to/best_val_miou.pt \
  --image /path/to/image.png \
  --output /tmp/inhouse_check \
  --device cuda
```

也可以使用面向产品调用的 CLI 别名：

```bash
python -m inhouse_inference.cli \
  --config /app/configs/finetune/inhouse_unet_effb3_finetune_prue_logcosh.yaml \
  --checkpoint /path/to/best_val_miou.pt \
  --image /path/to/image.png \
  --output /tmp/inhouse_check \
  --device cuda
```

输出文件包括：

- `input.png`：原始输入图。
- `resized_input.png`：resize 到模型输入尺寸后的图片。
- `mask_512.png`：模型输入尺寸下的类别 mask。
- `mask_original.png`：映射回原图尺寸的类别 mask。
- `boundary_512.png`：模型输入尺寸下的边界 mask。
- `boundary_original.png`：映射回原图尺寸的边界 mask。
- `overlay_512.png`：模型输入尺寸下的叠加预览。
- `overlay_original.png`：原图尺寸下的叠加预览。
- `result.json`：结构化推理结果。

## Python API

如果后续产品服务希望直接在 Python 代码中调用，可以使用 `InhousePredictor`：

```python
from inhouse_inference import InhousePredictor

predictor = InhousePredictor(
    config_path="config.yaml",
    checkpoint_path="best_val_miou.pt",
    device="cuda",
)
result = predictor.predict("image.png")
print(result.to_json_dict())
```

`InhousePredictor` 初始化时会加载模型配置和 checkpoint。正式服务中应在进程启动时创建一次 predictor，后续请求复用同一个 predictor，避免每次推理重复加载模型。

## HTTP 服务

该模块也提供独立 HTTP 服务入口。服务启动时会加载模型节点，之后每次 `/predict` 请求都会复用常驻内存中的模型。

```bash
export INHOUSE_CONFIG_PATH=/app/configs/finetune/inhouse_unet_effb3_finetune_prue_logcosh.yaml
export INHOUSE_CHECKPOINT_PATH=/models/inhouse/best_val_miou.pt
export INHOUSE_DEVICE=cuda
export INHOUSE_INPUT_SIZE=512
python -m inhouse_inference.service --host 0.0.0.0 --port 8088
```

Docker `CMD` 示例：

```dockerfile
CMD ["python", "-m", "inhouse_inference.service", "--host", "0.0.0.0", "--port", "8088"]
```

服务接口：

- `GET /health`：检查服务是否启动、模型是否完成加载。
- `POST /predict`：上传图片并返回 polygon、bbox、boundary mask 和 segmentation mask。

`/predict` 返回的 mask 与 boundary 使用 RLE `value_counts` 编码，避免直接返回完整像素数组导致响应体过大。

调用示例：

```bash
curl -X POST \
  -F "file=@/path/to/image.png" \
  "http://127.0.0.1:8088/predict?min_area=16&epsilon_ratio=0.003"
```

## 环境变量

- `INHOUSE_CONFIG_PATH`：模型配置文件路径，必填。
- `INHOUSE_CHECKPOINT_PATH`：模型 checkpoint 路径，必填。
- `INHOUSE_DEVICE`：推理设备，默认 `cuda`。
- `INHOUSE_INPUT_SIZE`：模型输入尺寸，默认 `512`。
- `INHOUSE_STRICT_LOAD`：是否严格加载 checkpoint，默认 `0`。
- `INHOUSE_HOST`：HTTP 服务监听地址，默认 `0.0.0.0`。
- `INHOUSE_PORT`：HTTP 服务端口，默认 `8088`。
- `INHOUSE_WORKERS`：Uvicorn worker 数，默认 `1`。

除非明确希望加载多份模型，否则建议保持 `INHOUSE_WORKERS=1`。每个 Uvicorn worker 都会单独加载一份 checkpoint，并额外占用 GPU 显存。

## 依赖

独立部署时可以安装：

```bash
pip install -r requirements-inhouse-inference.txt
```

如果部署环境已经有匹配 CUDA 版本的 PyTorch，也可以先安装对应的 `torch` / `torchvision`，再安装其余依赖。
