# SAM3 bridge

`server/app/adapters/sam3.py` loads real SAM3 inference through a small bridge
module. This keeps the FastAPI service independent from the exact SAM3 checkout
or internal server wrapper.

## Required environment variables

```env
SAM_GEO_BACKEND=sam3
SAM_GEO_MODEL_HOST_PATH=/home/nvme1/rx/models/
SAM_GEO_MODEL_DIR=/models/sam3
SAM_GEO_CHECKPOINT_PATH=/models/sam3/sam3.1_multiplex.pt
SAM_GEO_CODE_HOST_PATH=
SAM_GEO_CODE_PATH=
SAM_GEO_SAM3_ENTRYPOINT=
SAM_GEO_DEVICE=6,7
SAM_GEO_INFERENCE_DTYPE=bfloat16
```

With an empty `SAM_GEO_SAM3_ENTRYPOINT`, the service uses the official SAM3
image API:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

The Docker image installs `sam3-main.zip` from the project root during build, so
`SAM_GEO_CODE_PATH` can stay empty for the normal official API path.

## Optional custom bridge

If the server uses a custom wrapper, set `SAM_GEO_SAM3_ENTRYPOINT` to a
`module:function` value. With this entrypoint:

```env
SAM_GEO_SAM3_ENTRYPOINT=server_sam3_adapter:create_predictor
```

place this file on the server host:

```text
/opt/sam3/server_sam3_adapter.py
```

Use `docs/server_sam3_adapter_example.py` as the starting template.

## Bridge contract

The entrypoint must be formatted as:

```text
module:function
```

The function receives:

```python
create_predictor(model_dir: str, device: str)
```

It must return either:

- a callable accepting `image`, `prompt`, `box`, `points`
- an object with `predict(image, prompt, box, points)`

The prediction output can be:

```python
{"masks": masks, "scores": scores}
```

or just:

```python
masks
```

`masks` can be a numpy array, torch tensor, PIL image, or a list of masks. Each
mask should resolve to a 2D `H x W` array.
