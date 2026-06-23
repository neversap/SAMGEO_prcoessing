# Put SAM3 inside the Docker image

There are two supported deployment modes.

## Mode A: mount model files at runtime

This is the default and is usually better for development:

```bash
cp .env.example .env
mkdir -p ./models/sam3
docker compose up -d --build
```

The container reads:

```text
/models/sam3
```

from the host path configured by:

```env
SAM_GEO_MODEL_HOST_PATH=./models/sam3
```

## Mode B: bake model files into the image

Use this when the server must run offline or you want an immutable image that
contains both API code and model files.

On the server, put SAM3 files in a directory such as:

```text
/srv/models/sam3
```

Then build with Docker Compose:

```bash
cp .env.example .env
sed -i 's/SAM_GEO_BACKEND=mock/SAM_GEO_BACKEND=sam3/' .env
SAM_GEO_MODEL_HOST_PATH=/srv/models/sam3 docker compose -f docker-compose.baked-model.yml build
docker compose -f docker-compose.baked-model.yml up -d
```

Or build directly with BuildKit:

```bash
docker buildx build \
  --build-context sam3_model=/srv/models/sam3 \
  -f Dockerfile.server.baked-model \
  -t sam-geo-api:sam3 \
  .
```

## SAM3 code dependencies

The Dockerfiles install `sam3-main.zip` directly:

```bash
pip install "/opt/sam3-main[notebooks]"
```

This lets pip resolve dependencies and notebook extras from the official
`pyproject.toml` inside the zip instead of duplicating them in this project. The
SAM3 image API imports `einops`, which is declared in that official extra.

PyTorch is not declared in that `pyproject.toml`, so the Dockerfiles install the
official SAM3 README requirement before installing the zip:

```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

Keep the large model artifacts outside git. The baked-model Dockerfile copies
them from the named build context into:

```text
/models/sam3
```

## Tradeoffs

- Mounted model: fastest to update weights, smaller image, best for iteration.
- Baked model: easiest to move as one artifact, works offline, slower rebuilds
  and much larger image.
