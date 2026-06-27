# Data Process Pipeline

This module builds training data for remote-sensing cropland segmentation.

## Input Layout

Preferred layout:

```text
dataset/
  raw/
    images/
      image_1.tif
      image_2.tif
    labels/
      farmland.shp
      farmland.shx
      farmland.dbf
      farmland.prj
      farmland.cpg
```

Legacy layout is also supported:

```text
dataset/
  tif/
    image_1.tif
    image_2.tif
  shp/
    farmland.shp
    farmland.shx
    farmland.dbf
    farmland.prj
    farmland.cpg
```

## Output Layout

```text
dataset/
  processed/
    masks/
      image_1_mask.tif
      image_2_mask.tif
    patches/
      train/
        images/
        masks/
      val/
        images/
        masks/
      test/
        images/
        masks/
  metadata/
    image_info.csv
    label_info.csv
    overlap_report.csv
    patch_index.csv
    band_stats.json
    pipeline_config.json
```

## Pipeline

1. Load all TIF files from `raw/images` or `tif`.
2. Load and merge all SHP files from `raw/labels` or `shp`.
3. Reproject labels to each image CRS when needed.
4. Rasterize cropland polygons into full-size mask TIF files.
5. Cut image and mask patches with `tile_size` and `overlap`.
6. Save patch metadata, including source image, pixel position, cropland ratio, ignore ratio, and split.
7. Save image-level metadata and per-band statistics.

Two mask modes are supported:

- `binary`: legacy mode, where `0 = background` and `1 = cropland`.
- `field_boundary_3class`: PRUE-style mode, where `0 = background`, `1 = farmland interior`, `2 = farmland boundary`, and `255 = ignore`.

## Usage

```powershell
python -m data_process_pipeline.pipeline --dataset-dir dataset --tile-size 512 --overlap 64
```

Useful options:

```powershell
python -m data_process_pipeline.pipeline `
  --dataset-dir dataset `
  --tile-size 512 `
  --overlap 64 `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1 `
  --split-strategy image `
  --mask-mode field_boundary_3class `
  --boundary-width-pixels 2
```

`--split-strategy image` keeps patches from the same source TIF in the same split. This is the safer default for remote-sensing validation because neighboring patches can be highly correlated.

Use `--split-strategy patch` only for quick experiments where strict spatial separation is not important.

Use `--drop-empty` if you only want patches containing cropland pixels. The default keeps empty patches so the model can learn negative/background regions.

Use `--test-process` for a fast smoke test. It keeps the normal output layout, but only processes the first TIF found under `raw/images` or `tif`.

In `field_boundary_3class` mode, `--boundary-width-pixels` controls how wide the rasterized field boundary class is. `--drop-empty` keeps boundary and interior patches, drops high-ignore patches, and samples background patches according to `--background-keep-ratio`.

`--max-ignore-ratio` skips patches dominated by nodata or black border pixels. `--black-pixel-threshold` treats pixels whose bands are all less than or equal to the threshold as invalid.

## Metadata

`patch_index.csv` contains:

- `patch_name`
- `source_tif`
- `x`
- `y`
- `width`
- `height`
- `cropland_ratio`
- `ignore_ratio`
- `interior_ratio`
- `boundary_ratio`
- `background_ratio`
- `patch_type`
- `split`
- `image_path`
- `mask_path`

In binary mode, `cropland_ratio` is the ratio of mask pixels equal to `1`. In `field_boundary_3class` mode, it is `interior_ratio + boundary_ratio`.

`ignore_ratio` is calculated from image nodata pixels and all-band black pixels.

## Web Job API

The FastAPI server exposes preprocessing as a background job:

```text
POST /preprocess/jobs
GET  /preprocess/jobs/{job_id}
GET  /preprocess/jobs
POST /preprocess/jobs/{job_id}/cancel
```

The frontend stores the active `job_id` in browser local storage and polls the status endpoint. Closing or refreshing the browser does not stop the backend job.

Job state files are stored under `SAM_GEO_DATA_PROCESS_JOBS_DIR`, which defaults to `runtime/preprocess_jobs`.

To restrict user-entered absolute dataset paths, set:

```text
SAM_GEO_DATA_PROCESS_ALLOWED_ROOTS=/home/nvme1/rx,/data
```
