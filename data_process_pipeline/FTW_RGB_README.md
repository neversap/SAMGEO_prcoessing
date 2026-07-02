# FTW RGB-only Preprocessing

This module converts official FTW-style samples into an RGB-only pretraining dataset that matches the in-house field boundary format.

## Goal

Official FTW samples are expected to contain stacked Sentinel-2 channels:

```text
B04_t1, B03_t1, B02_t1, B08_t1,
B04_t2, B03_t2, B02_t2, B08_t2
```

The converter writes two RGB samples by default:

```text
t1 = R_t1, G_t1, B_t1
t2 = R_t2, G_t2, B_t2
```

Masks keep the same 3-class semantics used by the in-house preprocessing pipeline:

```text
0   background
1   field interior
2   field boundary
255 ignore
```

## Recommended Server Layout

```text
/home/nvme1/datasets/cropland_pretrain/
  raw/
    ftw/
      official/
        Rwanda/
        France/
        Vietnam/
    inhouse/
      tif/
      shp/
  processed/
    ftw_rgb/
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
    ftw_inspection.csv
    ftw_patch_index.csv
    ftw_band_stats.json
    ftw_rgb_config.json
```

Keep `raw/ftw/official` read-only. The converter only writes under `processed/ftw_rgb` and `metadata`.

## Preferred Usage With Manifest

Use a manifest whenever possible, because the exact official FTW directory layout may change.

```csv
image_path,mask_path,split,country,sample_id
Rwanda/sample_001.tif,Rwanda/sample_001_mask.tif,train,Rwanda,sample_001
Rwanda/sample_002.tif,Rwanda/sample_002_mask.tif,val,Rwanda,sample_002
```

Then run:

```bash
python -m data_process_pipeline.ftw_rgb \
  --ftw-root /home/nvme1/datasets/cropland_pretrain/raw/ftw/official \
  --manifest /home/nvme1/datasets/cropland_pretrain/metadata/ftw_manifest.csv \
  --output-dir /home/nvme1/datasets/cropland_pretrain/processed/ftw_rgb \
  --metadata-dir /home/nvme1/datasets/cropland_pretrain/metadata
```

If `split` is empty or missing, the converter assigns `train/val/test` using the configured ratios.

## Quick Smoke Test

Use a small subset first:

```bash
python -m data_process_pipeline.ftw_rgb \
  --ftw-root /home/nvme1/datasets/cropland_pretrain/raw/ftw/official \
  --manifest /home/nvme1/datasets/cropland_pretrain/metadata/ftw_manifest.csv \
  --output-dir /home/nvme1/datasets/cropland_pretrain/processed/ftw_rgb \
  --metadata-dir /home/nvme1/datasets/cropland_pretrain/metadata \
  --max-samples 20
```

## Web UI

The FastAPI server exposes FTW jobs on the `/preprocess` page.

## Docker Dependency

The server image installs the official FTW CLI from a local zip to avoid GitHub access during Docker build. Put the downloaded FTW repository archive in the project root as:

```text
ftw-baselines-main.zip
```

`requirements-ftw.txt` installs `/tmp/ftw-baselines-main.zip` after the Dockerfile copies it into the image.

Backend routes:

```text
POST /ftw/download
POST /ftw/preprocess
GET  /ftw/jobs/{job_id}
POST /ftw/jobs/{job_id}/cancel
```

`/ftw/download` runs the official CLI command:

```text
ftw data download --countries=<countries>
```

The server sets `FTW_DATA_DIR` and `FTW_DATA_ROOT` to the configured FTW root. If the official CLI in your environment requires additional flags, put them in the UI's `Extra args` field.

`/ftw/preprocess` calls `data_process_pipeline.ftw_rgb` and writes RGB-only samples plus metadata.

## Auto Discovery Mode

Without a manifest, the converter tries to discover:

- `.npz` samples containing image keys such as `image`, `x`, or `arr_0`, and mask keys such as `mask`, `y`, or `label`.
- 8-band GeoTIFF images with nearby mask files named like `_mask`, `_label`, or `_labels`.

This is useful for experiments, but manifest mode is better for repeatable training.

## Output Metadata

`ftw_inspection.csv` records sample shape, dtype, split, country, and mask values.

`ftw_patch_index.csv` records:

- `patch_id`
- `source_dataset`
- `country`
- `source_sample`
- `window_id`
- `split`
- `interior_ratio`
- `boundary_ratio`
- `background_ratio`
- `ignore_ratio`
- `image_path`
- `mask_path`

`ftw_band_stats.json` stores RGB band statistics over valid pixels.

## Integration With In-house Data

After conversion, FTW RGB and in-house patches share the same training contract:

```text
image: 3-band GeoTIFF
mask:  uint8 GeoTIFF
classes: 0, 1, 2, 255
metadata: patch index CSV
```

This lets the future training module load FTW pretraining data and in-house fine-tuning data through a common dataset wrapper.
