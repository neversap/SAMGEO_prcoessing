# SAM Mask Postprocess Module

This document describes the current SAM mask postprocessing flow and how to move it into a production project as a plug-and-play module.

## Purpose

The postprocess module receives raw boolean masks from SAM/SAM3 and converts them into cleaner, more regular outputs for visualization and downstream use.

It is intentionally model-agnostic. Any model that outputs masks shaped as `H x W` arrays can use the same module.

```text
model inference
  -> raw masks
  -> mask postprocess
  -> visualization / bbox / area / API response
```

## Current Entry Point

Current implementation:

```text
server/app/utils/images.py
```

Main function:

```python
postprocess_masks(
    masks,
    mode="polygon",
    epsilon_ratio=0.003,
    min_area=64,
    max_area=None,
    fill_holes=True,
    max_hole_area=256,
    connectivity=8,
    quad_mode="axis",
)
```

Input:

- `masks`: list of numpy arrays. Each mask should be convertible to boolean.

Output:

- list of boolean numpy masks after postprocessing.

## Processing Flow

All enabled postprocess modes first run a shared cleaning step:

```text
raw mask
  -> connected component cleanup
  -> optional hole filling
  -> selected shape postprocess
```

### 1. Component Cleanup

Function:

```python
clean_mask_components()
```

Behavior:

- Splits masks into connected components.
- Removes components smaller than `min_area`.
- Optionally removes components larger than `max_area`.
- Optionally fills small internal holes.
- Supports 4-connectivity or 8-connectivity.

This is the base cleanup layer for `clean`, `polygon`, `orthogonal`, and `quad` modes.

### 2. Clean Mode

Mode:

```python
mode="clean"
```

Behavior:

- Only applies component cleanup and hole filling.
- Keeps the original mask shape.
- Useful as a baseline mode.

### 3. Polygon Mode

Mode:

```python
mode="polygon"
```

Function:

```python
polygonize_mask()
```

Behavior:

- Finds external contours with OpenCV.
- Simplifies each contour using `cv2.approxPolyDP`.
- Fills the simplified polygon back into a mask.

Use this when the output should preserve the original object outline while reducing pixel-level jaggedness.

### 4. Quad Mode

Mode:

```python
mode="quad"
```

Supported `quad_mode` values:

- `axis`
- `rotated`

#### Axis Quad

Function:

```python
quadify_component()
```

Behavior:

- Finds a main body region.
- Converts the body into an axis-aligned box.
- Adds connected protrusion boxes when they are close enough to the main body.

This mode is simple, but it cannot follow rotated buildings well.

#### Rotated Quad

Function:

```python
quadify_component_rotated()
```

Behavior:

```text
component mask
  -> estimate dominant angle from minAreaRect
  -> rotate component into local coordinates
  -> recursively decompose local mask into rectangles
  -> rotate rectangles back into image coordinates
  -> fill missed foreground regions with fixed-angle rectangles
  -> close small tears
```

Important helper functions:

- `min_area_rect_polygon()`
- `polygon_long_edge_angle()`
- `component_to_local_mask()`
- `decompose_local_rectangles()`
- `cover_uncovered_regions()`
- `close_output_mask()`

This mode is the current preferred regularization mode for building-like objects because it can represent one mask as multiple same-direction rectangles.

## Rendering Helpers

The current project also includes visualization helpers:

```python
mask_to_png_base64()
semantic_mask_to_png_base64()
instance_masks_to_png_base64()
bbox_from_mask()
```

These are useful for the demo API and frontend, but they should be optional in a production package.

Recommended split:

```text
postprocess module
  -> mask geometry only

render module
  -> PNG/base64/overlay only
```

## Recommended Production Package Layout

For production reuse, extract the logic into a small package:

```text
sam_mask_postprocess/
  __init__.py
  config.py
  masks.py
  render.py
```

Optional:

```text
sam_mask_postprocess/
  proposals.py
```

Use `proposals.py` only if the production project also needs OpenCV-generated box/point proposals.

## Recommended Public API

Expose one stable function for common usage:

```python
from sam_mask_postprocess import postprocess_sam_masks

result = postprocess_sam_masks(
    masks=raw_masks,
    mode="quad",
    quad_mode="rotated",
    min_area=64,
    fill_holes=True,
)
```

Recommended return type:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class PostprocessResult:
    masks: list[np.ndarray]
    bboxes: list[list[int]]
    areas: list[int]
    area_ratios: list[float]
```

This keeps the production integration simple and avoids coupling the module to FastAPI, SAM3, or frontend-specific base64 images.

## Example Integration

```python
raw_masks = sam_model.segment(image, prompt="building")

processed = postprocess_sam_masks(
    masks=raw_masks,
    mode="quad",
    quad_mode="rotated",
    min_area=64,
    max_area=None,
    fill_holes=True,
    max_hole_area=256,
)

return {
    "masks": processed.masks,
    "bboxes": processed.bboxes,
    "areas": processed.areas,
    "area_ratios": processed.area_ratios,
}
```

## Dependencies

Required runtime dependencies:

- `numpy`
- `opencv-python-headless`

Optional rendering dependencies:

- `Pillow`

If the production service only needs processed masks and bounding boxes, `Pillow` is not required.

## Configuration Mapping

Current environment/settings fields that map to the postprocess module:

```text
SAM_GEO_MASK_POSTPROCESS      -> mode
SAM_GEO_QUAD_MODE             -> quad_mode
SAM_GEO_POLYGON_EPSILON_RATIO -> epsilon_ratio
SAM_GEO_MIN_MASK_AREA         -> min_area
SAM_GEO_MAX_MASK_AREA         -> max_area
SAM_GEO_FILL_MASK_HOLES       -> fill_holes
SAM_GEO_MAX_HOLE_AREA         -> max_hole_area
SAM_GEO_COMPONENT_CONNECTIVITY -> connectivity
```

Recommended defaults:

```python
mode = "polygon"
quad_mode = "rotated"
epsilon_ratio = 0.003
min_area = 64
max_area = None
fill_holes = True
max_hole_area = 256
connectivity = 8
```

## Migration Notes

When copying this into a production project:

1. Keep the postprocess module independent from the SAM3 adapter.
2. Accept only raw masks and plain config values as inputs.
3. Return masks and geometric metadata, not API response objects.
4. Keep visualization helpers optional.
5. Add regression samples for `polygon` and `quad_rotated` modes before tuning thresholds.

## Known Limitations

- `polygon` mode preserves irregular boundaries and does not enforce right angles.
- `quad_rotated` mode regularizes building-like shapes but may over-regularize shadows or ambiguous masks.
- Very fragmented masks should be cleaned before quad conversion.
- The current quad decomposition is raster-based and may need domain-specific tuning for other imagery sources.
