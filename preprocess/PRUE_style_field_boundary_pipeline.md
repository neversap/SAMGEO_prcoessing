# PRUE-style 农田/耕地边界分割落地流程

> 目标：参考 **PRUE: A Practical Recipe for Field Boundary Segmentation at Scale** 的思想，基于已有的大型 GeoTIFF 遥感影像和对应 shp 矢量文件，构建一个可落地的耕地/农田边界分割训练与推理流程。

---

## 0. 当前数据假设

当前已知数据形式：

```text
raw_data/
  images/
    image_01.tif
    image_02.tif
    image_03.tif
    image_04.tif
    image_05.tif
    image_06.tif

  labels/
    farmland.shp
    farmland.shx
    farmland.dbf
    farmland.prj
    farmland.cpg
```

其中：

- 每个 `.tif` 文件约 3GB；
- `.shp/.shx/.dbf/.prj/.cpg` 是一套完整的矢量文件；
- 尚不确定 `.shp` 是耕地/地块 polygon，还是仅为研究区边界；
- 尚不确定 6 个 `.tif` 是不同区域、不同时间，还是不同波段/产品。

因此，第一步不是训练模型，而是**数据检查与任务确认**。

---

## 1. 总体落地流程

整体流程如下：

```text
原始 GeoTIFF + shp
        ↓
数据检查：CRS / 分辨率 / 波段 / bounds / shp 属性
        ↓
判断 shp 是否为耕地/地块标签
        ↓
shp 与 tif 坐标系统一
        ↓
shp 栅格化为 3-class mask
        ↓
大图切 patch
        ↓
patch 过滤与样本统计
        ↓
训练 U-Net + EfficientNet encoder
        ↓
滑窗推理 + overlap 加权拼接
        ↓
输出 raster mask
        ↓
connected components / polygonization
        ↓
输出 field polygons / shapefile / GeoJSON
```

PRUE 的核心启发不是盲目使用大模型，而是：

> 用一个稳定、高效的 U-Net 系列模型，把遥感输入、边界标签、loss、增强、滑窗推理和后处理全部设计正确。

---

## 2. 推荐项目目录结构

建议项目结构如下：

```text
field_boundary_project/
  data/
    raw/
      images/
        image_01.tif
        image_02.tif
        image_03.tif
        image_04.tif
        image_05.tif
        image_06.tif
      labels/
        farmland.shp
        farmland.shx
        farmland.dbf
        farmland.prj
        farmland.cpg

    processed/
      masks/
        image_01_mask.tif
        image_02_mask.tif
        ...

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

  src/
    01_inspect_data.py
    02_rasterize_labels.py
    03_crop_patches.py
    04_compute_stats.py
    05_train.py
    06_infer_large_tif.py
    07_polygonize.py

  configs/
    dataset.yaml
    train_unet_effb3.yaml
    train_unet_effb7.yaml

  outputs/
    checkpoints/
    predictions/
    polygons/
    logs/
```

---

## 3. 阶段 A：数据检查

### 3.1 检查 tif 基本信息

每个 tif 都需要检查：

| 检查项 | 作用 |
|---|---|
| CRS | 判断是否与 shp 坐标一致 |
| width / height | 判断影像尺寸 |
| resolution | 决定 mask 栅格化精度 |
| band count | 判断输入通道数 |
| dtype | 判断归一化方式 |
| nodata | 判断无效区域 |
| bounds | 判断 shp 是否覆盖该 tif |

示例代码：

```python
import rasterio
import glob
import pandas as pd

records = []

for tif_path in glob.glob("data/raw/images/*.tif"):
    with rasterio.open(tif_path) as src:
        records.append({
            "path": tif_path,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": ",".join(src.dtypes),
            "nodata": src.nodata,
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "bounds": str(src.bounds),
        })

info_df = pd.DataFrame(records)
info_df.to_csv("data/metadata/image_info.csv", index=False)
print(info_df)
```

---

### 3.2 检查 shp 属性

重点判断 shp 是否真的是耕地/地块标签。

```python
import geopandas as gpd

shp_path = "data/raw/labels/farmland.shp"
gdf = gpd.read_file(shp_path)

print("CRS:", gdf.crs)
print("Columns:", gdf.columns)
print("Geometry types:")
print(gdf.geometry.type.value_counts())
print("First rows:")
print(gdf.head())
```

需要回答以下问题：

1. `geometry` 是否为 Polygon / MultiPolygon？
2. 属性表中是否有地类字段？例如：
   - `class`
   - `land_type`
   - `type`
   - `crop`
   - `DLMC`
   - `地类名称`
3. 是否能明确筛选出“耕地”或“农田地块”？
4. 如果没有类别字段，是否可以确认所有 polygon 都是耕地/地块？
5. 如果 shp 只是行政区或研究区边界，则它不能直接作为训练标签。

---

## 4. 阶段 B：判断 tif 与 shp 是否重叠

每个 tif 都需要确认是否被 shp 覆盖。

```python
import rasterio
import geopandas as gpd
from shapely.geometry import box
import glob
import pandas as pd

shp_path = "data/raw/labels/farmland.shp"
gdf = gpd.read_file(shp_path)

records = []

for tif_path in glob.glob("data/raw/images/*.tif"):
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        raster_bounds = box(*src.bounds)

    gdf_proj = gdf.to_crs(raster_crs)
    overlap = gdf_proj[gdf_proj.intersects(raster_bounds)]

    records.append({
        "tif_path": tif_path,
        "overlap_polygon_count": len(overlap),
        "has_overlap": len(overlap) > 0,
    })

report = pd.DataFrame(records)
report.to_csv("data/metadata/overlap_report.csv", index=False)
print(report)
```

解释：

- 如果某个 tif 与 shp 完全没有重叠，则不能作为正样本训练；
- 可以作为负样本、无标签数据或推理数据；
- 若所有 tif 都不重叠，则很可能坐标系错误，或 shp 不是对应标签。

---

## 5. 阶段 C：标签设计

参考 PRUE，建议不要只做二分类，而是构建三类 mask：

```text
0   = background / non-field / non-cropland
1   = field interior / cropland interior
2   = field boundary
255 = ignore / nodata / invalid
```

### 5.1 为什么需要 boundary 类？

普通二分类：

```text
0 = 非耕地
1 = 耕地
```

问题是相邻地块容易粘连，模型只知道“这里是耕地”，但不知道“两个地块之间有边界”。

PRUE-style 三分类：

```text
0 = 背景
1 = 地块内部
2 = 地块边界
```

优势：

- 可以显式学习田块边界；
- 后续可以用 boundary 类分隔相邻地块；
- 更适合转成独立 polygon；
- 更接近 field boundary segmentation 任务。

---

## 6. 阶段 D：从 shp 生成三类 mask

### 6.1 标签生成逻辑

对于每个 polygon：

```text
polygon 内部腐蚀后区域 → interior class = 1
polygon 边界 buffer 区域 → boundary class = 2
polygon 外部 → background class = 0
无效影像区域 → ignore class = 255
```

对于 10m 分辨率 Sentinel-2：

```text
boundary_width = 1 pixel  ≈ 10m
boundary_width = 2 pixels ≈ 20m
```

一般建议先用：

```text
boundary_width = 1~2 pixels
```

如果地块很小，boundary width 不宜太大，否则会吞掉地块内部。

---

### 6.2 简化版 rasterize 代码

下面代码适合作为第一版实现。它直接将 polygon 栅格化为 interior，并用边界线 buffer 生成 boundary。

```python
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from pathlib import Path


def generate_three_class_mask(
    tif_path,
    shp_path,
    out_mask_path,
    boundary_width_pixels=2,
    cropland_filter=None,
):
    """
    Generate 3-class mask:
    0   background
    1   field / cropland interior
    2   boundary
    255 ignore
    """

    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        transform = src.transform
        out_shape = (src.height, src.width)
        profile = src.profile.copy()
        pixel_size = abs(src.res[0])
        nodata = src.nodata

        # Read a small validity proxy if needed
        # For very large files, avoid reading all bands at once here.

    gdf = gpd.read_file(shp_path).to_crs(raster_crs)

    # Optional: filter cropland polygons by attribute
    if cropland_filter is not None:
        field_name, valid_values = cropland_filter
        gdf = gdf[gdf[field_name].isin(valid_values)]

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.is_valid]

    # Rasterize full polygon as interior first
    interior_shapes = [(geom, 1) for geom in gdf.geometry]
    interior = rasterize(
        shapes=interior_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    # Generate boundary by buffering polygon boundaries
    boundary_width_map_units = boundary_width_pixels * pixel_size
    boundary_geoms = gdf.geometry.boundary.buffer(boundary_width_map_units)
    boundary_shapes = [(geom, 2) for geom in boundary_geoms if geom is not None]

    boundary = rasterize(
        shapes=boundary_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    mask = interior.copy()
    mask[boundary == 2] = 2

    profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="lzw",
    )

    Path(out_mask_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_mask_path, "w", **profile) as dst:
        dst.write(mask, 1)


# Example
# generate_three_class_mask(
#     tif_path="data/raw/images/image_01.tif",
#     shp_path="data/raw/labels/farmland.shp",
#     out_mask_path="data/processed/masks/image_01_mask.tif",
#     boundary_width_pixels=2,
# )
```

---

## 7. 阶段 E：影像通道设计

PRUE 使用的是双时相 RGBN：

```text
planting season: R, G, B, NIR
harvest season:  R, G, B, NIR
```

最终输入：

```text
8 channels = 4 channels × 2 temporal windows
```

对于当前 6 个 tif，我们需要先判断它们属于哪种情况。

---

### 7.1 情况 A：6 个 tif 是不同区域

则第一版建议：

```text
每个 tif 独立作为单时相输入
input shape = [C, H, W]
```

模型：

```text
U-Net + EfficientNet-B3
input_channels = C
output_classes = 3
```

---

### 7.2 情况 B：6 个 tif 是同一区域不同时间

可以参考 PRUE 选择两个关键时相：

```text
T1 = 播种期 / 早季
T2 = 收获期 / 晚季
```

输入：

```text
T1_RGBN + T2_RGBN = 8 channels
```

如果没有明确播种/收获时间，可以先根据 NDVI 或影像日期粗略选：

```text
早季低植被/裸土阶段
晚季成熟/收获阶段
```

---

### 7.3 情况 C：tif 是多波段产品

如果每个 tif 本身包含多个波段，可以使用：

```text
R, G, B, NIR
```

或更完整的：

```text
Blue, Green, Red, NIR, RedEdge, SWIR
```

第一版不建议通道过多，优先跑通：

```text
RGBN 或 RGB
```

---

## 8. 阶段 F：大图切 patch

3GB tif 不能直接整体送入模型，需要切 patch。

建议：

```text
patch_size = 512
stride = 384
即 overlap = 25%
```

对于 10m 分辨率数据：

```text
512 × 512 patch = 5.12 km × 5.12 km
```

---

### 8.1 patch 命名规则

建议包含来源 tif 和窗口坐标：

```text
image_01_x000000_y000000.tif
image_01_x000384_y000000.tif
image_01_x000768_y000000.tif
...
```

对应 mask：

```text
image_01_x000000_y000000_mask.tif
```

---

### 8.2 切 patch 代码

```python
import os
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from pathlib import Path


def crop_patches(
    image_path,
    mask_path,
    out_img_dir,
    out_mask_dir,
    patch_size=512,
    stride=384,
    max_ignore_ratio=0.5,
    keep_empty_ratio=0.2,
):
    Path(out_img_dir).mkdir(parents=True, exist_ok=True)
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)

    patch_records = []
    src_name = Path(image_path).stem

    with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
        width, height = src_img.width, src_img.height

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)

                img_patch = src_img.read(window=window)
                mask_patch = src_mask.read(1, window=window)

                ignore_ratio = float((mask_patch == 255).mean())
                interior_ratio = float((mask_patch == 1).mean())
                boundary_ratio = float((mask_patch == 2).mean())
                background_ratio = float((mask_patch == 0).mean())

                if ignore_ratio > max_ignore_ratio:
                    continue

                # Keep all positive or boundary patches.
                # For empty background patches, keep only a subset.
                is_empty = (interior_ratio == 0 and boundary_ratio == 0)
                if is_empty and np.random.rand() > keep_empty_ratio:
                    continue

                img_profile = src_img.profile.copy()
                img_profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": src_img.window_transform(window),
                    "compress": "lzw",
                })

                mask_profile = src_mask.profile.copy()
                mask_profile.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": src_mask.window_transform(window),
                    "compress": "lzw",
                })

                patch_name = f"{src_name}_x{x:06d}_y{y:06d}.tif"
                img_out = Path(out_img_dir) / patch_name
                mask_out = Path(out_mask_dir) / patch_name.replace(".tif", "_mask.tif")

                with rasterio.open(img_out, "w", **img_profile) as dst:
                    dst.write(img_patch)

                with rasterio.open(mask_out, "w", **mask_profile) as dst:
                    dst.write(mask_patch, 1)

                patch_records.append({
                    "patch_name": patch_name,
                    "source_tif": image_path,
                    "x": x,
                    "y": y,
                    "interior_ratio": interior_ratio,
                    "boundary_ratio": boundary_ratio,
                    "background_ratio": background_ratio,
                    "ignore_ratio": ignore_ratio,
                })

    return pd.DataFrame(patch_records)
```

---

## 9. 阶段 G：patch 过滤策略

因为 boundary 类占比通常很低，不能完全随机采样。

建议保留策略：

| patch 类型 | 策略 |
|---|---|
| boundary_ratio > 0 | 强制保留 |
| interior_ratio > 0.05 | 保留 |
| interior_ratio 很低但有 boundary | 保留 |
| 全背景 patch | 只保留一部分 |
| ignore_ratio > 0.5 | 丢弃 |
| 全 nodata | 丢弃 |

推荐训练集比例：

```text
正样本 / 内部 patch: 40%~60%
边界 patch:          20%~40%
背景 patch:          10%~20%
```

PRUE-style 任务中，边界样本非常重要。若训练集里边界太少，模型会倾向于输出大块 interior，而忽略地块分割线。

---

## 10. 阶段 H：训练/验证/测试划分

不要随机把所有 patch 混合后划分，否则相邻 patch 会同时出现在 train 和 test，导致指标虚高。

### 10.1 如果 6 个 tif 是不同区域

建议：

```text
train: image_01, image_02, image_03, image_04
val:   image_05
test:  image_06
```

这样可以测试跨影像泛化能力。

### 10.2 如果 6 个 tif 是同一区域不同时间

建议：

```text
train: 早期若干时间
val:   中间时间
test:  后期时间
```

或者按空间 block 划分。

### 10.3 如果同一 tif 很大

建议按空间区域划分：

```text
左侧区域 train
中间区域 val
右侧区域 test
```

避免空间泄漏。

---

## 11. 阶段 I：归一化与统计

### 11.1 uint8 RGB 数据

```python
img = img.astype("float32") / 255.0
```

### 11.2 uint16 遥感反射率数据

常见范围为 0~10000：

```python
img = img.astype("float32") / 10000.0
img = np.clip(img, 0, 1)
```

### 11.3 训练集 mean/std

建议按 band 统计训练集 mean/std：

```python
import json
import rasterio
import numpy as np
from pathlib import Path


def compute_band_stats(image_paths, scale=10000.0):
    sums = None
    sq_sums = None
    count = 0

    for path in image_paths:
        with rasterio.open(path) as src:
            img = src.read().astype("float32") / scale
            img = np.clip(img, 0, 1)

        c = img.shape[0]
        if sums is None:
            sums = np.zeros(c, dtype=np.float64)
            sq_sums = np.zeros(c, dtype=np.float64)

        flat = img.reshape(c, -1)
        sums += flat.sum(axis=1)
        sq_sums += (flat ** 2).sum(axis=1)
        count += flat.shape[1]

    mean = sums / count
    std = np.sqrt(sq_sums / count - mean ** 2)
    return mean.tolist(), std.tolist()

# Example:
# mean, std = compute_band_stats(train_image_paths)
# with open("data/metadata/band_stats.json", "w") as f:
#     json.dump({"mean": mean, "std": std}, f, indent=2)
```

---

## 12. 阶段 J：模型设计

### 12.1 第一版模型

建议先使用：

```text
U-Net + EfficientNet-B3 encoder
```

输入：

```text
image: [C, H, W]
```

输出：

```text
mask logits: [3, H, W]
```

类别：

```text
0 = background
1 = interior
2 = boundary
```

loss：

```text
Loss = DiceLoss + WeightedCrossEntropyLoss
```

推荐 class weights：

```text
background: 1.0
interior:   2.0
boundary:   4.0~8.0
```

boundary 权重需要根据实际类别比例调节。

---

### 12.2 第二版 PRUE-style 模型

当第一版跑通后，升级为：

```text
U-Net + EfficientNet-B5/B7 encoder
```

增强：

```text
brightness augmentation
resize augmentation
channel shuffle, 如果是双时相输入
```

loss：

```text
weighted log-cosh Dice loss
```

大图推理：

```text
25% overlap + Gaussian weighted stitching
```

---

## 13. PyTorch Dataset 设计

```python
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class FieldBoundaryDataset(Dataset):
    def __init__(self, image_paths, mask_paths, mean=None, std=None, scale=10000.0, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.mean = mean
        self.std = std
        self.scale = scale
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        with rasterio.open(img_path) as src:
            image = src.read().astype("float32")

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype("int64")

        image = image / self.scale
        image = np.clip(image, 0, 1)

        if self.mean is not None and self.std is not None:
            mean = np.array(self.mean).reshape(-1, 1, 1)
            std = np.array(self.std).reshape(-1, 1, 1)
            image = (image - mean) / (std + 1e-6)

        # Optional augmentations should apply to both image and mask.
        if self.transforms is not None:
            augmented = self.transforms(image=image.transpose(1, 2, 0), mask=mask)
            image = augmented["image"].transpose(2, 0, 1)
            mask = augmented["mask"]

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask
```

---

## 14. Loss 设计

### 14.1 第一版：Weighted CE + Dice

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        valid = target != self.ignore_index

        target_valid = target.clone()
        target_valid[~valid] = 0

        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(target_valid, num_classes=num_classes).permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        target_onehot = target_onehot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice.mean()
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)
```

---

### 14.2 第二版：Log-Cosh Dice Loss

PRUE 提到 log-cosh Dice 对边界优化更平滑。可以实现为：

```python
class LogCoshDiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        dice_loss = self.dice(logits, target)
        return torch.log(torch.cosh(dice_loss))
```

实际使用时可以组合：

```text
Loss = WeightedCE + LogCoshDice
```

---

## 15. 数据增强策略

PRUE 强调增强不是为了“图像好看”，而是为了提高部署鲁棒性。

推荐增强：

| 增强 | 推荐程度 | 说明 |
|---|---|---|
| horizontal flip | 推荐 | 遥感图像无固定方向 |
| vertical flip | 推荐 | 遥感图像无固定方向 |
| 90° rotation | 推荐 | 农田方向多变 |
| brightness jitter | 推荐但幅度小 | 模拟反射率差异 |
| resize scale jitter | 推荐 | 增强尺度鲁棒性 |
| Gaussian noise | 可选 | 模拟传感器噪声 |
| channel shuffle | 仅双时相时使用 | 减少输入顺序敏感 |

不建议过度使用自然图像风格的颜色增强，因为多光谱波段具有物理意义。

---

## 16. 阶段 K：大图推理

大 tif 推理不能一次性读入内存，应使用滑窗。

推荐参数：

```text
patch_size = 512
stride = 384
overlap = 25%
```

### 16.1 推理逻辑

```text
读取一个 window
        ↓
模型输出 softmax 概率图
        ↓
用 Gaussian weight 给 patch 中心更高权重
        ↓
累加到整图 probability canvas
        ↓
累加 weight canvas
        ↓
probability canvas / weight canvas
        ↓
argmax 得到最终 mask
```

这样能减少 patch 边缘产生的网格状断裂。

---

### 16.2 Gaussian weight map

```python
import numpy as np


def gaussian_weight_map(size=512, sigma_scale=0.25):
    ax = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(ax, ax)
    sigma = sigma_scale
    weight = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    weight = weight / weight.max()
    return weight.astype("float32")
```

---

## 17. 阶段 L：后处理与 polygonization

模型输出为三类 raster mask：

```text
0 = background
1 = interior
2 = boundary
```

后处理流程：

```text
1. 取 interior 类作为候选地块区域
2. 使用 boundary 类分隔相邻地块
3. connected components 得到单独实例
4. 去除过小对象
5. raster polygonize
6. geometry repair
7. 输出 shp / GeoJSON / GeoPackage
```

### 17.1 小斑块过滤

需要根据分辨率设置面积阈值。

例如 10m 分辨率：

```text
1 pixel = 100 m²
min_area_pixels = 10
min_area = 1000 m²
```

过小区域通常是噪声。

---

### 17.2 polygonize 示例

```python
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape


def polygonize_mask(mask_tif, out_vector_path, target_class=1):
    records = []

    with rasterio.open(mask_tif) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

        binary = (mask == target_class).astype("uint8")

        for geom, value in shapes(binary, mask=binary == 1, transform=transform):
            if value == 1:
                records.append({"geometry": shape(geom), "class": int(target_class)})

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Repair invalid geometries
    gdf["geometry"] = gdf.geometry.buffer(0)

    gdf.to_file(out_vector_path)
```

---

## 18. 评估指标

不要只看 mIoU。PRUE 的价值在于强调部署指标。

### 18.1 像素级指标

| 指标 | 说明 |
|---|---|
| IoU | 每类交并比 |
| mIoU | 多类别平均 IoU |
| Dice / F1 | 对边界类尤其重要 |
| Precision | 预测边界是否准确 |
| Recall | 是否漏掉边界 |

### 18.2 对象级指标

| 指标 | 说明 |
|---|---|
| Object Precision | 预测 polygon 中有多少是真的 |
| Object Recall | 真实地块有多少被找回 |
| Object F1 | polygon 级综合指标 |
| AP@0.5 | IoU 阈值 0.5 的实例检测质量 |
| AP@0.5:0.95 | 更严格的实例质量 |

### 18.3 部署鲁棒性指标

建议额外评估：

```text
1. patch 平移一致性
2. 输入亮度变化敏感性
3. 输入分辨率变化敏感性
4. overlap 拼接伪影
5. 不同 tif / 不同区域泛化能力
6. 推理速度，单位 km²/s 或 pixels/s
```

---

## 19. 第一版最小可行实验 MVP

建议先不要一口气复现 PRUE 全部细节，而是做一个 MVP：

```text
Step 1: 检查 6 个 tif 和 shp
Step 2: 判断 shp 是否为耕地/地块 polygon
Step 3: 生成 3-class mask
Step 4: 切 512×512 patch
Step 5: 过滤无效 patch，统计类别比例
Step 6: 按 tif 或空间区域划分 train/val/test
Step 7: 训练 U-Net + EfficientNet-B3
Step 8: 使用 Weighted CE + Dice Loss
Step 9: 滑窗推理 + 25% overlap
Step 10: 输出 raster mask
Step 11: connected components + polygonize
```

MVP 成功标准：

```text
1. 可以从 tif + shp 自动生成训练数据
2. 模型可以正常训练并收敛
3. 推理结果没有明显网格伪影
4. 可以输出 raster mask 和 polygon
5. boundary 类不完全塌缩
```

---

## 20. 第二版升级方向

当 MVP 跑通后，再升级：

```text
1. EfficientNet-B3 → EfficientNet-B5/B7
2. Dice + CE → Log-Cosh Dice + boundary weighting
3. 单时相输入 → 双时相 RGBN 输入
4. 普通增强 → brightness + resize + channel shuffle
5. 普通拼接 → Gaussian weighted stitching
6. 像素级评估 → object-level F1 / AP
7. 单区域测试 → 跨 tif / 跨区域泛化测试
```

---

## 21. 关键风险与排查

| 风险 | 表现 | 排查方式 |
|---|---|---|
| shp 不是耕地标签 | mask 大面积错误 | 检查属性表和可视化叠加 |
| CRS 不一致 | mask 与影像错位 | 对比 bounds，统一 CRS |
| 标签边界过宽 | 小田块被吞掉 | 调小 boundary_width_pixels |
| boundary 类太少 | 模型不预测边界 | 增加 boundary patch 采样和 class weight |
| 背景 patch 太多 | 模型全预测背景 | 平衡采样 |
| tif 数值范围不明 | 训练不收敛 | 检查 dtype/min/max，重新归一化 |
| 推理网格伪影 | 拼接处断裂 | 增加 overlap 和 Gaussian weighting |
| 空间泄漏 | 测试指标虚高 | 按 tif 或空间区域划分 |

---

## 22. 最终推荐技术路线

```text
Input:
    Large GeoTIFF images + field/cropland polygons

Preprocess:
    CRS alignment
    rasterization
    3-class mask generation
    patch cropping
    patch filtering

Model:
    U-Net + EfficientNet-B3/B5/B7
    output = background / interior / boundary

Training:
    Weighted CE + Dice
    upgrade to Log-Cosh Dice
    boundary-focused sampling
    geometric + brightness + scale augmentation

Inference:
    sliding window
    25% overlap
    Gaussian weighted stitching

Postprocess:
    boundary-aware connected components
    remove small objects
    polygonization

Evaluation:
    pixel IoU / Dice
    boundary F1
    object F1
    large-tile robustness
```

---

## 23. 一句话总结

PRUE-style 落地方案的核心是：

> 不把农田边界分割当成普通 patch 语义分割，而是把它作为一个完整的 GeoML 工程问题：从矢量标签栅格化、边界类别设计、类别不平衡处理、滑窗拼接，到 polygon 输出与对象级评估，全部围绕“大图可部署”来设计。

