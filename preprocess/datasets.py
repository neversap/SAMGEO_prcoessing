import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import random
from sklearn.model_selection import train_test_split

import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.features import rasterize
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box, Polygon
from PIL import Image


def get_tif_bounds(tif_path: str) -> Tuple[float, float, float, float]:
    """获取tif文件的边界框 (left, bottom, right, top)"""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
    return bounds


def get_shp_in_bounds(shp_path: str, bounds: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """获取与边界框相交的shp内容"""
    gdf = gpd.read_file(shp_path)
    
    # 创建边界框多边形
    left, bottom, right, top = bounds
    bbox = box(left, bottom, right, top)
    
    # 筛选与边界框相交的几何体
    gdf = gdf[gdf.intersects(bbox)]
    
    return gdf


def create_mask_from_shp(
    shp_gdf: gpd.GeoDataFrame,
    transform: Affine,
    height: int,
    width: int,
    bounds: Tuple[float, float, float, float]
) -> np.ndarray:
    """根据shp内容创建mask"""
    if len(shp_gdf) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # 确保shp与tif坐标系一致
    shapes = [(geom, 1) for geom in shp_gdf.geometry if geom is not None]
    
    if len(shapes) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    mask_array = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )
    
    return mask_array


def tile_tif_with_shp(
    tif_path: str,
    shp_gdf: gpd.GeoDataFrame,
    tile_size: int = 256,
    output_dir: str = "./datasets/all"
) -> List[Tuple[str, str]]:
    """
    将tif裁剪成块，只保留包含shp内容的块
    
    Returns:
        List[Tuple[str, str]]: 生成的(image_path, mask_path)列表
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    generated_pairs = []
    
    with rasterio.open(tif_path) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
        
        # 计算tile数量
        tile_idx = 0
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # 计算实际tile尺寸（边界处理）
                actual_width = min(tile_size, width - x)
                actual_height = min(tile_size, height - y)
                
                # 如果tile太小，跳过
                if actual_width < tile_size or actual_height < tile_size:
                    continue
                
                # 读取tile数据
                window = Window(x, y, actual_width, actual_height)
                tile_data = src.read(window=window)
                
                # 计算tile的地理变换矩阵
                tile_transform = src.window_transform(window)
                
                # 计算tile的边界
                tile_bounds = rasterio.windows.bounds(window, src.transform)
                
                # 获取tile范围内的shp内容
                tile_shp_gdf = shp_gdf[shp_gdf.intersects(box(*tile_bounds))]
                
                # 创建mask
                mask_data = create_mask_from_shp(
                    tile_shp_gdf,
                    tile_transform,
                    actual_height,
                    actual_width,
                    tile_bounds
                )
                
                # 如果mask中有内容，保存tile
                if np.sum(mask_data) > 0:
                    tile_id = f"{Path(tif_path).stem}_tile_{tile_idx:05d}"
                    
                    # 保存image (取前3个波段作为RGB)
                    image_path = images_dir / f"{tile_id}.png"
                    if tile_data.shape[0] >= 3:
                        # 归一化到0-255
                        rgb_data = tile_data[:3].transpose(1, 2, 0)
                        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
                    else:
                        # 单波段复制成3通道
                        rgb_data = np.stack([tile_data[0]] * 3, axis=-1)
                        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
                    
                    Image.fromarray(rgb_data).save(image_path)
                    
                    # 保存mask
                    mask_path = masks_dir / f"{tile_id}.png"
                    Image.fromarray(mask_data * 255).save(mask_path)
                    
                    generated_pairs.append((str(image_path), str(mask_path)))
                    tile_idx += 1
    
    return generated_pairs


def split_train_val(
    all_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """将all文件夹中的数据划分为train和val"""
    all_dir = Path(all_dir)
    output_dir = Path(output_dir)
    
    images_dir = all_dir / "images"
    masks_dir = all_dir / "masks"
    
    # 获取所有image文件
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.tif']])
    
    # 划分train和val
    train_files, val_files = train_test_split(
        image_files,
        train_size=train_ratio,
        random_state=seed
    )
    
    # 创建输出目录
    splits = {
        'train': train_files,
        'val': val_files
    }
    
    for split_name, files in splits.items():
        split_images_dir = output_dir / split_name / "images"
        split_masks_dir = output_dir / split_name / "masks"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_masks_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in files:
            # 对应的mask文件
            mask_file = masks_dir / img_file.name
            
            if mask_file.exists():
                # 复制到对应目录
                shutil.copy2(img_file, split_images_dir / img_file.name)
                shutil.copy2(mask_file, split_masks_dir / mask_file.name)
        
        print(f"{split_name}: {len(files)} samples")


def main(
    datasets_dir: str = "./datasets",
    tile_size: int = 256,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    主函数：完成样本制作
    
    Args:
        datasets_dir: 数据集根目录
        tile_size: 裁剪块大小
        train_ratio: 训练集比例
        seed: 随机种子
    """
    datasets_dir = Path(datasets_dir)
    tif_dir = datasets_dir / "tif"
    shp_dir = datasets_dir / "shp"
    all_dir = datasets_dir / "all"
    
    # 检查目录是否存在
    if not tif_dir.exists():
        raise FileNotFoundError(f"TIF directory not found: {tif_dir}")
    if not shp_dir.exists():
        raise FileNotFoundError(f"SHP directory not found: {shp_dir}")
    
    # 获取所有tif文件
    tif_files = list(tif_dir.glob("*.tif"))
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No TIF files found in {tif_dir}")
    
    print(f"Found {len(tif_files)} TIF files")
    
    # 获取所有shp文件
    shp_files = list(shp_dir.glob("*.shp"))
    if len(shp_files) == 0:
        raise FileNotFoundError(f"No SHP files found in {shp_dir}")
    
    print(f"Found {len(shp_files)} SHP files")
    
    # 合并所有shp文件
    all_shp_gdfs = []
    for shp_file in shp_files:
        gdf = gpd.read_file(shp_file)
        all_shp_gdfs.append(gdf)
    
    if len(all_shp_gdfs) > 0:
        combined_shp_gdf = gpd.GeoDataFrame(pd.concat(all_shp_gdfs, ignore_index=True))
    else:
        combined_shp_gdf = gpd.GeoDataFrame()
    
    print(f"Combined SHP has {len(combined_shp_gdf)} features")
    
    # 处理每个tif文件
    all_generated_pairs = []
    for tif_file in tif_files:
        print(f"\nProcessing: {tif_file.name}")
        
        # 获取tif边界
        tif_bounds = get_tif_bounds(str(tif_file))
        print(f"  TIF bounds: {tif_bounds}")
        
        # 获取与tif边界相交的shp内容
        tif_shp_gdf = combined_shp_gdf[combined_shp_gdf.intersects(box(*tif_bounds))]
        print(f"  SHP features in bounds: {len(tif_shp_gdf)}")
        
        if len(tif_shp_gdf) == 0:
            print(f"  Warning: No SHP features found for {tif_file.name}, skipping...")
            continue
        
        # 裁剪tif为tiles
        pairs = tile_tif_with_shp(
            str(tif_file),
            tif_shp_gdf,
            tile_size=tile_size,
            output_dir=str(all_dir)
        )
        all_generated_pairs.extend(pairs)
        print(f"  Generated {len(pairs)} tiles")
    
    print(f"\nTotal generated tiles: {len(all_generated_pairs)}")
    
    if len(all_generated_pairs) == 0:
        print("Warning: No tiles were generated!")
        return
    
    # 划分train和val
    print("\nSplitting into train and val...")
    split_train_val(
        str(all_dir),
        str(datasets_dir),
        train_ratio=train_ratio,
        seed=seed
    )
    
    print("\nDataset preparation completed!")


if __name__ == "__main__":
    import pandas as pd
    
    main(
        datasets_dir="./datasets",
        tile_size=512,
        train_ratio=0.8,
        seed=42
    )
