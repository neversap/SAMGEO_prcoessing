"""Input data preprocessing module for prediction tasks."""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import math
import logging
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)


class InputPreprocessor:
    """Input data preprocessor for prediction tasks."""

    def __init__(self, sample_path: str, output_base_path: str, job_id: int, user_id: str):
        """
        Initialize the preprocessor.

        Args:
            sample_path: Path to the input sample data
            output_base_path: Base path for output directories
            job_id: Prediction job ID
            user_id: User ID
        """
        self.sample_path = sample_path
        self.output_base_path = output_base_path
        self.job_id = job_id
        self.user_id = user_id

        # Create output directories
        self.result_dir = os.path.join(output_base_path, "result")
        self.resulttif_dir = os.path.join(output_base_path, "resulttif")
        self.resultweng_dir = os.path.join(output_base_path, "resultweng")

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.resulttif_dir, exist_ok=True)
        os.makedirs(self.resultweng_dir, exist_ok=True)

    def detect_sample_type(self) -> str:
        """
        Detect the type of sample data.

        Returns:
            'tif' if TIF files are found, 'image' otherwise
        """
        tif_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

        # Check for TIF files
        for ext in tif_extensions:
            if list(Path(self.sample_path).rglob(f'*{ext}')):
                return 'tif'

        # Check for regular image files
        for ext in image_extensions:
            if list(Path(self.sample_path).rglob(f'*{ext}')):
                return 'image'

        return 'unknown'

    def preprocess_regular_images(self) -> Dict[str, str]:
        """
        Preprocess regular image files.

        Returns:
            Dictionary containing preprocessing information
        """
        logger.info("Preprocessing regular images...")

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(self.sample_path).rglob(f'*{ext}'))

        logger.info(f"Found {len(image_files)} regular images")

        # Copy images to result directory
        processed_images = []
        for img_path in image_files:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to read image: {img_path}")
                    continue

                # Save to result directory
                output_filename = f"{img_path.stem}.jpg"
                output_path = os.path.join(self.result_dir, output_filename)
                cv2.imwrite(output_path, img)

                processed_images.append({
                    'original_path': str(img_path),
                    'processed_path': output_path,
                    'filename': output_filename
                })
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")

        # Save preprocessing metadata
        metadata = {
            'sample_type': 'image',
            'total_images': len(processed_images),
            'images': processed_images,
            'output_directory': self.result_dir
        }

        metadata_path = os.path.join(self.output_base_path, 'preprocessing_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Preprocessed {len(processed_images)} images to {self.result_dir}")

        return {
            'input_directory': self.result_dir,
            'output_directory': self.result_dir,
            'metadata_path': metadata_path,
            'sample_type': 'image'
        }

    def preprocess_tif_files(self, tile_size: int = 512, overlap: int = 64) -> Dict[str, str]:
        """
        Preprocess TIF files by tiling them.

        Args:
            tile_size: Size of each tile
            overlap: Overlap between tiles

        Returns:
            Dictionary containing preprocessing information
        """
        logger.info("Preprocessing TIF files with tiling...")

        tif_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        tif_files = []

        for ext in tif_extensions:
            tif_files.extend(Path(self.sample_path).rglob(f'*{ext}'))

        logger.info(f"Found {len(tif_files)} TIF files")

        all_tiles_info = []

        for tif_idx, tif_path in enumerate(tif_files):
            logger.info(f"Processing TIF file {tif_idx + 1}/{len(tif_files)}: {tif_path}")

            # Tile TIF file, all tiles go to the same directory
            tiles_info = self._tile_tif_file(tif_path, self.resulttif_dir, tile_size, overlap)

            # Save tile information for this TIF
            tif_metadata = {
                'original_path': str(tif_path),
                'filename': tif_path.stem,
                'total_tiles': len(tiles_info),
                'tiles': tiles_info,
                'tile_size': tile_size,
                'overlap': overlap
            }

            all_tiles_info.append(tif_metadata)

        # Save overall preprocessing metadata
        metadata = {
            'sample_type': 'tif',
            'total_tif_files': len(tif_files),
            'tile_size': tile_size,
            'overlap': overlap,
            'tif_files': all_tiles_info,
            'output_directory': self.resulttif_dir
        }

        metadata_path = os.path.join(self.output_base_path, 'preprocessing_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Preprocessed {len(tif_files)} TIF files to {self.resulttif_dir}")

        return {
            'input_directory': self.resulttif_dir,
            'output_directory': self.resulttif_dir,
            'metadata_path': metadata_path,
            'sample_type': 'tif',
            'total_tif_files': len(tif_files)
        }

    def _tile_tif_file(self, tif_path: Path, output_dir: str, tile_size: int, overlap: int) -> List[Dict]:
        """
        Tile a single TIF file into smaller pieces.

        Args:
            tif_path: Path to the TIF file
            output_dir: Directory to save tiles
            tile_size: Size of each tile
            overlap: Overlap between tiles

        Returns:
            List of tile information dictionaries
        """
        # Read TIF file
        img = cv2.imread(str(tif_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            # Try PIL if OpenCV fails
            pil_img = Image.open(tif_path)
            img = np.array(pil_img)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        height, width = img.shape[:2]

        # Read TIF geospatial information
        geospatial_info = None
        try:
            with rasterio.open(str(tif_path)) as src:
                # Get geospatial information
                transform = src.transform
                crs = src.crs
                bounds = src.bounds  # Projected coordinates
                
                # Convert bounds to geographic coordinates (WGS84) if needed
                if crs and str(crs) != 'EPSG:4326':
                    try:
                        from rasterio.warp import transform_bounds
                        from rasterio.crs import CRS
                        
                        src_crs = CRS.from_string(str(crs))
                        dst_crs = CRS.from_epsg(4326)  # WGS84
                        
                        # Transform bounds to WGS84
                        geo_bounds = transform_bounds(src_crs, dst_crs, bounds)
                        
                        logger.info(f"TIF bounds (projected): {bounds}")
                        logger.info(f"TIF bounds (geographic): {geo_bounds}")
                        
                        # Use geographic bounds
                        bounds = geo_bounds
                    except Exception as e:
                        logger.warning(f"Failed to transform bounds to WGS84: {e}")
                        # Use projected bounds as fallback
                else:
                    logger.info(f"TIF bounds (already WGS84 or no CRS): {bounds}")
                
                geospatial_info = {
                    'transform': [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
                    'crs': str(crs) if crs else None,
                    'bounds': [bounds.left, bounds.bottom, bounds.right, bounds.top],  # Geographic or projected bounds
                    'width': width,
                    'height': height
                }
                
                logger.info(f"TIF geospatial info: bounds={bounds}, crs={crs}")
        except Exception as e:
            logger.warning(f"Failed to read geospatial info from TIF: {e}")
            geospatial_info = None

        # Calculate number of tiles
        cols = math.ceil(width / (tile_size - 2 * overlap))
        rows = math.ceil(height / (tile_size - 2 * overlap))

        tiles_info = []

        for row in range(rows):
            for col in range(cols):
                # Calculate window boundaries
                x_start = max(0, col * (tile_size - 2 * overlap))
                y_start = max(0, row * (tile_size - 2 * overlap))

                x_end = min(width, x_start + tile_size)
                y_end = min(height, y_start + tile_size)

                # Ensure window is not too small
                if x_end - x_start < tile_size and x_end - x_start < 100:
                    x_start = max(0, x_end - tile_size)
                if y_end - y_start < tile_size and y_end - y_start < 100:
                    y_start = max(0, y_end - tile_size)

                x_end = min(width, x_start + tile_size)
                y_end = min(height, y_start + tile_size)

                # Extract tile
                tile = img[y_start:y_end, x_start:x_end]

                # Save tile with original filename + row + col
                tile_filename = f"{tif_path.stem}_{row:04d}_{col:04d}.jpg"
                tile_path = os.path.join(output_dir, tile_filename)
                cv2.imwrite(tile_path, tile)

                # Calculate geospatial coordinates for this tile if available
                tile_geospatial = None
                if geospatial_info:
                    try:
                        # Get transform from the stored parameters
                        from rasterio.transform import Affine
                        transform_params = geospatial_info['transform']
                        
                        # Affine需要6个参数：a, b, c, d, e, f
                        if len(transform_params) == 6:
                            # 使用6个参数创建Affine变换矩阵
                            transform = Affine(
                                transform_params[0],  # a
                                transform_params[1],  # b
                                transform_params[2],  # c
                                transform_params[3],  # d
                                transform_params[4],  # e
                                transform_params[5]   # f
                            )
                        else:
                            logger.warning(f"Invalid transform parameters length: {len(transform_params)}")
                            continue
                        
                        # Get four corners of the tile in geographic coordinates
                        # Top-left (x_start, y_start)
                        lon_tl, lat_tl = transform * (x_start, y_start)
                        # Top-right (x_end, y_start)
                        lon_tr, lat_tr = transform * (x_end, y_start)
                        # Bottom-right (x_end, y_end)
                        lon_br, lat_br = transform * (x_end, y_end)
                        # Bottom-left (x_start, y_end)
                        lon_bl, lat_bl = transform * (x_start, y_end)
                        
                        # Create GeoJSON polygon for this tile boundary
                        tile_geojson = {
                            "type": "Feature",
                            "properties": {
                                "tile_filename": tile_filename,
                                "coords": [x_start, y_start, x_end, y_end],
                                "index": [row, col],
                                "size": [x_end - x_start, y_end - y_start]
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [lon_tl, lat_tl],  # Top-left
                                    [lon_tr, lat_tr],  # Top-right
                                    [lon_br, lat_br],  # Bottom-right
                                    [lon_bl, lat_bl],  # Bottom-left
                                    [lon_tl, lat_tl]   # Close polygon
                                ]]
                            }
                        }
                        
                        tile_geospatial = {
                            'bounds': [min(lon_tl, lon_bl), min(lat_br, lat_bl), max(lon_tr, lon_br), max(lat_tl, lat_tr)],
                            'corners': [
                                [lon_tl, lat_tl],  # Top-left
                                [lon_tr, lat_tr],  # Top-right
                                [lon_br, lat_br],  # Bottom-right
                                [lon_bl, lat_bl]   # Bottom-left
                            ],
                            'crs': geospatial_info['crs'],
                            'transform': geospatial_info['transform'],
                            'geojson': tile_geojson  # ✅ 保存tile的GeoJSON边界
                        }
                    except Exception as e:
                        logger.warning(f"Failed to calculate geospatial coords for tile {tile_filename}: {e}")
                        import traceback
                        logger.warning(f"Traceback: {traceback.format_exc()}")

                tile_info = {
                    'tile_filename': tile_filename,
                    'tile_path': tile_path,
                    'coords': [x_start, y_start, x_end, y_end],
                    'index': [row, col],
                    'size': [x_end - x_start, y_end - y_start],
                    'geospatial': tile_geospatial,
                    'original_filename': tif_path.stem  # ✅ 保存原始TIF文件名
                }

                tiles_info.append(tile_info)

        return tiles_info

    def preprocess(self, tile_size: int = 512, overlap: int = 64) -> Dict[str, str]:
        """
        Main preprocessing method that detects sample type and processes accordingly.

        Args:
            tile_size: Size of tiles for TIF files
            overlap: Overlap between tiles for TIF files

        Returns:
            Dictionary containing preprocessing information
        """
        sample_type = self.detect_sample_type()

        if sample_type == 'tif':
            return self.preprocess_tif_files(tile_size, overlap)
        elif sample_type == 'image':
            return self.preprocess_regular_images()
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")
