"""Output data postprocessing module for prediction tasks."""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import logging
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform_bounds, transform

logger = logging.getLogger(__name__)


class OutputPostprocessor:
    """Output data postprocessor for prediction tasks."""

    def __init__(self, output_base_path: str, job_id: int, user_id: str, preprocessing_info: Dict = None):
        """
        Initialize the postprocessor.

        Args:
            output_base_path: Base path for output directories
            job_id: Prediction job ID
            user_id: User ID
            preprocessing_info: Preprocessing information from database
        """
        self.output_base_path = output_base_path
        self.job_id = job_id
        self.user_id = user_id
        self.preprocessing_info = preprocessing_info or {}

        # Directory paths
        self.resulttif_dir = os.path.join(output_base_path, "resulttif")
        self.resultweng_dir = os.path.join(output_base_path, "resultweng")

        # Ensure directories exist
        os.makedirs(self.resulttif_dir, exist_ok=True)
        os.makedirs(self.resultweng_dir, exist_ok=True)

    def postprocess_regular_images(self) -> Dict[str, str]:
        """
        Postprocess regular image predictions.

        Returns:
            Dictionary containing postprocessing information
        """
        logger.info("Postprocessing regular image predictions...")

        # For regular images, results are already in the correct format
        # Just copy them to resultweng if needed

        # 预测结果在output_base_path/predictions/下
        predictions_dir = os.path.join(self.output_base_path, "predictions")
        logger.info(f"Looking for predictions in: {predictions_dir}")
        
        if os.path.exists(predictions_dir):
            # Copy predictions to resultweng
            final_output_dir = os.path.join(self.resultweng_dir, "predictions")
            if os.path.exists(final_output_dir):
                shutil.rmtree(final_output_dir)
            shutil.copytree(predictions_dir, final_output_dir)

            logger.info(f"Copied predictions to {final_output_dir}")
        else:
            logger.warning(f"Predictions directory not found: {predictions_dir}")

        return {
            'output_directory': self.resultweng_dir,
            'sample_type': 'image'
        }

    def postprocess_tif_results(self) -> Dict[str, str]:
        """
        Postprocess TIF prediction results by merging labels and converting to GeoJSON.

        Returns:
            Dictionary containing postprocessing information
        """
        logger.info("Postprocessing TIF prediction results...")

        # 从preprocessing_info中获取metadata_path
        metadata_path = self.preprocessing_info.get('metadata_path')
        if not metadata_path:
            # 如果没有，尝试从output_base_path查找
            metadata_path = os.path.join(self.output_base_path, 'preprocessing_metadata.json')
        
        logger.info(f"Looking for preprocessing metadata at: {metadata_path}")
        
        if os.path.exists(metadata_path):
            # 有预处理元数据，按照TIF切分信息处理
            with open(metadata_path, 'r') as f:
                preprocessing_metadata = json.load(f)

            tif_files_info = preprocessing_metadata.get('tif_files', [])
            logger.info(f"Processing {len(tif_files_info)} TIF files with preprocessing metadata")

            all_results = []

            for tif_info in tif_files_info:
                logger.info(f"Processing TIF {tif_info['filename']}")

                # Merge predictions for this TIF
                merged_result = self._merge_tif_predictions(
                    tif_info,
                    self.resulttif_dir,
                    self.resultweng_dir
                )

                all_results.append(merged_result)

            # Save postprocessing metadata
            postprocessing_metadata = {
                'sample_type': 'tif',
                'total_tif_files': len(tif_files_info),
                'tif_files': all_results,
                'output_directory': self.resultweng_dir
            }

            postprocessing_metadata_path = os.path.join(self.output_base_path, 'postprocessing_metadata.json')
            with open(postprocessing_metadata_path, 'w') as f:
                json.dump(postprocessing_metadata, f, indent=2)

            logger.info(f"Postprocessed {len(tif_files_info)} TIF files to {self.resultweng_dir}")

            return {
                'output_directory': self.resultweng_dir,
                'sample_type': 'tif',
                'total_tif_files': len(tif_files_info)
            }
        else:
            # 没有预处理元数据，直接处理predictions目录中的标签文件
            logger.info("No preprocessing metadata found, processing predictions directly")
            
            # 查找labels目录 - 直接在output_base_path下的predictions目录查找
            labels_dir = os.path.join(self.output_base_path, "predictions", "labels")
            logger.info(f"Looking for labels in: {labels_dir}")
            
            if not os.path.exists(labels_dir):
                logger.warning(f"Labels directory not found: {labels_dir}")
                return {
                    'output_directory': self.resultweng_dir,
                    'sample_type': 'tif',
                    'total_tif_files': 0,
                    'message': 'No labels found'
                }
            
            # 获取所有标签文件
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            logger.info(f"Found {len(label_files)} label files")
            
            # 按TIF文件分组（通过文件名前缀）
            tif_groups = {}
            for label_file in label_files:
                # 文件名格式：original_file_0000_0000.txt
                # 提取原始TIF文件名（去掉行号和列号）
                parts = label_file.replace('.txt', '').rsplit('_', 2)
                if len(parts) >= 1:
                    tif_name = parts[0]
                    if tif_name not in tif_groups:
                        tif_groups[tif_name] = []
                    tif_groups[tif_name].append(label_file)
            
            logger.info(f"Grouped labels into {len(tif_groups)} TIF files")
            
            all_results = []
            
            # 为每个TIF文件生成GeoJSON
            for tif_name, tile_labels in tif_groups.items():
                logger.info(f"Processing TIF {tif_name} with {len(tile_labels)} tiles")
                
                # 合并这个TIF文件的所有标签
                geojson_result = self._merge_labels_to_geojson(
                    tile_labels,
                    labels_dir,
                    tif_name
                )
                
                # 保存GeoJSON文件
                geojson_path = os.path.join(self.resultweng_dir, f"{tif_name}.geojson")
                with open(geojson_path, 'w') as f:
                    json.dump(geojson_result, f, indent=2)
                
                logger.info(f"Saved GeoJSON to {geojson_path}")
                
                all_results.append({
                    'filename': tif_name,
                    'geojson_path': geojson_path,
                    'total_tiles': len(tile_labels),
                    'total_detections': len(geojson_result['features'])
                })
            
            # Save postprocessing metadata
            postprocessing_metadata = {
                'sample_type': 'tif',
                'total_tif_files': len(tif_groups),
                'tif_files': all_results,
                'output_directory': self.resultweng_dir
            }

            postprocessing_metadata_path = os.path.join(self.output_base_path, 'postprocessing_metadata.json')
            with open(postprocessing_metadata_path, 'w') as f:
                json.dump(postprocessing_metadata, f, indent=2)

            logger.info(f"Postprocessed {len(tif_groups)} TIF files to {self.resultweng_dir}")

            return {
                'output_directory': self.resultweng_dir,
                'sample_type': 'tif',
                'total_tif_files': len(tif_groups)
            }

    def _merge_labels_to_geojson(self, tile_labels: List[str], labels_dir: str, tif_name: str) -> Dict:
        """
        Merge predicted labels from tiles and convert to GeoJSON format.
        This method is used when preprocessing metadata is not available.

        Args:
            tile_labels: List of label file names
            labels_dir: Directory containing label files
            tif_name: Name of the TIF file

        Returns:
            GeoJSON FeatureCollection
        """
        all_detections = []

        # Try to find the original TIF file to get geospatial information
        tif_geospatial_info = None
        tif_file_path = None
        
        # Search for TIF file in various locations
        search_paths = [
            self.output_base_path,
            os.path.join(self.output_base_path, "resulttif"),
            os.path.join(self.output_base_path, "predictions"),
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                for ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
                    potential_path = os.path.join(search_path, f"{tif_name}{ext}")
                    if os.path.exists(potential_path):
                        tif_file_path = potential_path
                        break
                if tif_file_path:
                    break
        
        # If TIF file found, read geospatial information
        if tif_file_path:
            try:
                with rasterio.open(tif_file_path) as src:
                    transform = src.transform
                    crs = src.crs
                    bounds = src.bounds
                    width = src.width
                    height = src.height
                    
                    tif_geospatial_info = {
                        'transform': [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
                        'crs': str(crs) if crs else None,
                        'bounds': [bounds.left, bounds.bottom, bounds.right, bounds.top],
                        'width': width,
                        'height': height
                    }
                    logger.info(f"Read geospatial info from TIF {tif_name}: bounds={bounds}, crs={crs}")
            except Exception as e:
                logger.warning(f"Failed to read geospatial info from TIF {tif_name}: {e}")
                tif_geospatial_info = None
        
        # Parse tile information from filenames
        tile_info_map = {}
        for label_file in tile_labels:
            # Filename format: original_file_0000_0000.txt
            # Extract row and column numbers
            parts = label_file.replace('.txt', '').rsplit('_', 2)
            if len(parts) == 3:
                try:
                    row = int(parts[1])
                    col = int(parts[2])
                    tile_info_map[label_file] = {'row': row, 'col': col}
                except ValueError:
                    logger.warning(f"Could not parse row/col from filename: {label_file}")
                    tile_info_map[label_file] = {'row': 0, 'col': 0}

        for label_file in tile_labels:
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                continue

            # Read labels
            with open(label_path, 'r') as f:
                labels = f.readlines()

            # Get tile info
            tile_info = tile_info_map.get(label_file, {'row': 0, 'col': 0})
            row = tile_info['row']
            col = tile_info['col']

            # Convert labels to GeoJSON features
            for label in labels:
                label = label.strip()
                if not label:
                    continue

                parts = label.split()
                if len(parts) < 5:
                    continue

                class_id = parts[0]
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # Calculate pixel coordinates from normalized coordinates
                # Assuming tile size is 512 (common default)
                tile_size = 512
                x_center = x_center_norm * tile_size
                y_center = y_center_norm * tile_size
                bbox_width = width_norm * tile_size
                bbox_height = height_norm * tile_size

                # Calculate absolute pixel coordinates
                x1 = x_center - bbox_width / 2
                y1 = y_center - bbox_height / 2
                x2 = x_center + bbox_width / 2
                y2 = y_center + bbox_height / 2

                # Calculate tile position in original image
                tile_x_start = col * tile_size
                tile_y_start = row * tile_size
                tile_x_end = tile_x_start + tile_size
                tile_y_end = tile_y_start + tile_size

                # Convert to absolute pixel coordinates in original image
                abs_x1 = tile_x_start + x1
                abs_y1 = tile_y_start + y1
                abs_x2 = tile_x_start + x2
                abs_y2 = tile_y_start + y2

                # Create GeoJSON feature
                if tif_geospatial_info:
                    # Convert to geographic coordinates
                    try:
                        transform_params = tif_geospatial_info['transform']
                        transform = Affine(
                            transform_params[0],  # a
                            transform_params[1],  # b
                            transform_params[2],  # c
                            transform_params[3],  # d
                            transform_params[4],  # e
                            transform_params[5]   # f
                        )

                        # Get geographic coordinates for the four corners
                        geo_x1, geo_y1 = transform * (abs_x1, abs_y1)
                        geo_x2, geo_y2 = transform * (abs_x2, abs_y1)
                        geo_x3, geo_y3 = transform * (abs_x2, abs_y2)
                        geo_x4, geo_y4 = transform * (abs_x1, abs_y2)

                        feature = {
                            "type": "Feature",
                            "properties": {
                                "class_id": int(class_id),
                                "confidence": 1.0,
                                "pixel_coords": [[abs_x1, abs_y1], [abs_x2, abs_y1], [abs_x2, abs_y2], [abs_x1, abs_y2], [abs_x1, abs_y1]],
                                "crs": tif_geospatial_info['crs']
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [geo_x1, geo_y1],
                                    [geo_x2, geo_y2],
                                    [geo_x3, geo_y3],
                                    [geo_x4, geo_y4],
                                    [geo_x1, geo_y1]
                                ]]
                            }
                        }
                    except Exception as e:
                        logger.warning(f"Failed to convert to geographic coordinates: {e}")
                        # Fallback to pixel coordinates
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "class_id": int(class_id),
                                "confidence": 1.0
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [abs_x1, abs_y1],
                                    [abs_x2, abs_y1],
                                    [abs_x2, abs_y2],
                                    [abs_x1, abs_y2],
                                    [abs_x1, abs_y1]
                                ]]
                            }
                        }
                else:
                    # No geospatial information available, use pixel coordinates
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "class_id": int(class_id),
                            "confidence": 1.0
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [abs_x1, abs_y1],
                                [abs_x2, abs_y1],
                                [abs_x2, abs_y2],
                                [abs_x1, abs_y2],
                                [abs_x1, abs_y1]
                            ]]
                        }
                    }

                all_detections.append(feature)

        # Apply non-maximum suppression to remove duplicate detections
        filtered_detections = self._apply_nms_to_features(all_detections, iou_threshold=0.5)

        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": filtered_detections
        }

        return geojson

    def _merge_tif_predictions(self, tif_info: Dict, tif_output_dir: str, final_output_dir: str) -> Dict:
        """
        Merge prediction results for a single TIF file and convert to GeoJSON.

        Args:
            tif_info: TIF metadata from preprocessing
            tif_output_dir: Directory containing tile predictions
            final_output_dir: Directory to save merged results

        Returns:
            Dictionary containing merged result information
        """
        original_path = tif_info['original_path']
        filename = tif_info['filename']
        tiles_info = tif_info['tiles']

        logger.info(f"Merging predictions for {filename} ({len(tiles_info)} tiles)")

        # Read original TIF to get dimensions
        original_img = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
        if original_img is None:
            pil_img = Image.open(original_path)
            original_img = np.array(pil_img)
            if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        height, width = original_img.shape[:2]

        # Create output directory for GeoJSON
        os.makedirs(final_output_dir, exist_ok=True)

        # Merge labels and convert to GeoJSON
        geojson_result = self._merge_predicted_labels_to_geojson(tiles_info, tif_output_dir, height, width)

        # Save GeoJSON file
        geojson_path = os.path.join(final_output_dir, f"{filename}.geojson")
        with open(geojson_path, 'w') as f:
            json.dump(geojson_result, f, indent=2)

        logger.info(f"Saved GeoJSON to {geojson_path}")

        return {
            'original_path': original_path,
            'filename': filename,
            'geojson_path': geojson_path,
            'total_tiles': len(tiles_info),
            'total_detections': len(geojson_result['features'])
        }

    def _merge_predicted_labels_to_geojson(self, tiles_info: List[Dict], tif_output_dir: str, height: int, width: int) -> Dict:
        """
        Merge predicted labels from tiles and convert to GeoJSON format with geographic coordinates.

        Args:
            tiles_info: List of tile information (including geospatial data)
            tif_output_dir: Directory containing tile predictions
            height: Height of original image
            width: Width of original image

        Returns:
            GeoJSON FeatureCollection with geographic coordinates
        """
        # 按原始TIF文件分组
        tif_files_dict = {}
        for tile_info in tiles_info:
            tile_filename = tile_info['tile_filename']
            # 从tile文件名提取原始TIF文件名
            # 格式: original_filename_row_col.jpg -> original_filename
            original_tif_name = tile_info.get('original_filename', tile_filename.split('_')[0])
            
            if original_tif_name not in tif_files_dict:
                tif_files_dict[original_tif_name] = []
            tif_files_dict[original_tif_name].append(tile_info)
        
        logger.info(f"DEBUG: Found {len(tif_files_dict)} original TIF files")
        for tif_name, tiles in tif_files_dict.items():
            logger.info(f"DEBUG: TIF file '{tif_name}' has {len(tiles)} tiles")
        
        # 为每个原始TIF文件创建GeoJSON
        all_detections = []

        for original_tif_name, tiles in tif_files_dict.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing TIF file: {original_tif_name}")
            logger.info(f"{'='*80}")
            
            for tile_info in tiles:
                tile_filename = tile_info['tile_filename']
                coords = tile_info['coords']
                x_start, y_start, x_end, y_end = coords
                geospatial = tile_info.get('geospatial')

                # 打印tile的地理坐标范围
                if geospatial:
                    tile_geojson = geospatial.get('geojson')
                    if tile_geojson:
                        tile_coords_geo = tile_geojson['geometry']['coordinates'][0]
                        lon_tl, lat_tl = tile_coords_geo[0]
                        lon_tr, lat_tr = tile_coords_geo[1]
                        lon_br, lat_br = tile_coords_geo[2]
                        lon_bl, lat_bl = tile_coords_geo[3]
                        
                        min_lon = min(lon_tl, lon_bl)
                        max_lon = max(lon_tr, lon_br)
                        min_lat = min(lat_br, lat_bl)
                        max_lat = max(lat_tl, lat_tr)
                        
                        logger.info(f"\nTile: {tile_filename}")
                        logger.info(f"  Pixel coords: [{x_start}, {y_start}] -> [{x_end}, {y_end}]")
                        logger.info(f"  Geographic coords: [{min_lon:.6f}, {min_lat:.6f}] -> [{max_lon:.6f}, {max_lat:.6f}]")
                        logger.info(f"  Tile boundary: {tile_coords_geo}")

                # Read label file for this tile
                tile_stem = Path(tile_filename).stem
                label_filename = f"{tile_stem}.txt"

                # Read predicted label from predictions subdirectory
                predicted_label_path = os.path.join(self.output_base_path, "predictions", "labels", label_filename)
                
                if not os.path.exists(predicted_label_path):
                    logger.info(f"  No label file found: {label_filename}")
                    continue

                # Read labels
                with open(predicted_label_path, 'r') as f:
                    labels = f.readlines()
                
                logger.info(f"  Found {len(labels)} labels in {label_filename}")

                # Convert labels to absolute coordinates and merge
                for label in labels:
                    label = label.strip()
                    if not label:
                        continue

                    parts = label.split()
                    if len(parts) < 5:
                        continue

                    class_id = parts[0]
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    # Convert pixel coordinates to geographic coordinates if available
                    if geospatial:
                        try:
                            # Get tile GeoJSON boundary from preprocessing
                            tile_geojson = geospatial.get('geojson')
                            if tile_geojson:
                                # Use GeoJSON boundary from preprocessing
                                tile_coords_geo = tile_geojson['geometry']['coordinates'][0]
                                # Extract tile boundary coordinates
                                # tile_coords_geo format: [[lon_tl, lat_tl], [lon_tr, lat_tr], [lon_br, lat_br], [lon_bl, lat_bl], [lon_tl, lat_tl]]
                                lon_tl, lat_tl = tile_coords_geo[0]
                                lon_tr, lat_tr = tile_coords_geo[1]
                                lon_br, lat_br = tile_coords_geo[2]
                                lon_bl, lat_bl = tile_coords_geo[3]
                                
                                # Calculate tile geographic bounds
                                min_lon = min(lon_tl, lon_bl)
                                max_lon = max(lon_tr, lon_br)
                                min_lat = min(lat_br, lat_bl)
                                max_lat = max(lat_tl, lat_tr)
                                
                                # Convert YOLO normalized coordinates directly to geographic coordinates
                                # YOLO format: (x_center_norm, y_center_norm, width_norm, height_norm) all in [0, 1]
                                # GeoJSON boundary: [[lon_tl, lat_tl], [lon_tr, lat_tr], [lon_br, lat_br], [lon_bl, lat_bl]]
                                
                                # Note: YOLO y increases downward, but geographic y (latitude) increases upward
                                geo_x_center = min_lon + x_center_norm * (max_lon - min_lon)
                                geo_y_center = max_lat - y_center_norm * (max_lat - min_lat)
                                geo_width = width_norm * (max_lon - min_lon)
                                geo_height = height_norm * (max_lat - min_lat)
                                
                                # Convert to (x1, y1, x2, y2) format for GeoJSON polygon
                                geo_x1 = geo_x_center - geo_width / 2
                                geo_y1 = geo_y_center + geo_height / 2  # Top (higher latitude)
                                geo_x2 = geo_x_center + geo_width / 2
                                geo_y2 = geo_y_center - geo_height / 2  # Bottom (lower latitude)
                                
                                logger.info(f"    Detection (class {class_id}):")
                                logger.info(f"      YOLO coords: [{x_center_norm:.3f}, {y_center_norm:.3f}, {width_norm:.3f}, {height_norm:.3f}]")
                                logger.info(f"      Geographic coords: [{geo_x1:.6f}, {geo_y1:.6f}] -> [{geo_x2:.6f}, {geo_y2:.6f}]")
                                
                                # Create GeoJSON feature with geographic coordinates
                                feature = {
                                    "type": "Feature",
                                    "properties": {
                                        "class_id": int(class_id),
                                        "confidence": 1.0,
                                        "yolo_coords": [x_center_norm, y_center_norm, width_norm, height_norm],
                                        "tile_filename": tile_filename,
                                        "original_tif": original_tif_name,
                                        "crs": geospatial.get('crs')
                                    },
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [[
                                            [geo_x1, geo_y1],  # Top-left
                                            [geo_x2, geo_y1],  # Top-right
                                            [geo_x2, geo_y2],  # Bottom-right
                                            [geo_x1, geo_y2],  # Bottom-left
                                            [geo_x1, geo_y1]   # Close polygon
                                        ]]
                                    }
                                }
                            else:
                                # Fallback to bounds
                                tile_bounds = geospatial['bounds']  # [min_lon, min_lat, max_lon, max_lat]
                                min_lon, min_lat, max_lon, max_lat = tile_bounds
                                
                                # Convert YOLO normalized coordinates to geographic coordinates using bounds
                                geo_x_center = min_lon + x_center_norm * (max_lon - min_lon)
                                geo_y_center = max_lat - y_center_norm * (max_lat - min_lat)
                                geo_width = width_norm * (max_lon - min_lon)
                                geo_height = height_norm * (max_lat - min_lat)
                                
                                geo_x1 = geo_x_center - geo_width / 2
                                geo_y1 = geo_y_center + geo_height / 2
                                geo_x2 = geo_x_center + geo_width / 2
                                geo_y2 = geo_y_center - geo_height / 2
                                
                                feature = {
                                    "type": "Feature",
                                    "properties": {
                                        "class_id": int(class_id),
                                        "confidence": 1.0,
                                        "yolo_coords": [x_center_norm, y_center_norm, width_norm, height_norm],
                                        "tile_filename": tile_filename,
                                        "original_tif": original_tif_name,
                                        "crs": geospatial.get('crs')
                                    },
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [[
                                            [geo_x1, geo_y1],
                                            [geo_x2, geo_y1],
                                            [geo_x2, geo_y2],
                                            [geo_x1, geo_y2],
                                            [geo_x1, geo_y1]
                                        ]]
                                    }
                                }
                        except Exception as e:
                            logger.warning(f"Failed to convert to geographic coordinates for detection in {tile_filename}: {e}")
                            import traceback
                            logger.warning(f"Traceback: {traceback.format_exc()}")
                            # Fallback to pixel coordinates
                            tile_width = x_end - x_start
                            tile_height = y_end - y_start
                            
                            x_center = x_center_norm * tile_width + x_start
                            y_center = y_center_norm * tile_height + y_start
                            bbox_width = width_norm * tile_width
                            bbox_height = height_norm * tile_height
                            
                            x1 = x_center - bbox_width / 2
                            y1 = y_center - bbox_height / 2
                            x2 = x_center + bbox_width / 2
                            y2 = y_center + bbox_height / 2
                            
                            feature = {
                                "type": "Feature",
                                "properties": {
                                    "class_id": int(class_id),
                                    "confidence": 1.0,
                                    "tile_filename": tile_filename,
                                    "original_tif": original_tif_name
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[
                                        [x1, y1],
                                        [x2, y1],
                                        [x2, y2],
                                        [x1, y2],
                                        [x1, y1]
                                    ]]
                                }
                            }
                    else:
                        # No geospatial info, use pixel coordinates
                        tile_width = x_end - x_start
                        tile_height = y_end - y_start
                        
                        x_center = x_center_norm * tile_width + x_start
                        y_center = y_center_norm * tile_height + y_start
                        bbox_width = width_norm * tile_width
                        bbox_height = height_norm * tile_height
                        
                        x1 = x_center - bbox_width / 2
                        y1 = y_center - bbox_height / 2
                        x2 = x_center + bbox_width / 2
                        y2 = y_center + bbox_height / 2
                        
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "class_id": int(class_id),
                                "confidence": 1.0,
                                "tile_filename": tile_filename,
                                "original_tif": original_tif_name
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [x1, y1],
                                    [x2, y1],
                                    [x2, y2],
                                    [x1, y2],
                                    [x1, y1]
                                ]]
                            }
                        }

                    all_detections.append(feature)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Total detections before NMS: {len(all_detections)}")
        logger.info(f"{'='*80}")

        # Apply non-maximum suppression to remove duplicate detections
        filtered_detections = self._apply_nms_to_features(all_detections, iou_threshold=0.5)

        logger.info(f"Total detections after NMS: {len(filtered_detections)}")

        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "features": filtered_detections
        }

        return geojson

    def _apply_nms_to_features(self, features: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.

        Args:
            features: List of GeoJSON features
            iou_threshold: IoU threshold for NMS

        Returns:
            List of filtered features
        """
        if not features:
            return []

        # Extract bounding boxes from features
        detections = []
        for feature in features:
            coords = feature['geometry']['coordinates'][0]
            # Get bounding box from polygon coordinates
            x_coords = [coord[0] for coord in coords]
            y_coords = [coord[1] for coord in coords]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)

            detections.append({
                'class_id': feature['properties']['class_id'],
                'bbox': [x1, y1, x2, y2],
                'feature': feature
            })

        # Sort by class_id
        detections.sort(key=lambda x: x['class_id'])

        # Apply NMS for each class separately
        filtered_detections = []
        current_class = None
        class_detections = []

        for detection in detections:
            if detection['class_id'] != current_class:
                # Process previous class
                if class_detections:
                    filtered_detections.extend(self._nms_class_features(class_detections, iou_threshold))
                # Start new class
                current_class = detection['class_id']
                class_detections = [detection]
            else:
                class_detections.append(detection)

        # Process last class
        if class_detections:
            filtered_detections.extend(self._nms_class_features(class_detections, iou_threshold))

        # Return filtered features
        return [d['feature'] for d in filtered_detections]

    def _nms_class_features(self, detections: List[Dict], iou_threshold: float) -> List[Dict]:
        """
        Apply NMS for a single class.

        Args:
            detections: List of detections for same class
            iou_threshold: IoU threshold

        Returns:
            List of filtered detections
        """
        if not detections:
            return []

        # Sort by area (larger first)
        detections.sort(key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)

        filtered = []
        while detections:
            # Take the detection with the largest area
            best = detections.pop(0)
            filtered.append(best)

            # Remove detections with high IoU
            remaining = []
            for detection in detections:
                iou = self._calculate_iou(best['bbox'], detection['bbox'])
                if iou < iou_threshold:
                    remaining.append(detection)

            detections = remaining

        return filtered

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def postprocess(self, sample_type: str) -> Dict[str, str]:
        """
        Main postprocessing method that processes based on sample type.

        Args:
            sample_type: Type of sample ('tif' or 'image')

        Returns:
            Dictionary containing postprocessing information
        """
        if sample_type == 'tif':
            return self.postprocess_tif_results()
        elif sample_type == 'image':
            return self.postprocess_regular_images()
        else:
            raise ValueError(f"Unknown sample type: {sample_type}")
