from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inhouse_inference.predictor import InhousePredictor


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    predictor = InhousePredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        input_size=args.input_size,
        strict=args.strict,
    )
    result = predictor.predict_to_files(
        image=args.image,
        output_dir=args.output,
        min_area=args.min_area,
        epsilon_ratio=args.epsilon_ratio,
    )
    print(json.dumps({
        "original_size": result["original_size"],
        "model_input_size": result["model_input_size"],
        "field_area_ratio": result["field_area_ratio"],
        "boundary_area_ratio": result["boundary_area_ratio"],
        "polygon_count": result["polygon_count"],
        "largest_bbox": result["polygons"][0]["bbox"] if result["polygons"] else [],
        "output": str(Path(args.output)),
    }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick-check inhouse segmentation inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--epsilon-ratio", type=float, default=0.003)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
