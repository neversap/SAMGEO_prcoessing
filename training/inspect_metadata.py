from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.config import DEFAULT_INDEX_CSV, DEFAULT_STATS_JSON
from data_process_pipeline.dataloader import read_mask


def main() -> None:
    args = parse_args()
    index_path = Path(args.index_csv)
    stats_path = Path(args.stats_json)
    if not index_path.exists():
        raise FileNotFoundError(f"metadata index does not exist: {index_path}")
    with index_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        missing_paths = []
        for index, row in enumerate(reader):
            if index < args.limit:
                if args.inspect_mask_values:
                    row["mask_values"] = _mask_values(row["mask_path"])
                rows.append(row)
            if args.check_paths and (not Path(row["image_path"]).exists() or not Path(row["mask_path"]).exists()):
                missing_paths.append(row.get("sample_id", str(index)))
                if len(missing_paths) >= 20:
                    break
    print(f"index_csv={index_path}")
    print(f"first_rows={len(rows)}")
    for row in rows:
        print(
            json.dumps(
                {
                    "sample_id": row.get("sample_id"),
                    "split": row.get("split"),
                    "country": row.get("country"),
                    "window": row.get("window"),
                    "bucket": row.get("bucket"),
                    "image_path": row.get("image_path"),
                    "mask_path": row.get("mask_path"),
                    "mask_values": row.get("mask_values"),
                },
                ensure_ascii=False,
            )
        )
    if stats_path.exists():
        print("stats=" + stats_path.read_text(encoding="utf-8"))
    else:
        print(f"stats_json_missing={stats_path}")
    if missing_paths:
        raise FileNotFoundError(f"Missing image/mask paths for samples: {missing_paths}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect FTW dataloader metadata.")
    parser.add_argument("--index-csv", default=str(DEFAULT_INDEX_CSV))
    parser.add_argument("--stats-json", default=str(DEFAULT_STATS_JSON))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--inspect-mask-values", action="store_true")
    return parser.parse_args()


def _mask_values(path: str) -> list[int]:
    import numpy as np

    values = np.unique(read_mask(Path(path)))
    return [int(value) for value in values[:50]]


if __name__ == "__main__":
    main()
