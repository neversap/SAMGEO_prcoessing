import argparse
import base64
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the SAM GEO segmentation API.")
    parser.add_argument("--server", required=True, help="API server base URL.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--prompt", default="object", help="Concept prompt.")
    parser.add_argument("--box", help="Optional box: x1,y1,x2,y2.")
    parser.add_argument("--points", help="Optional points JSON: [[x,y,label], ...].")
    parser.add_argument("--out", required=True, help="Output mask PNG path.")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    form = {"prompt": args.prompt}
    if args.box:
        form["box"] = args.box
    if args.points:
        json.loads(args.points)
        form["points"] = args.points

    with image_path.open("rb") as file_obj:
        response = requests.post(
            f"{args.server.rstrip('/')}/segment",
            data=form,
            files={"image": (image_path.name, file_obj, "application/octet-stream")},
            timeout=120,
        )
    response.raise_for_status()

    payload = response.json()
    if not payload["masks"]:
        raise RuntimeError("Server returned no masks.")

    mask_bytes = base64.b64decode(payload["masks"][0]["png_base64"])
    out_path.write_bytes(mask_bytes)
    print(
        json.dumps(
            {
                "backend": payload["backend"],
                "size": [payload["width"], payload["height"]],
                "score": payload["masks"][0]["score"],
                "bbox": payload["masks"][0]["bbox"],
                "out": str(out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

