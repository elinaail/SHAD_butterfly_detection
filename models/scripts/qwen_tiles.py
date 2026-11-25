"""Complete difficult Qwen images using overlapping crops and geometric NMS."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from run_models import box_from_xyxy, nms, parse_qwen, read_jsonl


def combine_tiles(tiles, width, height):
    boxes, priorities = [], []
    elapsed = 0.0
    for tile in tiles:
        prediction = tile["prediction"]
        if (
            prediction["status"] != "ok"
            or not prediction.get("ended_with_eos")
            or prediction.get("response_truncated")
        ):
            raise ValueError("Incomplete tile response")
        if parse_qwen(prediction["raw_text"]) != prediction["boxes"]:
            raise ValueError("Tile boxes differ from raw answer")
        x0, y0, x1, y1 = tile["crop_xyxy"]
        elapsed += prediction["elapsed_seconds"]
        for b in prediction["boxes"]:
            boxes.append(
                box_from_xyxy(
                    [
                        x0 + b["left"] * (x1 - x0),
                        y0 + b["top"] * (y1 - y0),
                        x0 + (b["left"] + b["width"]) * (x1 - x0),
                        y0 + (b["top"] + b["height"]) * (y1 - y0),
                    ],
                    width,
                    height,
                )
            )
            # Prefer boxes farther from crop edges; this is not model confidence.
            priorities.append(
                min(
                    b["left"],
                    b["top"],
                    1 - b["left"] - b["width"],
                    1 - b["top"] - b["height"],
                )
            )
    return [boxes[i] for i in nms(boxes, priorities, 0.5)], elapsed


def recover(
    row, manifest_row, revision, results_dir=Path("results/qwen"), image_root=Path(".")
):
    from PIL import Image

    folder = results_dir / "tiles"
    folder.mkdir(parents=True, exist_ok=True)
    cache = (
        Path("inaturalist_dataset/photos/qwen_tiles")
        if results_dir == Path("results/qwen")
        else folder / "photos"
    ) / str(row["photo_id"])
    cache.mkdir(parents=True, exist_ok=True)
    # local_path in the main manifest is relative to the project root.
    original = image_root / Path(manifest_row["local_path"].replace("\\", "/"))
    if hashlib.sha256(original.read_bytes()).hexdigest() != row["image_sha256"]:
        raise ValueError("Source image hash changed")
    entries = []
    with Image.open(original) as source:
        image = source.convert("RGB")
        w, h = image.size
        for iy in range(4):
            for ix in range(4):
                # Four cells per axis, expanded 10% of cell width on each side.
                bounds = [
                    max(0, round((ix - 0.1) * w / 4)),
                    max(0, round((iy - 0.1) * h / 4)),
                    min(w, round((ix + 1.1) * w / 4)),
                    min(h, round((iy + 1.1) * h / 4)),
                ]
                index = iy * 4 + ix
                path = cache / f"{index:02}.png"
                image.crop(bounds).save(path)
                entries.append(
                    dict(photo_id=index, local_path=str(path), crop_xyxy=bounds)
                )
    manifest = folder / f"{row['photo_id']}.manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in entries))
    output = folder / f"{row['photo_id']}.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_models.py",
            "--model",
            "qwen",
            "--input",
            str(manifest),
            "--root",
            ".",
            "--output",
            str(output),
            "--prompt-file",
            "prompts/qwen_complete.txt",
            "--max-new-tokens",
            "1024",
            "--image-max-side",
            "768",
            "--repetition-penalty",
            "1.15",
            "--require-complete",
            "--revision",
            revision,
        ],
        check=True,
    )
    predictions = {r["photo_id"]: r for r in read_jsonl(output)}
    if set(predictions) != set(range(16)):
        raise ValueError("Missing tile predictions")
    tiles = [
        dict(crop_xyxy=e["crop_xyxy"], prediction=predictions[e["photo_id"]])
        for e in entries
    ]
    boxes, elapsed = combine_tiles(tiles, w, h)
    return dict(
        row,
        boxes=boxes,
        scores=[None] * len(boxes),
        status="ok",
        raw_text=None,
        response_truncated=False,
        ended_with_eos=True,
        generated_tokens=sum(t["prediction"]["generated_tokens"] for t in tiles),
        elapsed_seconds=elapsed,
        warmup=any(t["prediction"]["warmup"] for t in tiles),
        completion_method="overlapping_tiles",
        tiles=tiles,
        retry_metadata=str(output.with_suffix(".meta.json")),
        raw_text_note="No single full-image answer: complete raw answers are in tiles[].prediction.raw_text.",
    )
