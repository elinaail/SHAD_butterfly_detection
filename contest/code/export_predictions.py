"""Validate a complete run and export the agreed gold-compatible box format."""

import argparse
import json
from pathlib import Path
from run_models import read_jsonl, box_from_xyxy


def export(source, manifest, destination, threshold):
    rows = read_jsonl(source)
    expected = [r["photo_id"] for r in read_jsonl(manifest)]
    by_id = {r["photo_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate predictions")
    if set(by_id) != set(expected):
        raise ValueError(
            f"Incomplete or wrong split: expected {len(expected)}, got {len(by_id)}"
        )
    output = []
    for photo_id in expected:
        row = by_id[photo_id]
        if row["status"] not in {"ok", "truncated"} or len(row["boxes"]) != len(
            row["scores"]
        ):
            raise ValueError(f"Invalid prediction: {photo_id}")
        boxes, scores = [], []
        for box, score in zip(row["boxes"], row["scores"]):
            box_from_xyxy(
                [
                    box["left"],
                    box["top"],
                    box["left"] + box["width"],
                    box["top"] + box["height"],
                ],
                1,
                1,
            )
            if score is None or score >= threshold:
                boxes.append(box)
                scores.append(score)
        output.append(
            dict(
                row,
                boxes=boxes,
                scores=scores,
                threshold=threshold if not row["model"].startswith("Qwen/") else None,
            )
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as f:
        for row in output:
            f.write(json.dumps(row) + "\n")
    return len(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path, default=Path("inaturalist_dataset/main.jsonl")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.25)
    a = p.parse_args()
    meta = json.loads(a.source.with_suffix(".meta.json").read_text())
    import hashlib

    if (
        meta["config"]["input_sha256"]
        != hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    ):
        p.error("Manifest differs from the inference run")
    if a.threshold < meta["config"]["threshold"]:
        p.error("Requested threshold is below the saved candidate floor")
    print(f"Exported {export(a.source, a.manifest, a.output, a.threshold)} images")
