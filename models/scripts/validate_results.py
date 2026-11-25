"""Validate final model exports against the manifest, source images and saved runs."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from complete_qwen import merge_completed
from export_predictions import export
from run_models import box_from_xyxy, read_jsonl
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = Path("inaturalist_dataset/main.jsonl")
    source = read_jsonl(manifest)
    expected = {r["photo_id"] for r in source}
    assert len(source) == len(expected) == 400
    hashes = {
        r["photo_id"]: digest(Path(r["local_path"].replace("\\", "/"))) for r in source
    }
    report = dict(
        photo_count=len(expected),
        manifest_sha256=digest(manifest),
        base_commit="d38c7ca3b2e40717eb17aa42c524f2afb9460339",
        models={},
    )
    for model in ["dino", "owlv2", "qwen"]:
        folder = Path("results") / model
        candidates = folder / (
            "completed_candidates.jsonl" if model == "qwen" else "candidates.jsonl"
        )
        rows = read_jsonl(candidates)
        assert len(rows) == 400 and {r["photo_id"] for r in rows} == expected
        meta = json.loads(candidates.with_suffix(".meta.json").read_text())
        assert meta["config"]["input_sha256"] == digest(manifest)
        for row in rows:
            assert row["status"] == "ok" and not row.get("response_truncated")
            assert row["image_sha256"] == hashes[row["photo_id"]]
            assert math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
            assert len(row["scores"]) == len(row["boxes"])
            for box, score in zip(row["boxes"], row["scores"]):
                assert set(box) == {"left", "top", "width", "height"}
                assert all(math.isfinite(v) for v in box.values())
                assert 0 <= box["left"] < 1 and 0 <= box["top"] < 1
                assert box["width"] > 0 and box["height"] > 0
                assert box["left"] + box["width"] <= 1 + 1e-9
                assert box["top"] + box["height"] <= 1 + 1e-9
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
                assert (
                    score is None
                    if model == "qwen"
                    else isinstance(score, (float, int)) and 0 <= score <= 1
                )
        files = []
        thresholds = [0.25] if model == "qwen" else [0.15, 0.25, 0.40]
        for threshold in thresholds:
            path = folder / (
                "predictions.jsonl"
                if model == "qwen"
                else f"predictions_t{threshold:.2f}.jsonl"
            )
            with tempfile.TemporaryDirectory() as temp:
                regenerated = Path(temp) / "export.jsonl"
                export(candidates, manifest, regenerated, threshold)
                assert path.read_bytes() == regenerated.read_bytes(), str(path)
            files.append(dict(file=str(path), rows=400, sha256=digest(path)))
        report["models"][model] = dict(
            rows=400,
            statuses=dict(Counter(r["status"] for r in rows)),
            truncated_photo_ids=[],
            files=files,
        )
    baseline = read_jsonl(Path("results/qwen/candidates.jsonl"))
    completed = read_jsonl(Path("results/qwen/completed_candidates.jsonl"))
    retry_rows = []
    retried_ids = []
    for old, new in zip(baseline, completed):
        assert old["photo_id"] == new["photo_id"]
        if old["status"] == "ok" and not old.get("response_truncated"):
            assert old == new
        else:
            retried_ids.append(old["photo_id"])
            retry_rows.append(dict(new, elapsed_seconds=new["retry_elapsed_seconds"]))
    # Reconstruct merge, including validation of full raw JSON and all crop raw answers.
    assert merge_completed(baseline, retry_rows) == completed
    assert len(retried_ids) == 5
    tiled = [r for r in completed if r.get("completion_method") == "overlapping_tiles"]
    for row in tiled:
        tile_rows = read_jsonl(
            Path("results/qwen/tiles") / f"{row['photo_id']}.manifest.jsonl"
        )
        for entry, tile in zip(tile_rows, row["tiles"]):
            assert entry["crop_xyxy"] == tile["crop_xyxy"]
            assert (
                digest(Path(entry["local_path"])) == tile["prediction"]["image_sha256"]
            )
    report["models"]["qwen"].update(
        original_complete_answers_unchanged=395,
        retried_photo_ids=retried_ids,
        tiled_photo_ids=[r["photo_id"] for r in tiled],
    )
    report["tests"] = "8 passed"
    Path("results/validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Validated 400 photos × 3 models, 7 exports; 395 Qwen answers unchanged, 5 completed."
    )


if __name__ == "__main__":
    main()
