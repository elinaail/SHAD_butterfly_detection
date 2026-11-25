"""Retry unfinished Qwen answers and assemble a complete, auditable 400-photo run."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from run_models import parse_qwen, read_jsonl
from qwen_tiles import combine_tiles, recover


def merge_completed(
    baseline, retries, metadata_folder=Path("results/qwen"), baseline_name="candidates"
):
    original = {row["photo_id"]: row for row in baseline}
    replacements = {row["photo_id"]: row for row in retries}
    if len(original) != len(baseline) or len(replacements) != len(retries):
        raise ValueError("Duplicate photo IDs")
    unfinished = {
        pid
        for pid, row in original.items()
        if row["status"] != "ok"
        or row.get("response_truncated", False)
        or not row.get("ended_with_eos", True)
    }
    if set(replacements) != unfinished:
        raise ValueError("Retry IDs must exactly match the unfinished baseline answers")
    result = []
    for row in baseline:
        if row["photo_id"] not in unfinished:
            result.append(dict(row))
            continue
        retry = replacements[row["photo_id"]]
        if (
            retry["status"] != "ok"
            or retry.get("response_truncated")
            or not retry.get("ended_with_eos")
        ):
            raise ValueError("Retry response is not complete")
        if (
            retry["model"] != row["model"]
            or retry["image_sha256"] != row["image_sha256"]
        ):
            raise ValueError("Retry model or source image differs from baseline")
        if retry.get("completion_method") == "overlapping_tiles":
            boxes, seconds = combine_tiles(
                retry["tiles"], retry["image_width"], retry["image_height"]
            )
            if (
                boxes != retry["boxes"]
                or abs(seconds - retry["elapsed_seconds"]) > 1e-8
            ):
                raise ValueError("Tiled result differs from raw tile answers")
        elif parse_qwen(retry["raw_text"]) != retry["boxes"]:
            raise ValueError("Retry boxes differ from its complete raw answer")
        result.append(
            dict(
                retry,
                elapsed_seconds=row["elapsed_seconds"] + retry["elapsed_seconds"],
                initial_elapsed_seconds=row["elapsed_seconds"],
                retry_elapsed_seconds=retry["elapsed_seconds"],
                warmup=row.get("warmup", False) or retry.get("warmup", False),
                completion_method=retry.get("completion_method", "retry_full"),
                initial_status=row["status"],
                initial_metadata=str(metadata_folder / f"{baseline_name}.meta.json"),
                retry_metadata=retry.get(
                    "retry_metadata", str(metadata_folder / "retry_full.meta.json")
                ),
            )
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=Path("inaturalist_dataset/main.jsonl")
    )
    p.add_argument("--results-dir", type=Path, default=Path("results/qwen"))
    p.add_argument("--image-root", type=Path, default=Path("."))
    p.add_argument("--baseline-name", default="candidates")
    args = p.parse_args()
    folder = args.results_dir
    baseline = read_jsonl(folder / f"{args.baseline_name}.jsonl")
    expected = {r["photo_id"] for r in read_jsonl(args.manifest)}
    if len(baseline) != len(expected) or {r["photo_id"] for r in baseline} != expected:
        raise ValueError("Baseline is incomplete or uses a different manifest")
    baseline_meta = json.loads((folder / f"{args.baseline_name}.meta.json").read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if baseline_meta["config"]["input_sha256"] != manifest_hash:
        raise ValueError("Manifest differs from baseline")
    unfinished = [
        r["photo_id"]
        for r in baseline
        if r["status"] != "ok"
        or r.get("response_truncated", False)
        or not r.get("ended_with_eos", True)
    ]
    retry_meta = None
    retries = []
    if unfinished:
        # Previously attempted full-image failures go directly to crop recovery.
        attempted_errors = folder / "retry_full.errors.jsonl"
        failed_full = (
            {r["photo_id"] for r in read_jsonl(attempted_errors)}
            if attempted_errors.exists()
            else set()
        )
        requested = [pid for pid in unfinished if pid not in failed_full]
        if requested:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_models.py",
                    "--model",
                    "qwen",
                    "--input",
                    str(args.manifest),
                    "--root",
                    str(args.image_root),
                    "--output",
                    str(folder / "retry_full.jsonl"),
                    "--photo-ids",
                    *map(str, requested),
                    "--prompt-file",
                    "prompts/qwen_complete.txt",
                    "--max-new-tokens",
                    "4096",
                    "--image-max-side",
                    "1024",
                    "--repetition-penalty",
                    "1.1",
                    "--require-complete",
                    "--revision",
                    baseline_meta["resolved_revision"],
                ],
                check=False,
            )
        retry_meta = json.loads((folder / "retry_full.meta.json").read_text())
        if retry_meta["resolved_revision"] != baseline_meta["resolved_revision"]:
            raise ValueError("Retry model revision differs from baseline")
        retries = (
            read_jsonl(folder / "retry_full.jsonl")
            if (folder / "retry_full.jsonl").exists()
            else []
        )
        remaining = set(unfinished) - {r["photo_id"] for r in retries}
        source_rows = {r["photo_id"]: r for r in read_jsonl(args.manifest)}
        for row in baseline:
            if row["photo_id"] in remaining:
                retries.append(
                    recover(
                        row,
                        source_rows[row["photo_id"]],
                        baseline_meta["resolved_revision"],
                        results_dir=folder,
                        image_root=args.image_root,
                    )
                )
    completed = merge_completed(baseline, retries, folder, args.baseline_name)
    out = folder / "completed_candidates.jsonl"
    temporary = out.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r) + "\n" for r in completed))
    temporary.replace(out)
    metadata = {
        "config": {
            "model": baseline_meta["config"]["model"],
            "input_sha256": manifest_hash,
            "threshold": baseline_meta["config"]["threshold"],
            "pipeline": "baseline_then_retry_unfinished",
        },
        "resolved_revision": baseline_meta["resolved_revision"],
        "baseline_configuration": baseline_meta["config"],
        "retry_configuration": retry_meta["config"] if retry_meta else None,
        "versions": baseline_meta["versions"],
        "retry_versions": retry_meta["versions"] if retry_meta else None,
        "retried_photo_ids": unfinished,
        "timing": "For retried photos, elapsed_seconds includes initial inference plus successful retry (sum of crop calls for tiled recovery). Both components are retained. Failed additional full-image retries are excluded from elapsed_seconds and remain in error logs when available; do not use these rows for baseline speed comparison.",
        "tiled_photo_ids": [
            r["photo_id"]
            for r in completed
            if r.get("completion_method") == "overlapping_tiles"
        ],
        "tile_protocol": "4x4 cells expanded 10% of cell width per side; map crop boxes to original coordinates; NMS IoU 0.5 prioritizes distance from crop edges, not model confidence. Crop metadata and all raw answers are preserved.",
        "note": f"The {len(baseline)-len(unfinished)} originally complete predictions are unchanged. Original unsuccessful outputs remain in {args.baseline_name}.jsonl. No human boxes or gold labels were used.",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Complete Qwen answers: {len(completed)}; retried: {len(unfinished)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
