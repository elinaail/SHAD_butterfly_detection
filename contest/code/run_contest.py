"""Run the selected Qwen pipeline on the isolated 400-photo contest split."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_models import read_jsonl

ROOT = Path("contest_qwen_2026_11_20")
FOLDER = ROOT / "qwen"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(command, log, check=True):
    record = dict(started_utc=utc(), command=command)
    with (ROOT / "logs/commands.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    with (ROOT / "logs" / log).open("a") as output:
        result = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_ENABLE_MPS_FALLBACK="1"),
        )
    record.update(finished_utc=utc(), returncode=result.returncode)
    with (ROOT / "logs/commands.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    if check and result.returncode:
        raise RuntimeError(f"{command[1]} failed, see {log}")
    return result.returncode


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    (ROOT / "code").mkdir(exist_ok=True)
    scripts = [
        "run_models.py",
        "complete_qwen.py",
        "qwen_tiles.py",
        "export_predictions.py",
        "prefetch_contest.py",
        "run_contest.py",
        "package_contest.py",
    ]
    for name in scripts:
        shutil.copyfile(Path("scripts") / name, ROOT / "code" / name)
    shutil.copyfile("prompts/qwen_complete.txt", ROOT / "code/qwen_complete.txt")
    shutil.copyfile(
        "requirements-qwen-macos.txt", ROOT / "code/requirements-qwen-macos.txt"
    )
    original = json.loads(Path("results/qwen/candidates.meta.json").read_text())
    (ROOT / "original_configuration.json").write_text(
        json.dumps(original, indent=2) + "\n"
    )
    rows = read_jsonl(ROOT / "manifest.jsonl")
    dataset = read_jsonl(ROOT / "dataset.jsonl")
    assert len(rows) == len({r["photo_id"] for r in rows}) == 400
    assert [(r["photo_id"], r["url"]) for r in rows] == [
        (r["photo_id"], r["photo_url"]) for r in dataset
    ]
    # The downloader runs independently; inference starts only once all originals are verified.
    deadline = time.monotonic() + 600
    while not (ROOT / "sources.jsonl").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Downloads not complete: see logs/download.log")
        time.sleep(1)
    command = [
        sys.executable,
        "scripts/run_models.py",
        "--model",
        "qwen",
        "--input",
        str(ROOT / "manifest.jsonl"),
        "--root",
        str(ROOT),
        "--output",
        str(FOLDER / "candidates.jsonl"),
        "--revision",
        original["resolved_revision"],
        "--max-consecutive-errors",
        "400",
    ]
    run(command, "inference.log", check=False)
    candidates = (
        read_jsonl(FOLDER / "candidates.jsonl")
        if (FOLDER / "candidates.jsonl").exists()
        else []
    )
    errors_path = FOLDER / "candidates.errors.jsonl"
    errors = (
        {r["photo_id"]: r for r in read_jsonl(errors_path)}
        if errors_path.exists()
        else {}
    )
    answers = {r["photo_id"]: r for r in candidates}
    attempts = []
    for row in rows:
        answer = answers.get(row["photo_id"], errors.get(row["photo_id"]))
        if (
            answer is None
            or not answer.get("image_sha256")
            or answer.get("elapsed_seconds", 0) <= 0
        ):
            raise RuntimeError(f"Missing a timed inference attempt: {row['photo_id']}")
        attempts.append(answer)
    meta = json.loads((FOLDER / "candidates.meta.json").read_text())
    assert meta["resolved_revision"] == original["resolved_revision"]
    assert meta["versions"] == original["versions"]
    for key, value in original["config"].items():
        if key not in ["input_sha256", "revision_requested"]:
            assert meta["config"][key] == value, f"Configuration changed: {key}"
    (FOLDER / "baseline_attempts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in attempts)
    )
    (FOLDER / "baseline_attempts.meta.json").write_text(
        json.dumps(
            dict(
                meta,
                note="All baseline attempts, including explicitly marked inference errors retained for retry provenance.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "Baseline complete; applying the existing completion policy to unfinished answers.",
        flush=True,
    )
    run(
        [
            sys.executable,
            "scripts/complete_qwen.py",
            "--manifest",
            str(ROOT / "manifest.jsonl"),
            "--results-dir",
            str(FOLDER),
            "--image-root",
            str(ROOT),
            "--baseline-name",
            "baseline_attempts",
        ],
        "completion.log",
    )
    run(
        [
            sys.executable,
            "scripts/export_predictions.py",
            "--source",
            str(FOLDER / "completed_candidates.jsonl"),
            "--manifest",
            str(ROOT / "manifest.jsonl"),
            "--output",
            str(FOLDER / "predictions.jsonl"),
        ],
        "export.log",
    )
    run([sys.executable, "scripts/package_contest.py"], "validation.log")
    print("Contest Qwen run, validation and archive complete.", flush=True)


if __name__ == "__main__":
    main()
