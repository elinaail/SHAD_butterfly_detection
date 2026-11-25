"""Download the supplied contest images; log provenance without model inference."""

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path("contest_qwen_2026_11_20")


def fetch(row):
    path = ROOT / row["local_path"]
    if path.is_absolute() or ".." in Path(row["local_path"]).parts:
        raise ValueError("Unsafe path")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            if not path.exists():
                response = requests.get(row["url"], timeout=(10, 45))
                response.raise_for_status()
                temporary = path.with_suffix(".download")
                temporary.write_bytes(response.content)
                with Image.open(temporary) as image:
                    image.verify()
                temporary.replace(path)
            with Image.open(path) as image:
                width, height = image.size
            return dict(
                photo_id=row["photo_id"],
                url=row["url"],
                local_path=row["local_path"],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                width=width,
                height=height,
            )
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def main():
    rows = [json.loads(s) for s in (ROOT / "manifest.jsonl").read_text().splitlines()]
    dataset = [json.loads(s) for s in (ROOT / "dataset.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["photo_id"] for r in rows}) == 400
    assert [(r["photo_id"], r["url"]) for r in rows] == [
        (r["photo_id"], r["photo_url"]) for r in dataset
    ]
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, r): r for r in rows}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row)
            with (ROOT / "logs/downloads.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"Downloaded {len(completed)}/400 photo={row['photo_id']}", flush=True
            )
    by_id = {r["photo_id"]: r for r in completed}
    (ROOT / "sources.jsonl").write_text(
        "".join(json.dumps(by_id[r["photo_id"]]) + "\n" for r in rows)
    )


if __name__ == "__main__":
    main()
