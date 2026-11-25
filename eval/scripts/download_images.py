"""Скачивает large-версии (1024 px) 400 фото в eval/cache_images. Нужны только для картинок."""
import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "cache_images"
URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{}/large.{}"


def download(rec):
    pid = rec["photo_id"]
    ext = rec["url"].rsplit(".", 1)[-1]  # регистр важен: JPG и jpg разные ключи
    dst = OUT / f"{pid}.jpg"
    if dst.exists() and dst.stat().st_size > 1000:
        return pid, "cached"
    status = None
    for attempt in range(4):
        try:
            r = requests.get(URL.format(pid, ext), timeout=60, headers={"User-Agent": "butterfly-eval/1.0"})
            status = r.status_code
            if status == 200:
                dst.write_bytes(r.content)
                return pid, "ok"
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return pid, f"fail {status}"


OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(ROOT / "gold_400_for_metrics" / "main_metadata_v2.json"))
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(download, meta))
failed = [r for r in res if r[1].startswith("fail")]
print(f"{len(res)} files, failed: {failed}")
