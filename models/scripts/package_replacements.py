"""Validate the three new predictions and prepare a small Telegram handoff."""

import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from run_models import read_jsonl, parse_qwen


def main():
    root = Path(".").resolve()
    folder = root / "results/replacements_2026_11_18"
    manifest = read_jsonl(
        root / "inaturalist_dataset/replacements_2026_11_18/main.jsonl"
    )
    mapping = {r["replaces_photo_id"]: r["photo_id"] for r in manifest}
    new_ids = set(mapping.values())
    old_ids = {
        r["photo_id"] for r in read_jsonl(root / "inaturalist_dataset/main.jsonl")
    }
    assert (
        len(mapping) == len(new_ids) == 3
        and set(mapping) <= old_ids
        and not new_ids & old_ids
    )
    expected = (old_ids - set(mapping)) | new_ids
    assert len(expected) == 400
    patch = {}
    report = {
        "new_photo_ids": sorted(new_ids),
        "removed_photo_ids": sorted(mapping),
        "configurations": {},
    }
    sources = {r["photo_id"]: r for r in manifest}
    for row in manifest:
        assert (
            hashlib.sha256(Path(row["local_path"]).read_bytes()).hexdigest()
            == row["sha256"]
        )
    for model in ["dino", "owlv2", "qwen"]:
        paths = (
            [folder / model / "predictions.jsonl"]
            if model == "qwen"
            else [
                folder / model / f"predictions_t{t}.jsonl"
                for t in ["0.15", "0.25", "0.40"]
            ]
        )
        for path in paths:
            rows = read_jsonl(path)
            assert len(rows) == 3 and {r["photo_id"] for r in rows} == new_ids
            for row in rows:
                assert row["status"] == "ok" and not row.get("response_truncated")
                assert row["image_sha256"] == sources[row["photo_id"]]["sha256"]
                assert (
                    math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
                )
                assert len(row["scores"]) == len(row["boxes"])
                for box, score in zip(row["boxes"], row["scores"]):
                    assert all(math.isfinite(v) for v in box.values())
                    assert 0 <= box["left"] < 1 and 0 <= box["top"] < 1
                    assert box["width"] > 0 and box["height"] > 0
                    assert (
                        box["left"] + box["width"] <= 1 + 1e-9
                        and box["top"] + box["height"] <= 1 + 1e-9
                    )
                    assert score is None if model == "qwen" else 0 <= score <= 1
                if model == "qwen":
                    assert (
                        row["ended_with_eos"]
                        and parse_qwen(row["raw_text"]) == row["boxes"]
                    )
            original_path = root / "results" / model / path.name
            original_lines = original_path.read_text().splitlines()
            new_lines = {
                json.loads(line)["photo_id"]: line
                for line in path.read_text().splitlines()
            }
            combined = []
            unchanged = 0
            for line in original_lines:
                pid = json.loads(line)["photo_id"]
                if pid in mapping:
                    combined.append(new_lines[mapping[pid]])
                else:
                    combined.append(line)
                    unchanged += 1
            assert unchanged == 397 and len(combined) == 400
            assert {json.loads(line)["photo_id"] for line in combined} == expected
            destination = root / "results/main_v2" / model / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("\n".join(combined) + "\n")
            name = (
                "qwen.jsonl"
                if model == "qwen"
                else f'{model}_t{path.stem.split("_t")[1]}.jsonl'
            )
            compact = [
                {
                    k: r[k]
                    for k in [
                        "photo_id",
                        "boxes",
                        "scores",
                        "elapsed_seconds",
                        "warmup",
                    ]
                }
                for r in rows
            ]
            patch[name] = "".join(
                json.dumps(r, separators=(",", ":")) + "\n" for r in compact
            ).encode()
            report["configurations"][name] = dict(
                new_rows=3,
                updated_total=400,
                old_rows_unchanged=unchanged,
                new_box_counts={r["photo_id"]: len(r["boxes"]) for r in rows},
                merged_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            )
    patch["replacements.json"] = (
        json.dumps(
            [{"old_photo_id": k, "new_photo_id": v} for k, v in mapping.items()],
            indent=2,
        )
        + "\n"
    ).encode()
    patch["README.txt"] = (
        """Предсказания на трёх новых фото, заменяющих старые:
615002914 -> 765032073
247404664 -> 765036915
359179108 -> 764941215

В каждом JSONL ровно 3 новых photo_id. В прежних файлах model_results.zip
удалить три старых ID и добавить строки из одноимённых файлов этого архива.
Остальные 397 строк не менять. Получится одинаковый обновлённый набор 400 фото
для всех 7 конфигураций: DINO и OWLv2 при 0.15/0.25/0.40, Qwen.

Формат прежний: photo_id, boxes (left/top/width/height от 0 до 1), scores,
elapsed_seconds, warmup. У Qwen scores=null для каждой рамки.
Основные варианты: dino_t0.25.jsonl, owlv2_t0.25.jsonl, qwen.jsonl.

Все три фото реально прогнаны заново. Веса, версии библиотек, промпты,
пороги и постобработка совпадают с первоначальными конфигурациями.
Qwen: исходный промпт, 768 px, 512 токенов, без sampling, SDPA;
все три ответа завершены с EOS, повторная обработка не понадобилась.
Ни gold, ни метаданные таксонов моделям не передавались; рамки не правились.

Время — препроцессинг + модель + постобработка + синхронизация, без загрузки
весов и фото; M3/MPS. Первый вызов каждого процесса warmup=true.
DINO/OWLv2: PyTorch 2.7, Qwen: 2.15. Пороговые варианты получены из одного
запуска, их время одинаковое. Это предсказания для оценки, не gold-разметка.
""".encode()
    )
    archive = root.parent / "model_replacements_results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in patch.items():
            z.writestr(name, data)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name in patch:
            assert z.read(name) == patch[name]
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Validated 7 configurations x 3 new photos; every merged export has 400 IDs and 397 unchanged rows."
    )
    print(archive.resolve(), archive.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
