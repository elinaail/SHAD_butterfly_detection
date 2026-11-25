"""Validate the isolated contest inference and package predictions plus audit logs."""

import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from run_models import read_jsonl, parse_qwen
from complete_qwen import merge_completed

ROOT = Path("contest_qwen_2026_11_20")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = read_jsonl(ROOT / "manifest.jsonl")
    dataset = read_jsonl(ROOT / "dataset.jsonl")
    rows = read_jsonl(ROOT / "qwen/predictions.jsonl")
    sources = read_jsonl(ROOT / "sources.jsonl")
    expected = [r["photo_id"] for r in manifest]
    assert len(expected) == len(set(expected)) == 400
    assert [r["photo_id"] for r in rows] == expected == [r["photo_id"] for r in dataset]
    assert len(sources) == 400 and [r["photo_id"] for r in sources] == expected
    original = json.loads((ROOT / "original_configuration.json").read_text())
    baseline_meta = json.loads((ROOT / "qwen/candidates.meta.json").read_text())
    for key, value in original["config"].items():
        if key not in ["input_sha256", "revision_requested"]:
            assert baseline_meta["config"][key] == value
    assert baseline_meta["versions"] == original["versions"]
    assert baseline_meta["resolved_revision"] == original["resolved_revision"]
    baseline = read_jsonl(ROOT / "qwen/baseline_attempts.jsonl")
    retries = []
    for row, source, entry in zip(rows, sources, manifest):
        assert (
            row["status"] == "ok"
            and not row.get("response_truncated")
            and row.get("ended_with_eos")
        )
        assert (
            row["image_sha256"]
            == source["sha256"]
            == digest(ROOT / entry["local_path"])
        )
        assert (
            row["image_width"] == source["width"]
            and row["image_height"] == source["height"]
        )
        assert math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
        assert len(row["boxes"]) == len(row["scores"]) and all(
            s is None for s in row["scores"]
        )
        for b in row["boxes"]:
            assert set(b) == {"left", "top", "width", "height"}
            assert all(math.isfinite(v) for v in b.values())
            assert (
                0 <= b["left"] < 1
                and 0 <= b["top"] < 1
                and b["width"] > 0
                and b["height"] > 0
            )
            assert (
                b["left"] + b["width"] <= 1 + 1e-9
                and b["top"] + b["height"] <= 1 + 1e-9
            )
        if row.get("completion_method"):
            retry = dict(row, elapsed_seconds=row["retry_elapsed_seconds"])
            retry.pop("threshold", None)
            retries.append(retry)
        else:
            assert parse_qwen(row["raw_text"]) == row["boxes"]
    completed = merge_completed(baseline, retries, ROOT / "qwen", "baseline_attempts")
    actual_completed = read_jsonl(ROOT / "qwen/completed_candidates.jsonl")
    assert completed == actual_completed
    for row, full in zip(rows, completed):
        assert {k: v for k, v in row.items() if k != "threshold"} == full
    clean = [dict(photo_id=r["photo_id"], boxes=r["boxes"]) for r in rows]
    prediction_path = ROOT / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in clean)
    )
    metadata = json.loads((ROOT / "qwen/completed_candidates.meta.json").read_text())
    report = dict(
        passed=True,
        photo_count=400,
        unique_photo_ids=400,
        statuses={"ok": 400},
        empty_images=sum(not r["boxes"] for r in rows),
        box_count=sum(len(r["boxes"]) for r in rows),
        baseline_settings_match_original=True,
        runtime_matches_original=True,
        resolved_revision=baseline_meta["resolved_revision"],
        retried_photo_ids=metadata["retried_photo_ids"],
        tiled_photo_ids=metadata["tiled_photo_ids"],
        manifest_sha256=digest(ROOT / "manifest.jsonl"),
        dataset_sha256=digest(ROOT / "dataset.jsonl"),
        predictions_sha256=digest(prediction_path),
        source_image_hashes_verified=400,
        original_project_results_directory="results/",
        contest_results_directory=str(ROOT),
    )
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "README.txt").write_text(
        """Qwen3-VL-2B-Instruct: отдельный экзаменационный набор Контеста, 400 фото.

predictions.jsonl — простой результат: photo_id и boxes (left/top/width/height
от 0 до 1 относительно оригинала). Пустой boxes — модель не нашла объектов.
Порядок и ID точно совпадают с присланными dataset.jsonl и manifest.jsonl.
Это предсказания модели, не эталонная разметка; метрики здесь не вычисляются.

qwen/predictions.jsonl — подробный результат со временем, SHA256 оригинала,
сырым текстом Qwen, количеством токенов и признаком EOS.
qwen/candidates* — первоначальный запуск; baseline_attempts* также сохраняет
неуспешные попытки, если они были. completed_candidates.meta.json описывает
дополнительную обработку оборванных ответов. У повторов сохранены все ответы
и параметры; у обработки фрагментами единого полнокадрового raw_text нет,
сырые ответы находятся в tiles. Никакие координаты вручную не исправлялись.

Настройки первоначального запуска совпадают с выбранной конфигурацией:
те же веса и промпт, 768 px, максимум 512 токенов, do_sample=false,
SDPA, FP16 MPS; версии runtime записаны в metadata. Применяется прежняя
политика дополнения незавершённых ответов (4096 токенов, 1024 px, penalty 1.1;
при необходимости 4x4 перекрывающихся фрагмента, как в предыдущем запуске).

logs/ — скачивания, команды и журналы инференса/экспорта/проверки.
sources.jsonl — URL, размеры и SHA256 всех 400 реально использованных фото.
code/ — снимок кода запуска. validation.json — результат проверки полноты,
координат, исходных фото и соответствия исходной конфигурации.
Время включает препроцессинг, модель, постпроцессинг и синхронизацию,
но не скачивание и загрузку весов. warmup отмечает первые вызовы процессов.
У повторов elapsed_seconds включает начальную и успешную повторную попытки.
Все файлы этого прогона отделены от результатов на датасете проекта.
"""
    )
    archive = Path.cwd().parent / "contest_qwen_results.zip"
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and "contest_photos" not in p.parts
        and "photos" not in p.parts
        and p.suffix not in [".lock", ".download", ".pyc"]
        and "__pycache__" not in p.parts
    ]
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(ROOT))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read("predictions.jsonl") == prediction_path.read_bytes()
        assert len(z.read("predictions.jsonl").splitlines()) == 400
    print(json.dumps(report, ensure_ascii=False), flush=True)
    print(f"Archive: {archive} ({archive.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
