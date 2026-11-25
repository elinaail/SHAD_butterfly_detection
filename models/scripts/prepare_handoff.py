"""Build an offline, audited handoff from saved runs. Never invokes inference."""
import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = args.destination.resolve()
    if dest.exists():
        raise SystemExit("Destination exists; choose a new directory to preserve prior work")
    dest.mkdir(parents=True)
    bundle = dest / "SHAD_project_handoff"
    bundle.mkdir()
    project = bundle / "project"
    project.mkdir()
    audit = {"checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=3))).isoformat(),
             "mode": "offline_saved_results_only", "inference_started": False,
             "uploads_or_messages_sent": False, "checks": [], "configurations": {}}
    source_rows = read(root / "inaturalist_dataset/main.jsonl")
    patch_rows = read(root / "inaturalist_dataset/replacements_2026_11_18/main.jsonl")
    original = {x["photo_id"]: x for x in source_rows}
    mapping = {x["replaces_photo_id"]: x["photo_id"] for x in patch_rows}
    assert len(original) == 400 and len(mapping) == 3
    assert set(mapping) <= original.keys() and not set(mapping.values()) & original.keys()
    final_sources = [next((p for p in patch_rows if p["replaces_photo_id"] == x["photo_id"]), x)
                     for x in source_rows]
    expected = {x["photo_id"] for x in final_sources}
    image_hashes = {}
    for row in source_rows + patch_rows:
        p = root / row["local_path"].replace("\\", "/")
        assert p.is_file(), p
        image_hashes[row["photo_id"]] = sha(p)
        if row.get("sha256"):
            assert image_hashes[row["photo_id"]] == row["sha256"]
    audit["checks"].append("403 original/replacement image files exist and were hashed")
    table = []
    for model in ["dino", "owlv2", "qwen"]:
        names = ["predictions.jsonl"] if model == "qwen" else [f"predictions_t{x}.jsonl" for x in ["0.15", "0.25", "0.40"]]
        base_meta = json.loads((root / "results" / model / "candidates.meta.json").read_text())
        patch_meta = json.loads((root / "results/replacements_2026_11_18" / model / "candidates.meta.json").read_text())
        assert base_meta["resolved_revision"] == patch_meta["resolved_revision"]
        for key in ["model", "device", "prompt", "max_new_tokens", "qwen_max_image_side", "attention_implementation", "nms_iou", "threshold"]:
            assert base_meta["config"].get(key) == patch_meta["config"].get(key), (model, key)
        for name in names:
            old = read(root / "results" / model / name)
            patch = read(root / "results/replacements_2026_11_18" / model / name)
            final = read(root / "results/main_v2" / model / name)
            old_map, patch_map = ({x["photo_id"]: x for x in rows} for rows in [old, patch])
            assert len(final) == len({x["photo_id"] for x in final}) == 400
            assert {x["photo_id"] for x in final} == expected
            assert len(patch_map) == 3 and set(patch_map) == set(mapping.values())
            untouched = 0
            for row in final:
                pid = row["photo_id"]
                assert row == (patch_map[pid] if pid in patch_map else old_map[pid])
                untouched += pid not in patch_map
                assert row["status"] == "ok" and not row.get("response_truncated")
                assert row["image_sha256"] == image_hashes[pid]
                assert math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
                assert len(row["boxes"]) == len(row["scores"])
                for b, score in zip(row["boxes"], row["scores"]):
                    assert set(b) == {"left", "top", "width", "height"}
                    assert all(math.isfinite(v) for v in b.values())
                    assert 0 <= b["left"] < 1 and 0 <= b["top"] < 1
                    assert b["width"] > 0 and b["height"] > 0
                    assert b["left"] + b["width"] <= 1 + 1e-9
                    assert b["top"] + b["height"] <= 1 + 1e-9
                    assert score is None if model == "qwen" else 0 <= score <= 1
            assert untouched == 397
            timing = [x["elapsed_seconds"] for x in final if not x.get("warmup") and not x.get("completion_method")]
            key = model + "/" + name
            row = {"configuration": key, "photos": 400, "boxes": sum(len(x["boxes"]) for x in final),
                   "empty_images": sum(not x["boxes"] for x in final), "unchanged_old_rows": untouched,
                   "replacement_rows": 3, "nonbaseline_rows": sum(bool(x.get("completion_method")) for x in final),
                   "timing_samples_no_warmup_or_retry": len(timing),
                   "median_seconds_no_warmup_or_retry": round(statistics.median(timing), 4),
                   "sha256": sha(root / "results/main_v2" / model / name)}
            audit["configurations"][key] = row
            table.append(row)
    baseline = read(root / "results/qwen/candidates.jsonl")
    completed = read(root / "results/qwen/completed_candidates.jsonl")
    truncated = [x["photo_id"] for x in baseline if x.get("response_truncated")]
    assert len(truncated) == 5
    old_complete = {x["photo_id"]: x for x in completed}
    assert sum(x == old_complete[x["photo_id"]] for x in baseline if x["photo_id"] not in truncated) == 395
    audit["qwen_original_truncated_ids"] = truncated
    audit["qwen_original_untouched_complete_rows"] = 395
    audit["qwen_final_retained_retry_ids"] = [x["photo_id"] for x in read(root / "results/main_v2/qwen/predictions.jsonl") if x.get("completion_method")]
    audit["checks"] += ["Seven final exports each contain exactly 400 matching unique photo IDs",
                         "All seven exports preserve 397 original rows byte-content-equivalently as JSON and use three replacement predictions",
                         "Final boxes, scores, image hashes, statuses and timing validated",
                         "Replacement model revisions and key inference parameters match baseline",
                         "395 originally complete Qwen raw records remain unchanged"]
    # Copy saved data exactly; omit environments, weights, images, git internals and lock files.
    for folder in ["results", "logs", "scripts", "prompts", "tests"]:
        for p in sorted((root / folder).rglob("*")):
            if not p.is_file() or p.suffix in {".lock", ".pyc"} or "__pycache__" in p.parts:
                continue
            target = project / p.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    for name in ["MODELS.md", "STATUS.md", "requirements.txt", "requirements-models.txt", "requirements-qwen-macos.txt"]:
        shutil.copy2(root / name, project / name)
    for name in ["main.jsonl", "metadata.jsonl", "replacements_2026_11_18/main.jsonl", "replacements_2026_11_18/replacements.json"]:
        target = project / "inaturalist_dataset" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "inaturalist_dataset" / name, target)
    # Keep the already delivered exam snapshot separate and do not include its gold labels or photos.
    exam = bundle / "exam_contest_separate"
    for p in sorted((root / "contest_qwen_2026_11_20").rglob("*")):
        if not p.is_file() or p.suffix.lower() in {".lock", ".pyc", ".jpg", ".png", ".jpeg", ".webp"} or "__pycache__" in p.parts or "contest_photos" in p.parts or "photos" in p.parts:
            continue
        target = exam / p.relative_to(root / "contest_qwen_2026_11_20")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    portable_sources = []
    for row in final_sources:
        portable_sources.append({"photo_id": row["photo_id"], "url": row["url"],
                                 "sha256": image_hashes[row["photo_id"]],
                                 "replaces_photo_id": row.get("replaces_photo_id")})
    (project / "inaturalist_dataset/main_v2_sources.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in portable_sources))
    write_json(bundle / "checks/validation.json", audit)
    write_json(bundle / "checks/replacements.json", mapping)
    buff = io.StringIO()
    writer = csv.DictWriter(buff, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
    (bundle / "checks/results_summary.csv").write_text(buff.getvalue())
    # These are data summaries, not claims of detection accuracy.
    summaries = "\n".join(f"| {x['configuration']} | {x['boxes']} | {x['empty_images']} | {x['timing_samples_no_warmup_or_retry']} | {x['median_seconds_no_warmup_or_retry']} |" for x in table)
    (bundle / "MODEL_DEFENSE_NOTES.md").write_text(f'''# Материалы Миши для защиты 30 ноября

## Что выполнено
На общей исходной выборке 400 фотографий запущены Grounding DINO tiny, OWLv2 base и Qwen3-VL-2B-Instruct. DINO и OWLv2 имеют по три порога: 0,15 / 0,25 / 0,40. Основной порог 0,25 выбран заранее. Три фотографии затем заменены: для всех моделей прогнаны новые фото, остальные 397 результатов сохранены.

## Короткий текст выступления
«Я подготовил воспроизводимый запуск трёх моделей и эксперимент с порогом уверенности детекторов. Для каждой фотографии сохранены рамки, настройки и версии моделей, SHA-256 исходного изображения и время выполнения. Qwen возвращает координаты текстом, поэтому его уверенность не сопоставлялась со scores детекторов. Оборванные ответы сохранялись и дообрабатывались отдельно, а не объявлялись пустыми предсказаниями. В итоговом обновлённом наборе у каждой конфигурации ровно 400 одинаковых photo_id. Метрики качества и их доверительные интервалы нужно брать из командного оценивания по проверенной разметке; число найденных объектов само по себе не показывает качество».

## Три блока для слайдов
1. Постановка: найти взрослых бабочек/мотыльков, исключить остальные объекты; три модели, единый набор, одинаковый формат рамок.
2. Эксперимент: три порога у двух детекторов, один генеративный baseline Qwen; аппаратная платформа Apple M3/MPS; сырые ответы и параметры сохранены.
3. Воспроизводимость и ограничения: три замены, отдельная история повторов, исходный и экзаменационный наборы разделены; качество — из проверенной командной разметки.

## Фактическая сводка обновлённого main_v2
| Конфигурация | Число рамок | Фото без рамок | Замеры без warmup/повторов | Медиана, с |
|---|---:|---:|---:|---:|
{summaries}

Время включает препроцессинг, модель, постобработку и синхронизацию; не включает скачивание и загрузку весов. Пороговые варианты детекторов получены из одного инференса и не являются независимыми замерами. Версии PyTorch различались: DINO/OWLv2 2.7, Qwen 2.15; сравнение скорости ограничено этой разницей.

## Повторы Qwen — что говорить честно
Первоначально оборвались пять ответов на лимите 512 токенов. Четыре завершены полнокадровым повтором с другим промптом/лимитом, один — по 16 перекрывающимся фрагментам. Позже две из этих фотографий заменены; в итоговом main_v2 остаются три полнокадровых повтора. Это неоднородная конфигурация: при сравнении качества отметить политику повторов; при сравнении времени исключать их. Ошибки и исходные ответы не удалены. У фрагментного результата нет единого raw_text, исходные ответы сохранены в tiles.

## Ответы на вероятные вопросы
- Почему три порога? Анализ чувствительности precision/recall, основной заранее зафиксированный порог 0,25.
- Почему нет confidence у Qwen? Текстовая генерация координат не даёт сопоставимых confidence; вместо выдуманных значений стоят null.
- Использовались ли правильные ответы? Таксоны и ожидаемый класс в запросы моделям не передавались; рамки не исправлялись вручную.
- Почему модель «лучше»? По этим файлам такой вывод не делается. Нужны командные метрики, одинаковый финальный набор и согласованное сопоставление рамок.
- Где bootstrap/leaderboard? Их считал другой участник. В этом пакете их кода и результатов нет; не представлять сводку количества рамок как leaderboard.

## Что команде ещё объединить
Добавить проверенную gold-разметку именно main_v2, код оценивания/построения графиков, параметры bootstrap, leaderboard и разбор ошибок от Kmdmrg; финальную презентацию, распределение выступлений и ссылку на репозиторий от координатора. Без них весь командный проект не объявляется готовым. Никакие новые вычисления моделей для передачи этого пакета не нужны.
''')
    (bundle / "README.md").write_text('''# Пакет результатов моделей для SHAD. Final project

Готов к передаче Андрею; локальная подготовка, отправка не выполнялась.

## Что открыть сначала
- `project/results/main_v2/` — актуальные 400 фото после трёх замен, семь конфигураций. Именно этот набор брать для текущей оценки.
- `project/results/qwen/` — все первоначальные raw_text, ошибки, полнокадровые повторы, фрагменты и параметры. Даже исходные удалённые из main_v2 фото сохранены как история.
- `project/results/replacements_2026_11_18/` — исходные результаты трёх новых фотографий для всех моделей.
- `project/results/dino/`, `project/results/owlv2/` — полные исходные кандидаты и параметры детекторов.
- `project/scripts/`, `project/prompts/`, `project/requirements*.txt` — код и настройки. Не запускать инференс повторно ради передачи архива.
- `checks/validation.json` и `checks/results_summary.csv` — свежая локальная проверка и сводка.
- `MODEL_DEFENSE_NOTES.md` — описание метода, текст выступления, ограничения и вопросы для защиты.
- `exam_contest_separate/` — ранее подготовленный экзаменационный пакет, отдельно от основного датасета. Не объединять его predictions с main_v2.

## Замены photo_id
615002914 → 765032073; 247404664 → 765036915; 359179108 → 764941215.
Во всех семи итоговых файлах 397 остальных строк совпадают с прежними результатами. `main_v2_sources.jsonl` содержит актуальные 400 ID, URL и хеши исходных изображений. Сами фотографии и веса не дублируются в архиве.

## Что означает raw/logs
Сырые ответы Qwen находятся в JSONL результатов, а не только в папке logs. `*.meta.json` содержит конфигурацию, версии и время загрузки. Сохранены `*.errors.jsonl` с неудачными попытками. `project/logs/` содержит только найденные короткие журналы smoke/debug-проверок; отдельного полного stdout первоначального основного прогона в исходной папке не найдено, он не восстанавливался задним числом. Полные сохранённые журналы экзаменационного прогона лежат в `exam_contest_separate/logs/`.

## Что в пакет не входит
Не включены окружения Python, модельные веса, git-каталог, временные lock-файлы, фотографии и exam_gold. Gold-разметки основного main_v2, кода командного leaderboard/графиков/метрик и bootstrap в найденной папке нет. Их должен добавить ответственный участник; метрики не выдумывались.

`project/MODELS.md` и `project/STATUS.md` сохранены как исходные исторические заметки и описывают исходный main. Актуальные замены и назначение файлов определяются этим README и проверкой в checks.

## Проверка целостности после распаковки
Запустить `python3 verify_package.py` из корня пакета. Проверяются SHA-256 всех файлов из `checksums.json` без запуска моделей и без сети. В журнале `WORK_LOG.txt` описано, что было сделано при подготовке.
''')
    (bundle / "MESSAGE_TO_ANDREY.txt").write_text('''Андрей, собрал полный пакет результатов для репозитория и защиты: исходные результаты трёх моделей, raw_text Qwen, все повторы и ошибки, параметры и код, плюс результаты трёх замен. Актуальные 400 фото после замен — project/results/main_v2. Проверил все семь файлов: одинаковые ID, 397 прежних строк сохранены, три новых результата подставлены правильно. Экзаменационный прогон вынесен отдельно в exam_contest_separate. Заново ничего не запускал. В README есть карта файлов, в MODEL_DEFENSE_NOTES.md — описание метода и ограничения для слайдов. Полного stdout первого основного прогона не нашлось: сохранены JSONL ответов/ошибок и метаданные; для экзамена журналы есть. Код leaderboard/графиков и bootstrap нужно добавить от Kmdmrg.
''')
    (bundle / "verify_package.py").write_text('''import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "checksums.json").read_text())
for name, expected in manifest.items():
    p = root / name
    assert p.is_file(), name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
print(f"OK: {len(manifest)} files match SHA-256; no inference or network access")
''')
    (bundle / "WORK_LOG.txt").write_text(audit["checked_at"] + '''
Read existing results and code. Hashed original/replacement photos locally.
Validated seven final 400-row exports, source IDs, raw data, boxes and timing.
Checked three replacements and 397 unchanged rows in every configuration.
Copied saved artifacts byte-for-byte, excluding transient locks/environments/images.
Created portable source manifest, data summary, defense notes and handoff message draft.
No model inference, paid requests, annotation work, messages, uploads, push or submission.
Missing team metrics/gold and original main stdout explicitly documented.
''')
    # Fail closed if a recognizable credential accidentally appears in copied artifacts.
    secret = re.compile(rb"hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
    for p in bundle.rglob("*"):
        if p.is_file() and secret.search(p.read_bytes()):
            raise RuntimeError("Potential credential found; archive not created: " + str(p.relative_to(bundle)))
    checksums = {str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    write_json(bundle / "checksums.json", checksums)
    archive = dest / "SHAD_project_handoff_2026-11-21.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                z.write(p, str(p.relative_to(dest)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in checksums.items():
            assert hashlib.sha256(z.read("SHAD_project_handoff/" + name)).hexdigest() == digest
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "files": len(checksums) + 1, "sha256": sha(archive),
                      "validated_configurations": len(table), "source_images_hashed": len(image_hashes)}, indent=2))


if __name__ == "__main__":
    main()
