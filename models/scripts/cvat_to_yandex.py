"""
Склейка ручной разметки из CVAT с метаданными датасета и подготовка файлов
для Яндекс Заданий.

Вход:
  - annotations.xml — экспорт из CVAT в формате «CVAT for images 1.1»
  - <split>.jsonl  — метаданные выборки (exam.jsonl или train.jsonl)

Ключ склейки (join) — photo_id: файл в CVAT называется <photo_id>.jpeg,
а в jsonl есть поле photo_id.

Выход (в --out-dir):
  - <split>_gold.jsonl — эталонная разметка: статус и рамки в долях от 0 до 1
  - <split>_tasks.tsv  — файл заданий для пула Яндекс Заданий:
        exam  → INPUT:image + GOLDEN:status               (контрольные задания)
        train → INPUT:image + GOLDEN:status + HINT:text   (обучающие задания)
  Фото с тегом unsure в TSV не попадают — их нужно сначала разобрать командой.

Запуск из корня проекта:
  python scripts/cvat_to_yandex.py --cvat annotations/exam/annotations.xml --split exam
  python scripts/cvat_to_yandex.py --cvat annotations/train/annotations.xml --split train
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

DATASET_DIR = Path("inaturalist_dataset")
BOX_LABEL = "butterfly"
UNSURE_TAG = "unsure"


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clip_to_unit(value: float) -> float:
    """Рамка могла чуть выйти за край картинки — обрезаем до [0, 1]."""
    return min(1.0, max(0.0, value))


def parse_cvat_xml(xml_path: Path) -> dict:
    """
    Читает annotations.xml и возвращает {photo_id: {"boxes": [...], "unsure": bool}}.
    Рамки переводятся из пикселей в доли: left = xtl / ширина и т. д.
    """
    root = ElementTree.parse(xml_path).getroot()
    annotations_by_photo = {}

    for image in root.iter("image"):
        # name может быть с подпапками: "exam/bg_plantae/127638745.jpeg" → 127638745
        photo_id = int(Path(image.get("name")).stem)
        image_width = float(image.get("width"))
        image_height = float(image.get("height"))

        boxes = []
        for box in image.iter("box"):
            if box.get("label") != BOX_LABEL:
                continue
            left = clip_to_unit(float(box.get("xtl")) / image_width)
            top = clip_to_unit(float(box.get("ytl")) / image_height)
            right = clip_to_unit(float(box.get("xbr")) / image_width)
            bottom = clip_to_unit(float(box.get("ybr")) / image_height)
            boxes.append({
                "left": round(left, 6),
                "top": round(top, 6),
                "width": round(right - left, 6),
                "height": round(bottom - top, 6),
            })

        is_unsure = any(tag.get("label") == UNSURE_TAG for tag in image.iter("tag"))
        annotations_by_photo[photo_id] = {"boxes": boxes, "unsure": is_unsure}

    return annotations_by_photo


def make_hint(category: str, status: str) -> str:
    """Подсказка для обучающего задания: показывается, если исполнитель ошибся."""
    if status == "has_butterflies":
        if category.startswith("moth_"):
            return ("На фото мотылёк или моль — их тоже размечаем. Выделите каждую бабочку "
                    "отдельной рамкой и выберите «Есть бабочки (выделены рамками)».")
        return ("На фото есть бабочка. Выделите каждую бабочку отдельной рамкой "
                "и выберите «Есть бабочки (выделены рамками)».")
    if category == "insect_caterpillar":
        return "На фото гусеница или куколка — их не размечаем. Выберите «Бабочек нет»."
    if category.startswith("insect_"):
        return "На фото другое насекомое, а не бабочка. Не рисуйте рамок и выберите «Бабочек нет»."
    return "На фото нет бабочек. Не рисуйте рамок и выберите «Бабочек нет»."


def main():
    parser = argparse.ArgumentParser(description="CVAT → эталон и TSV для Яндекс Заданий")
    parser.add_argument("--cvat", type=Path, required=True, help="путь к annotations.xml из CVAT")
    parser.add_argument("--split", choices=["exam", "train"], required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("annotations"))
    args = parser.parse_args()

    split_rows = read_jsonl(DATASET_DIR / f"{args.split}.jsonl")
    cvat_annotations = parse_cvat_xml(args.cvat)

    # Проверка, что наборы фото совпадают
    split_photo_ids = {row["photo_id"] for row in split_rows}
    missing_in_cvat = split_photo_ids - cvat_annotations.keys()
    extra_in_cvat = cvat_annotations.keys() - split_photo_ids
    if missing_in_cvat:
        print(f"⚠ Нет в CVAT ({len(missing_in_cvat)}): {sorted(missing_in_cvat)}")
    if extra_in_cvat:
        print(f"⚠ Есть в CVAT, но нет в {args.split}.jsonl ({len(extra_in_cvat)}): {sorted(extra_in_cvat)}")

    # Join по photo_id
    gold_rows = []
    for row in split_rows:
        annotation = cvat_annotations.get(row["photo_id"])
        if annotation is None:
            continue
        status = "has_butterflies" if annotation["boxes"] else "no_butterflies"
        gold_rows.append({
            "photo_id": row["photo_id"],
            "url": row["url"],
            "category": row["category"],
            "status": status,
            "boxes": annotation["boxes"],
            "unsure": annotation["unsure"],
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gold_path = args.out_dir / f"{args.split}_gold.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for gold_row in gold_rows:
            f.write(json.dumps(gold_row, ensure_ascii=False) + "\n")

    # TSV для пула: только уверенные ответы
    tasks_path = args.out_dir / f"{args.split}_tasks.tsv"
    header = ["INPUT:image", "GOLDEN:status"]
    if args.split == "train":
        header.append("HINT:text")
    confident_rows = [gold_row for gold_row in gold_rows if not gold_row["unsure"]]
    with open(tasks_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for gold_row in confident_rows:
            tsv_row = [gold_row["url"], gold_row["status"]]
            if args.split == "train":
                tsv_row.append(make_hint(gold_row["category"], gold_row["status"]))
            writer.writerow(tsv_row)

    # Сводка
    with_butterflies = sum(1 for gold_row in gold_rows if gold_row["status"] == "has_butterflies")
    total_boxes = sum(len(gold_row["boxes"]) for gold_row in gold_rows)
    unsure_rows = [gold_row for gold_row in gold_rows if gold_row["unsure"]]
    print(f"\nВыборка {args.split}: {len(gold_rows)} фото")
    print(f"  с бабочками: {with_butterflies} (рамок всего: {total_boxes})")
    print(f"  без бабочек: {len(gold_rows) - with_butterflies}")

    if unsure_rows:
        print(f"\n⚠ Помечены unsure ({len(unsure_rows)}) — разберите командой, в TSV не включены:")
        for gold_row in unsure_rows:
            print(f"  {gold_row['photo_id']}  {gold_row['category']}")

    # Контроль качества: расхождения с ожиданием по таксону iNaturalist
    expected_by_photo = {row["photo_id"]: row.get("expected_positive") for row in split_rows}
    mismatches = [
        gold_row for gold_row in gold_rows
        if expected_by_photo.get(gold_row["photo_id"]) is not None
        and expected_by_photo[gold_row["photo_id"]] != (gold_row["status"] == "has_butterflies")
    ]
    if mismatches:
        print(f"\nРасхождения с категорией iNaturalist ({len(mismatches)}) — стоит перепроверить:")
        for gold_row in mismatches:
            print(f"  {gold_row['photo_id']}  {gold_row['category']} → {gold_row['status']}")

    print(f"\n✓ {gold_path}")
    print(f"✓ {tasks_path} ({len(confident_rows)} заданий)")


if __name__ == "__main__":
    main()
