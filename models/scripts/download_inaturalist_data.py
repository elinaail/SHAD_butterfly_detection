"""
Скрипт для скачивания фото с iNaturalist (CC0) для проекта детекции бабочек.

Положительный класс — все взрослые чешуекрылые (Lepidoptera):
дневные бабочки, мотыльки и моль.

Распределение (~400 фото):
  70% положительный класс ≈ 280
      дневные бабочки ≈ 180 (4 семейства)
      мотыльки и моль ≈ 100 (5 семейств)
  30% негативы ≈ 120
      10% гусеницы и куколки чешуекрылых ≈ 40
      10% другие насекомые (ручейники, стрекозы, жуки) ≈ 40
      10% сложный фон (растения, листья, цветы без насекомых) ≈ 40

Три непересекающиеся выборки с одинаковыми пропорциями:
  main  — 400 фото, основной датасет (→ golden set)
  train — 60 фото, обучающие задания для исполнителей
  exam  — 30 фото, экзамен для допуска исполнителей

Запуск:
  pip install requests tqdm
  python download_inaturalist.py
"""

import requests
import os
import json
import time
import random
import hashlib
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ────────────────────────────────────────────
# Конфигурация
# ────────────────────────────────────────────
OUTPUT_DIR = Path("inaturalist_dataset")
PHOTOS_DIR = OUTPUT_DIR / "photos"
META_FILE  = OUTPUT_DIR / "metadata.jsonl"

API_BASE = "https://api.inaturalist.org/v1"

# Задержка между запросами к API (сек) — не нагружаем сервер
API_DELAY = 1.0

# Максимум фото на один запрос (макс. 200)
PER_PAGE = 200

# Размеры непересекающихся выборок. main = сумма target в CATEGORIES (400).
# train и exam получают те же пропорции категорий, что и main.
SPLIT_SIZES = {"main": 400, "train": 60, "exam": 30}

# Фиксируем seed, чтобы разбиение на выборки воспроизводилось при перезапуске
RANDOM_SEED = 42

# ────────────────────────────────────────────
# Категории и таксоны
# ────────────────────────────────────────────
# Каждая запись: (label, taxon_name, taxon_id, target_count)
# taxon_id=None  →  скрипт найдёт ID по имени через API
CATEGORIES = [
    # ===== ПОЛОЖИТЕЛЬНЫЙ КЛАСС (70%) — взрослые чешуекрылые =====
    # Дневные бабочки (~45%)
    ("butterfly_nymphalidae",  "Nymphalidae",   47922, 45),   # нимфалиды
    ("butterfly_pieridae",     "Pieridae",       None,  45),   # белянки
    ("butterfly_lycaenidae",   "Lycaenidae",     None,  45),   # голубянки
    ("butterfly_papilionidae", "Papilionidae",   None,  45),   # парусники

    # Мотыльки и моль (~25%)
    ("moth_geometridae",  "Geometridae",  None, 25),   # пяденицы
    ("moth_noctuidae",    "Noctuidae",    None, 25),   # совки
    ("moth_erebidae",     "Erebidae",     None, 25),   # эребиды (медведицы, ленточницы)
    ("moth_sphingidae",   "Sphingidae",   None, 15),   # бражники
    ("moth_crambidae",    "Crambidae",    None, 10),   # огнёвки-травянки (мелкая «моль»)

    # ===== НЕГАТИВЫ (30%) =====
    # Гусеницы и куколки чешуекрылых (10%) — не размечаются
    ("insect_caterpillar", "Lepidoptera", 47157, 40),

    # Другие насекомые (10%)
    ("insect_caddisfly",   "Trichoptera",  None, 15),  # ручейники — очень похожи на моль
    ("insect_dragonfly",   "Odonata",     47792, 15),  # стрекозы
    ("insect_beetle",      "Coleoptera",   None, 10),  # жуки

    # Сложный фон (10%)
    ("bg_plantae",  "Plantae",  47126, 40),  # растения (листья, цветы)
]

# Префиксы категорий положительного класса (взрослые чешуекрылые)
POSITIVE_PREFIXES = ("butterfly_", "moth_")

# ────────────────────────────────────────────
# Вспомогательные функции
# ────────────────────────────────────────────

def resolve_taxon_id(name: str) -> int:
    """Находит taxon_id по названию через API."""
    resp = requests.get(f"{API_BASE}/taxa", params={"q": name, "per_page": 5})
    resp.raise_for_status()
    results = resp.json().get("results", [])
    # Ищем точное совпадение по имени
    for t in results:
        if t.get("name", "").lower() == name.lower():
            print(f"  ✓ {name} → taxon_id={t['id']}")
            return t["id"]
    # Если точного нет — берём первый результат
    if results:
        tid = results[0]["id"]
        print(f"  ~ {name} → taxon_id={tid} (первый результат: {results[0].get('name')})")
        return tid
    raise ValueError(f"Таксон '{name}' не найден!")


def get_s3_url(photo: dict) -> str | None:
    """Строит прямую ссылку на S3 для фото."""
    url = photo.get("url", "")
    if not url:
        return None
    # Заменяем размер на original
    for size in ["square", "small", "medium", "large"]:
        url = url.replace(f"/{size}.", "/original.")
    # Преобразуем в S3 ссылку, если это стандартный URL iNaturalist
    if "inaturalist-open-data.s3.amazonaws.com" in url:
        return url
    if "static.inaturalist.org" in url:
        # https://static.inaturalist.org/photos/PHOTO_ID/original.jpeg
        # → https://inaturalist-open-data.s3.amazonaws.com/photos/PHOTO_ID/original.jpeg
        url = url.replace(
            "static.inaturalist.org",
            "inaturalist-open-data.s3.amazonaws.com"
        )
        return url
    return url


def fetch_observations(taxon_id: int, target: int, extra_params: dict = None) -> list:
    """
    Получает наблюдения из iNaturalist API.
    Возвращает список словарей с метаданными фото.
    """
    params = {
        "taxon_id": taxon_id,
        "photos": "true",
        "photo_license": "cc0",
        "quality_grade": "research",
        "per_page": min(PER_PAGE, target * 2),  # берём с запасом
        "order": "desc",
        "order_by": "votes",
    }
    if extra_params:
        params.update(extra_params)

    all_photos = []
    seen_photo_ids = set()
    page = 1

    while len(all_photos) < target and page <= 5:
        params["page"] = page
        try:
            resp = requests.get(f"{API_BASE}/observations", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    Ошибка API (стр. {page}): {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            if len(all_photos) >= target:
                break
            photos = obs.get("photos", [])
            # В одном наблюдении фото могут быть с разными лицензиями —
            # берём первое фото, у которого лицензия именно CC0
            cc0_photos = [p for p in photos if (p.get("license_code") or "").lower() == "cc0"]
            if not cc0_photos:
                continue
            photo = cc0_photos[0]
            photo_id = photo.get("id")
            if photo_id in seen_photo_ids:
                continue
            seen_photo_ids.add(photo_id)

            s3_url = get_s3_url(photo)
            if not s3_url:
                continue

            # Определяем расширение
            ext = "jpeg"
            if ".png" in s3_url:
                ext = "png"
            elif ".jpg" in s3_url:
                ext = "jpg"

            all_photos.append({
                "observation_id": obs.get("id"),
                "photo_id": photo_id,
                "url": s3_url,
                "ext": ext,
                "taxon_name": obs.get("taxon", {}).get("name", ""),
                "taxon_id": taxon_id,
                "license": photo.get("license_code"),
                "observation_license": obs.get("license_code"),
                "observation_url": f"https://www.inaturalist.org/observations/{obs.get('id')}",
                "attribution": photo.get("attribution", ""),
            })

        page += 1
        time.sleep(API_DELAY)

    return all_photos


def allocate_split_counts() -> dict:
    """
    Считает, сколько фото каждой категории пойдёт в каждую выборку,
    сохраняя пропорции основного датасета.

    Дробные доли округляются методом наибольшего остатка (largest remainder):
    сначала берём целую часть, а оставшиеся места отдаём категориям
    с наибольшей дробной частью — так сумма по выборке получается ровно SPLIT_SIZES[split].
    При равном остатке приоритет у маленьких категорий, чтобы моли/насекомые
    не потерялись при округлении.

    Возвращает {label: {"main": n, "train": n, "exam": n}}.
    """
    main_total = sum(target for _, _, _, target in CATEGORIES)
    counts = {label: {} for label, _, _, _ in CATEGORIES}

    for split_name, split_size in SPLIT_SIZES.items():
        exact_shares = {
            label: target * split_size / main_total
            for label, _, _, target in CATEGORIES
        }
        rounded_shares = {label: int(share) for label, share in exact_shares.items()}
        places_left = split_size - sum(rounded_shares.values())

        category_targets = {label: target for label, _, _, target in CATEGORIES}
        labels_by_remainder = sorted(
            exact_shares,
            key=lambda label: (exact_shares[label] - rounded_shares[label], -category_targets[label]),
            reverse=True,
        )
        for label in labels_by_remainder[:places_left]:
            rounded_shares[label] += 1

        for label, count in rounded_shares.items():
            counts[label][split_name] = count

    return counts


def download_photo(photo_meta: dict, category: str) -> dict | None:
    """Скачивает одно фото и возвращает обновлённые метаданные."""
    cat_dir = PHOTOS_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{photo_meta['photo_id']}.{photo_meta['ext']}"
    filepath = cat_dir / filename

    if filepath.exists():
        photo_meta["local_path"] = str(filepath)
        return photo_meta

    try:
        resp = requests.get(photo_meta["url"], timeout=60)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        photo_meta["local_path"] = str(filepath)
        return photo_meta
    except Exception as e:
        print(f"    ✗ Не удалось скачать {photo_meta['photo_id']}: {e}")
        return None


# ────────────────────────────────────────────
# Основная логика
# ────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PHOTOS_DIR.mkdir(exist_ok=True)

    global_seen_ids = set()  # для дедупликации по photo_id
    all_metadata = []

    split_counts = allocate_split_counts()
    rng = random.Random(RANDOM_SEED)

    print("=" * 60)
    print("Скачивание датасета iNaturalist (CC0)")
    print("=" * 60)

    for label, taxon_name, taxon_id, target in CATEGORIES:
        total_needed = sum(split_counts[label].values())
        print(f"\n{'─' * 50}")
        print(f"Категория: {label} ({taxon_name}), нужно: {total_needed} фото {split_counts[label]}")

        # 1. Резолвим taxon_id если не задан
        if taxon_id is None:
            print(f"  Ищу taxon_id для '{taxon_name}'...")
            try:
                taxon_id = resolve_taxon_id(taxon_name)
            except ValueError as e:
                print(f"  ✗ {e}")
                continue
            time.sleep(API_DELAY)
        else:
            print(f"  taxon_id = {taxon_id}")

        # 2. Доп. параметры для особых категорий
        # Коды стадий iNaturalist: term_id=1 (Life Stage), 2=Adult, 4=Pupa, 6=Larva
        extra = {}
        if label == "insect_caterpillar":
            # Гусеницы и куколки Lepidoptera
            extra["term_id"] = 1
            extra["term_value_id"] = "4,6"
            print("  (фильтр: стадия = гусеница или куколка)")
        elif label.startswith(("butterfly_", "moth_")):
            # Только взрослые, чтобы в «бабочки» не попали гусеницы
            extra["term_id"] = 1
            extra["term_value_id"] = 2
            print("  (фильтр: стадия = взрослая особь)")

        # 3. Собираем наблюдения
        print(f"  Запрашиваю наблюдения...")
        photos = fetch_observations(taxon_id, total_needed, extra)
        print(f"  Найдено: {len(photos)} фото")

        # 4. Дедупликация (глобальная — поэтому выборки не пересекаются)
        unique = []
        for p in photos:
            if p["photo_id"] not in global_seen_ids:
                global_seen_ids.add(p["photo_id"])
                unique.append(p)
        print(f"  После дедупликации: {len(unique)}")
        if len(unique) < total_needed:
            print(f"  ⚠ Не хватает фото: нужно {total_needed}, есть {len(unique)}")

        # 5. Перемешиваем и режем на непересекающиеся куски main / train / exam
        rng.shuffle(unique)
        photos_with_split = []
        start_index = 0
        for split_name in SPLIT_SIZES:
            split_count = split_counts[label][split_name]
            for photo_meta in unique[start_index:start_index + split_count]:
                photos_with_split.append((photo_meta, split_name))
            start_index += split_count

        # 6. Скачиваем в photos/<split>/<category>/
        print(f"  Скачиваю...")
        downloaded = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(download_photo, photo_meta, f"{split_name}/{label}"): split_name
                for photo_meta, split_name in photos_with_split
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {label}"):
                result = future.result()
                if result:
                    result["category"] = label
                    result["split"] = futures[future]
                    # Ожидание по таксону iNaturalist — НЕ разметка, только подсказка для контроля качества
                    result["expected_positive"] = label.startswith(POSITIVE_PREFIXES)
                    downloaded.append(result)

        all_metadata.extend(downloaded)
        print(f"  ✓ Скачано: {len(downloaded)} фото")

    # 7. Сохраняем метаданные: общий файл + отдельный на каждую выборку
    print(f"\n{'=' * 60}")
    print(f"Сохраняю метаданные в {META_FILE}...")
    with open(META_FILE, "w", encoding="utf-8") as f:
        for m in all_metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    for split_name in SPLIT_SIZES:
        split_file = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for m in all_metadata:
                if m["split"] == split_name:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"  {split_file}")

    # 8. Статистика по каждой выборке
    from collections import Counter
    for split_name in SPLIT_SIZES:
        split_metadata = [m for m in all_metadata if m["split"] == split_name]
        cat_counts = Counter(m["category"] for m in split_metadata)
        total = sum(cat_counts.values())

        print(f"\n{'=' * 60}")
        print(f"ВЫБОРКА {split_name.upper()} (цель {SPLIT_SIZES[split_name]}):")
        for cat, count in sorted(cat_counts.items()):
            print(f"  {cat}: {count}")
        print(f"  ───────────────────")
        print(f"  Всего: {total}")

        if total:
            butterflies = sum(c for k, c in cat_counts.items() if k.startswith("butterfly_"))
            moths = sum(c for k, c in cat_counts.items() if k.startswith("moth_"))
            positives = sum(c for k, c in cat_counts.items() if k.startswith(POSITIVE_PREFIXES))
            caterpillars = cat_counts.get("insect_caterpillar", 0)
            other_insects = sum(
                c for k, c in cat_counts.items()
                if k.startswith("insect_") and k != "insect_caterpillar"
            )
            bg = sum(c for k, c in cat_counts.items() if k.startswith("bg_"))
            print(f"\n  Положительный класс: {positives} ({positives/total*100:.0f}%)")
            print(f"    дневные бабочки:   {butterflies}")
            print(f"    мотыльки и моль:   {moths}")
            print(f"  Гусеницы/куколки:    {caterpillars} ({caterpillars/total*100:.0f}%)")
            print(f"  Другие насекомые:    {other_insects} ({other_insects/total*100:.0f}%)")
            print(f"  Фон:                 {bg} ({bg/total*100:.0f}%)")

    # Проверка, что выборки не пересекаются
    split_photo_ids = {
        split_name: {m["photo_id"] for m in all_metadata if m["split"] == split_name}
        for split_name in SPLIT_SIZES
    }
    overlap_main_train = split_photo_ids["main"] & split_photo_ids["train"]
    overlap_main_exam = split_photo_ids["main"] & split_photo_ids["exam"]
    overlap_train_exam = split_photo_ids["train"] & split_photo_ids["exam"]
    if overlap_main_train or overlap_main_exam or overlap_train_exam:
        print("\n⚠ Найдены пересечения между выборками!")
    else:
        print("\n✓ Выборки не пересекаются")

    print(f"\nФото: {PHOTOS_DIR.resolve()}")
    print(f"Метаданные: {META_FILE.resolve()}")
    print("Готово!")


if __name__ == "__main__":
    main()
