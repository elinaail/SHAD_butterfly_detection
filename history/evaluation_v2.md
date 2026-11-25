# eval: оценка моделей

## Запуск

Из корня `hw_5`:

```bash
uv venv eval/.venv --python 3.12
uv pip install --python eval/.venv/bin/python "numpy<2.3" scipy pandas pillow plotly kaleido requests

eval/.venv/bin/python eval/tests/test_metrics.py
eval/.venv/bin/python eval/scripts/merge_replacements.py
eval/.venv/bin/python eval/scripts/evaluate.py
eval/.venv/bin/python -I eval/scripts/download_images.py
eval/.venv/bin/python eval/scripts/make_figures.py
cd eval/scripts && ../.venv/bin/python make_gallery.py
```

| Скрипт | Что делает |
|---|---|
| `merge_replacements.py` | собирает `model_results_400/` из `model_results/` и `model_replacements_results/` |
| `evaluate.py` | считает метрики, пишет `results/*.csv` и `results/summary.json` |
| `download_images.py` | скачивает 400 фото (large, 1024 px) в `cache_images/`, нужны только для картинок |
| `make_figures.py` | графики в `figures/charts/*.png` |
| `make_gallery.py` | примеры ошибок в `figures/examples/*.png` |

`.venv/` и `cache_images/` в `.gitignore`.

## Данные

- Gold: `gold_400_for_metrics/gold_400_v2.jsonl`, 400 фото, 364 рамки, 120 фото без бабочек.
- Предсказания: 7 конфигураций (DINO и OWLv2 с порогами 0.15, 0.25, 0.40 и Qwen3-VL-2B), у всех один и тот же набор из 400 фото.
- Три фото датасета заменены (615002914 -> 765032073, 247404664 -> 765036915, 359179108 -> 764941215). Прогон на новых фото лежит в `model_replacements_results/`, код в `model_code/scripts/run_replacements.py`. В `model_results_400/` 397 строк взяты из `model_results/` без изменений, 3 строки заменены.
- sha256 из `model_code/results/replacements_2026_11_18/validation.json` с файлами в `model_results_400/` не совпадает: там хешируются полные файлы с `raw_text`. `merge_replacements.py` проверяет состав ID, координаты и отсутствие `error`.
- Qwen: фото 12297798, 76337845, 339831709 обработаны повторным запуском (поле `completion_method`, колонка `reprocessed_photos` в лидерборде).
- Время на фото считается без `warmup` и без фото с `completion_method`.

## Метрика

`metrics.py`. Внутри фото ищется максимальное паросочетание пар (gold, предсказание) с IoU >= порога, при равном числе пар берется большая сумма IoU. TP, FP, FN суммируются по всем фото, F1 = 2TP / (2TP + FP + FN). Доверительные интервалы: бутстреп по фото, 2000 повторов.

Типы ошибок при IoU 0.5:

| Тип | Условие |
|---|---|
| `fp_empty_frame` | рамка на фото без бабочек |
| `fp_duplicate` | IoU >= 0.5 с бабочкой, которая уже сопоставлена |
| `fp_group_box` | рамка накрывает 2 и больше бабочек (каждая внутри на 70% и больше) |
| `fp_part_box` | рамка на 80% и больше внутри одной бабочки, IoU < 0.35 |
| `fp_loose` | IoU с бабочкой от 0.1 до 0.5 |
| `fp_background` | IoU со всеми бабочками < 0.1 |
| `fn_in_group_box` | бабочка внутри рамки типа `fp_group_box` |
| `fn_loose` | есть рамка с IoU от 0.1 до 0.5 |
| `fn_missed` | нет рамки с IoU >= 0.1 |

## Результаты (400 фото)

| Конфигурация | P@0.5 | R@0.5 | F1@0.5 [95% ДИ] | P@0.75 | R@0.75 | F1@0.75 | Пустые фото с FP | FP на пустых | с/фото |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen3-VL-2B | 0.883 | 0.929 | 0.905 [0.875-0.933] | 0.786 | 0.827 | 0.806 | 22 / 120 | 24 | 3.97 |
| OWLv2, t=0.40 | 0.855 | 0.808 | 0.831 | 0.674 | 0.637 | 0.655 | 26 / 120 | 27 | 2.48 |
| OWLv2, t=0.25 | 0.735 | 0.901 | 0.810 [0.770-0.845] | 0.565 | 0.692 | 0.622 | 52 / 120 | 57 | 2.48 |
| DINO, t=0.40 | 0.746 | 0.830 | 0.785 | 0.691 | 0.769 | 0.728 | 91 / 120 | 92 | 3.37 |
| DINO, t=0.25 | 0.618 | 0.901 | 0.733 [0.687-0.775] | 0.565 | 0.824 | 0.670 | 117 / 120 | 167 | 3.37 |
| OWLv2, t=0.15 | 0.560 | 0.926 | 0.698 | 0.429 | 0.709 | 0.534 | 70 / 120 | 109 | 2.48 |
| DINO, t=0.15 | 0.468 | 0.931 | 0.623 | 0.423 | 0.841 | 0.562 | 120 / 120 | 304 | 3.37 |

Время: медиана, MacBook Air M3, MPS. Пороговые варианты одной модели получены из одного запуска.

Лучший F1@0.5 по кривой порога (`threshold_curve.csv`): DINO 0.789 при 0.36, OWLv2 0.838 при 0.30. Порог подбирался на тех же 400 фото.

Recall@0.5 по группам (DINO / OWLv2 / Qwen): дневные бабочки 0.87 / 0.90 / 0.91, ночные 0.98 / 0.94 / 0.97, фото с 4+ бабочками 0.63 / 0.81 / 0.74.

Доля совпавших при IoU 0.5 пар, у которых IoU >= 0.75: DINO 0.92, OWLv2 0.77, Qwen 0.89.

### По источнику gold (`gold_400_provenance.json`)

| Источник | Фото | Рамок | Пустых | F1@0.5 DINO / OWLv2 / Qwen | F1@0.75 DINO / OWLv2 / Qwen |
|---|--:|--:|--:|--:|--:|
| crowd | 278 | 192 | 104 | 0.70 / 0.81 / 0.95 | 0.66 / 0.66 / 0.91 |
| captain_review | 118 | 150 | 16 | 0.81 / 0.83 / 0.88 | 0.72 / 0.60 / 0.73 |
| Meow (ручной разбор) | 4 | 22 | 0 | 0.56 / 0.67 / 0.61 | 0.38 / 0.46 / 0.33 |

captain_review: рамки крауда, исправленные агентом (поле `reason`). Meow: 116540226, 136273568, 146688328, 390616794, совпадают с `_archive/review_results/`.

### Чувствительность

- Ручейники 7775492 и 16391324 размечены в gold как бабочки, остальные 13 фото ручейников пустые. Если считать эти два фото пустыми, F1 любой модели меняется не больше чем на 0.003.
- Без 15 фото ручейников F1@0.5: Qwen 0.922, OWLv2 0.829, DINO 0.751.

## Файлы

| Файл | Содержимое |
|---|---|
| `results/leaderboard.csv` | метрики всех конфигураций |
| `results/by_group.csv`, `by_category.csv`, `by_count.csv`, `by_gold_source.csv` | метрики по группам фото |
| `results/recall_by_size.csv` | recall по площади рамки gold |
| `results/fp_on_empty.csv` | FP на фото без бабочек по категориям |
| `results/error_types.csv` | число ошибок каждого типа |
| `results/threshold_curve.csv` | P, R, F1 в зависимости от порога для DINO и OWLv2 |
| `results/matched_iou.csv` | IoU всех пар TP при пороге 0.5 |
| `results/per_photo.csv` | TP, FP, FN и типы ошибок по каждому фото и конфигурации |
| `figures/charts/` | графики 01-10 |
| `figures/examples/` | примеры ошибок 01-12, описание в `errors_for_slides.md` |
