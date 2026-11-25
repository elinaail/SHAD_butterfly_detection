"""Примеры ошибок моделей: eval/figures/examples/*.png. ID отобраны вручную по results/per_photo.csv."""
from PIL import Image, ImageDraw

from viz_common import (BG, FIG, FP_COLOR, INK2, MODEL_COLOR, MODEL_NAME, MODELS, ROOT, compare_panels,
                        draw_box, draw_label, font, grid, legend, load_all, load_old_preds, open_image, save,
                        text_bar, title_bar, vstack)

OUT = FIG / "examples"
meta, gold, preds = load_all()

CATEGORY = {
    "butterfly_nymphalidae": "нимфалида", "butterfly_pieridae": "белянка", "butterfly_lycaenidae": "голубянка",
    "butterfly_papilionidae": "парусник", "moth_geometridae": "пяденица", "moth_noctuidae": "совка",
    "moth_erebidae": "эребида", "moth_sphingidae": "бражник", "moth_crambidae": "огневка",
    "insect_caterpillar": "гусеница", "bg_plantae": "растение", "insect_dragonfly": "стрекоза",
    "insect_caddisfly": "ручейник", "insect_beetle": "жук",
}


def photo_label(pid, note=""):
    m = meta[pid]
    label = f"{pid}, {CATEGORY[m['category']]} ({m['taxon_name']})"
    return f"{label}, {note}" if note else label


def example(name, title, subtitle, photos, footnote=None, panel_h=430):
    cols = 2 if len(photos) == 1 else 4  # одно фото раскладываем 2x2, чтобы картинка была ближе к 16:9
    rows = [(photo_label(pid, note), grid(compare_panels(pid, gold, preds, panel_h), cols)) for pid, note in photos]
    width = max(g.width for _, g in rows)
    parts = [title_bar(width, title, subtitle)]
    for label, g in rows:
        parts += [text_bar(width, label), g]
    parts.append(legend(width))
    if footnote:
        parts.append(text_bar(width, footnote, size=18, bold=False, color=INK2))
    save(vstack(parts), OUT / name)
    print(name)


def dino_empty_grid():
    photo_ids = [6398640, 11273123, 132284283, 224385644, 155762126, 22838470,
                 13940133, 68167772, 152197051, 32279278, 103939671, 137831879]
    tw, th, cols, gap = 520, 390, 6, 10
    cells = []
    for pid in photo_ids:
        r = preds["dino_t0.25"][pid]
        im = open_image(pid)
        im.thumbnail((tw, th), Image.LANCZOS)
        cell = Image.new("RGB", (tw, th + 38), BG)
        ox, oy = (tw - im.width) // 2, (th - im.height) // 2
        framed = im.copy()
        d = ImageDraw.Draw(framed, "RGBA")
        for b, s in zip(r["boxes"], r["scores"]):
            x, y = draw_box(d, b, *framed.size, FP_COLOR, 4)
            draw_label(d, x, max(26, y), f"FP {s:.2f}".replace(".", ","), FP_COLOR, size=16)
        cell.paste(framed, (ox, oy))
        ImageDraw.Draw(cell).text((10, th + 7), f"{CATEGORY[meta[pid]['category']]}, {pid}", fill=INK2, font=font(18))
        cells.append(cell)
    body = Image.new("RGB", (cols * tw + (cols + 1) * gap, 2 * (th + 38) + 3 * gap), BG)
    for k, c in enumerate(cells):
        body.paste(c, (gap + (k % cols) * (tw + gap), gap + (k // cols) * (th + 38 + gap)))
    w = body.width
    save(vstack([
        title_bar(w, "Grounding DINO почти всегда находит бабочку, даже если ее нет",
                  "Рамки DINO (порог 0,25) на кадрах без бабочек, где OWLv2 и Qwen3-VL не ошиблись. "
                  "Число на метке: score рамки"),
        body,
        text_bar(w, "У DINO ложная рамка на 119 из 122 пустых кадров (171 рамка). Score таких рамок 0,3-0,65, "
                    "как у верных, поэтому порог не помогает (график 08).", size=18, bold=False, color=INK2),
    ]), OUT / "01_dino_fp_on_empty_grid")
    print("01_dino_fp_on_empty_grid")


def removed_colonies():
    photos = [(615002914, "err_photo_2026-11-18_18-44-21", "колония монархов", 765032073),
              (247404664, "err_photo_2026-11-18_18-47-11", "скопление желтушек", 765036915)]
    old = load_old_preds({p for p, *_ in photos})
    h = 420
    parts = []
    for pid, fname, what, new_pid in photos:
        im = open_image(pid, 1400, ROOT / f"{fname}.jpg")
        im = im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
        panels = [(im, "Исходное фото", "#ffffff", f"заменено на {new_pid}")]
        for m in MODELS:
            boxes = old[m][pid]["boxes"]
            framed = im.copy()
            d = ImageDraw.Draw(framed, "RGBA")
            for b in boxes:
                draw_box(d, b, *framed.size, MODEL_COLOR[m], 3)
            panels.append((framed, MODEL_NAME[m], MODEL_COLOR[m], f"рамок: {len(boxes)}"))
        g = grid(panels, 4)
        parts += [(f"{pid}, {what}, {fname}.jpg", g)]
    w = max(g.width for _, g in parts)
    out = [title_bar(w, "Почему два фото err_ убраны из оценки",
                     "Десятки и сотни перекрытых особей: отдельные рамки нельзя надежно проверить")]
    for label, g in parts:
        out += [text_bar(w, label), g]
    out.append(text_bar(w, "Показаны предсказания до замены фото. В метриках эти фото не участвуют.",
                        size=18, bold=False, color=INK2))
    save(vstack(out), OUT / "12_removed_dense_colonies")
    print("12_removed_dense_colonies")


dino_empty_grid()

example("02_drawings_rule_level",
        "Рисунок бабочки: FP у всех трех моделей",
        "По инструкции изображения бабочек не размечаются",
        [(336529250, "рисунок"), (357852705, "рисунок")], panel_h=380)

example("03_caterpillar_pupa",
        "Гусеница и куколка: все модели ставят рамку",
        "Невзрослые стадии не размечаются. FP на кадрах с гусеницами: DINO 100%, OWLv2 60%, Qwen 15%",
        [(46284, ""), (154589781, "куколка")], panel_h=380)

example("04_caddisflies",
        "Ручейники похожи на молей, здесь ошибаются все",
        "Ручейники исключены из целевого класса, даже если похожи на моль",
        [(77451033, ""), (87757962, "")], panel_h=380)

example("05_group_err_146688328",
        "Скопление парусников: одна рамка на всю группу",
        "Фото err_: в gold 7 особей. DINO и Qwen обводят группу целиком (0 TP), OWLv2 находит 3 из 7",
        [(146688328, "err_photo_2026-11-18_18-47-41")],
        "На 11 фото с 4+ бабочками recall 63-81% (график 09).", panel_h=470)

example("06_small_far",
        "Мелкие бабочки: DINO пропускает 6 из 8",
        "Бабочки занимают меньше 1% кадра. OWLv2 находит все, Qwen 7 из 8",
        [(76337845, "")],
        "Для Qwen это фото дообработано повторным запуском (1024 px, 4096 токенов), см. completion_method.",
        panel_h=520)

example("07_owl_loose_boxes",
        "OWLv2 находит бабочку, но срезает усики и края крыльев",
        "При IoU 0,5 это TP, при IoU 0,75 уже FP и FN. F1 OWLv2: 0,81 при IoU 0,5 и 0,62 при IoU 0,75",
        [(35524502, ""), (50219608, "")], panel_h=400)

example("08_dino_other_insects",
        "Рядом с бабочкой DINO размечает и других насекомых",
        "Мухи рядом с двумя бабочками: у DINO 10 лишних рамок, у Qwen ни одной",
        [(14825634, "")], panel_h=470)

example("09_owl_fragments",
        "OWLv2 дробит одну бабочку на части",
        "Пять рамок на крылья и тело одного бражника: 5 FP и пропуск самой бабочки",
        [(362693400, "")], panel_h=470)

example("10_pupa_next_to_adult",
        "Куколка рядом со взрослой бабочкой",
        "OWLv2 и Qwen ставят рамку на куколку, бабочку находят все",
        [(21908014, "")], panel_h=470)

example("11_gold_suspect_caddisfly",
        "Исправление gold: ручейники исключены из целевого класса",
        "В gold v3 оба кадра сохранены с пустыми рамками",
        [(7775492, ""), (16391324, "")],
        "Исправлены 3 рамки, все 400 фото сохранены. Qwen остаётся лидером; изменения F1 меньше 0,004.", panel_h=380)

removed_colonies()
