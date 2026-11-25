"""Загрузка данных и отрисовка фото с рамками для галереи."""
import json
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import classify_photo

IMG_DIR = ROOT / "eval" / "cache_images"
FIG = ROOT / "eval" / "figures"

MODELS = ["dino_t0.25", "owlv2_t0.25", "qwen"]
MODEL_COLOR = {"dino_t0.25": "#2a78d6", "owlv2_t0.25": "#eb6834", "qwen": "#1baf7a"}
MODEL_NAME = {"dino_t0.25": "Grounding DINO", "owlv2_t0.25": "OWLv2", "qwen": "Qwen3-VL-2B"}
GOLD_COLOR = "#ffffff"
FP_COLOR = "#d03b3b"
FN_COLOR = "#fab219"
INK = "#0b0b0b"
INK2 = "#52514e"
BG = "#fcfcfb"
HEADER_H = 66

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size, bold=False):
    candidates = [
        FONTS[bold],
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_all():
    meta = {r["photo_id"]: r for r in json.load(open(ROOT / "gold_400_for_metrics" / "main_metadata_v2.json"))}
    gold = {r["photo_id"]: r["boxes"] for r in load_jsonl(ROOT / "gold_400_for_metrics" / "gold_400_v3.jsonl")}
    preds = {m: {r["photo_id"]: r for r in load_jsonl(ROOT / "model_results_400" / f"{m}.jsonl")} for m in MODELS}
    return meta, gold, preds


def load_old_preds(photo_ids):
    """Предсказания до замены фото (только для иллюстрации, в метриках не участвуют)."""
    out = {}
    for m in MODELS:
        rows = load_jsonl(ROOT / "model_results" / f"{m}.jsonl")
        out[m] = {r["photo_id"]: r for r in rows if r["photo_id"] in photo_ids}
    return out


def open_image(pid, max_side=1024, path=None):
    im = Image.open(path or IMG_DIR / f"{pid}.jpg").convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def draw_box(draw, b, w, h, color, width, dashed=False):
    x0, y0 = max(0, b["left"] * w), max(0, b["top"] * h)
    x1, y1 = min(w - 1, (b["left"] + b["width"]) * w), min(h - 1, (b["top"] + b["height"]) * h)
    # темная подложка, чтобы рамку было видно на светлом фоне
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0, 150), width=width + 4)
    if not dashed:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        return x0, y0
    step = 14
    for ax, ay, bx, by in [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        n = max(int(max(abs(bx - ax), abs(by - ay)) // step), 1)
        for k in range(0, n, 2):
            t0, t1 = k / n, min((k + 1) / n, 1)
            draw.line([ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1],
                      fill=color, width=width)
    return x0, y0


def draw_label(draw, x, y, text, bg, fg="#ffffff", size=18):
    f = font(size, bold=True)
    tw, th = draw.textbbox((0, 0), text, font=f)[2:]
    y = max(0, y - th - 8)
    draw.rounded_rectangle([x, y, x + tw + 10, y + th + 8], radius=4, fill=bg)
    draw.text((x + 5, y + 3), text, fill=fg, font=f)


def draw_gold(im, gt, line=4):
    im = im.copy()
    draw = ImageDraw.Draw(im, "RGBA")
    for b in gt:
        draw_box(draw, b, *im.size, GOLD_COLOR, line)
    return im


def draw_pred(im, gt, pr, color, line=4):
    """Рамки модели: TP цветом модели с IoU, FP красным, пропущенные рамки gold желтым пунктиром."""
    im = im.copy()
    w, h = im.size
    draw = ImageDraw.Draw(im, "RGBA")
    pairs, iou, fp, fn = classify_photo(gt, pr, 0.5)
    for i in fn:
        draw_label(draw, *draw_box(draw, gt[i], w, h, FN_COLOR, line, dashed=True), "FN", FN_COLOR, INK)
    for j in fp:
        draw_label(draw, *draw_box(draw, pr[j], w, h, FP_COLOR, line), "FP", FP_COLOR)
    for g, p in pairs:
        x, y = draw_box(draw, pr[p], w, h, color, line)
        draw_label(draw, x, y, f"IoU {iou[g, p]:.2f}".replace(".", ","), color, size=16)
    return im


def text_bar(width, text, size=19, bold=True, color=INK, height=40):
    bar = Image.new("RGB", (width, height), BG)
    ImageDraw.Draw(bar).text((14, 8), text, fill=color, font=font(size, bold))
    return bar


def title_bar(width, title, subtitle):
    k = max(1.0, width / 2000)  # на широких картинках шрифт крупнее
    bar = Image.new("RGB", (width, int(104 * k)), BG)
    d = ImageDraw.Draw(bar)
    d.text((14, int(10 * k)), title, fill=INK, font=font(int(30 * k), bold=True))
    d.text((14, int(56 * k)), subtitle, fill=INK2, font=font(int(21 * k)))
    return bar


def panel_header(width, name, swatch, stats):
    bar = Image.new("RGB", (width, HEADER_H), BG)
    d = ImageDraw.Draw(bar)
    d.rounded_rectangle([10, 9, 26, 25], radius=3, fill=swatch, outline="#00000040")
    size = 21
    while size > 13 and d.textbbox((0, 0), name, font=font(size, True))[2] > width - 40:
        size -= 1
    d.text((34, 5), name, fill=INK, font=font(size, bold=True))
    d.text((10, 36), stats, fill=INK2, font=font(17))
    return bar


def grid(panels, cols, gap=10):
    """panels: список (картинка, название, цвет, подпись) одинакового размера."""
    pw, ph = panels[0][0].size
    nrows = -(-len(panels) // cols)
    cell_h = ph + HEADER_H + gap
    out = Image.new("RGB", (cols * pw + (cols + 1) * gap, nrows * cell_h), BG)
    for k, (im, name, swatch, stats) in enumerate(panels):
        x, y = gap + (k % cols) * (pw + gap), (k // cols) * cell_h
        out.paste(panel_header(pw, name, swatch, stats), (x, y))
        out.paste(im, (x, y + HEADER_H))
    return out


def compare_panels(pid, gold, preds, panel_h=420):
    """Эталон и три модели на одном фото."""
    gt = gold[pid]
    im = open_image(pid, 1280)
    im = im.resize((max(1, int(im.width * panel_h / im.height)), panel_h), Image.LANCZOS)
    line = max(3, panel_h // 130)
    panels = [(draw_gold(im, gt, line), "Эталон", GOLD_COLOR, f"{len(gt)} бабоч." if gt else "пусто")]
    for m in MODELS:
        pr = preds[m][pid]["boxes"]
        pairs, _, fp, fn = classify_photo(gt, pr, 0.5)
        panels.append((draw_pred(im, gt, pr, MODEL_COLOR[m], line), MODEL_NAME[m], MODEL_COLOR[m],
                       f"TP {len(pairs)}, FP {len(fp)}, FN {len(fn)}"))
    return panels


def legend(width):
    bar = Image.new("RGB", (width, 40), BG)
    d = ImageDraw.Draw(bar)
    f = font(17)
    x = 14
    items = [("рамка эталона", "#333333"), ("совпала с эталоном (IoU >= 0,5), цвет модели", MODEL_COLOR["dino_t0.25"]),
             ("FP: лишняя рамка", FP_COLOR), ("FN: пропущенная бабочка", FN_COLOR)]
    for text, color in items:
        d.rectangle([x, 12, x + 26, 28], outline=color, width=3)
        d.text((x + 34, 10), text, fill=INK2, font=f)
        x += 34 + d.textbbox((0, 0), text, font=f)[2] + 26
    return bar


def vstack(parts):
    out = Image.new("RGB", (max(p.width for p in parts), sum(p.height for p in parts)), BG)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height
    return out


def save(im, path):
    path = pathlib.Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, optimize=True)
