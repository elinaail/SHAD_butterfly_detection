"""Метрики всех конфигураций на gold_400_v3. Предсказания берутся из model_results_400/.

Результат: eval/results/*.csv и summary.json.
"""
import json
import pathlib
import sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import FN_TYPES, FP_TYPES, classify_photo, iou_matrix, match_pairs, prf

GOLD = ROOT / "gold_400_for_metrics"
PRED = ROOT / "model_results_400"
OUT = ROOT / "eval" / "results"

# конфигурация: (модель, порог)
CONFIGS = {
    "dino_t0.25": ("Grounding DINO", 0.25),
    "owlv2_t0.25": ("OWLv2", 0.25),
    "qwen": ("Qwen3-VL-2B", None),
    "dino_t0.15": ("Grounding DINO", 0.15),
    "dino_t0.40": ("Grounding DINO", 0.40),
    "owlv2_t0.15": ("OWLv2", 0.15),
    "owlv2_t0.40": ("OWLv2", 0.40),
}
MAIN = ["dino_t0.25", "owlv2_t0.25", "qwen"]

GROUPS = {
    "butterfly": "Дневные бабочки", "moth": "Ночные бабочки",
    "insect_dragonfly": "Похожие насекомые", "insect_caddisfly": "Похожие насекомые",
    "insect_beetle": "Похожие насекомые", "insect_caterpillar": "Гусеницы", "bg_plantae": "Растения и фон",
}
GROUP_ORDER = ["Дневные бабочки", "Ночные бабочки", "Похожие насекомые", "Гусеницы", "Растения и фон"]
CATEGORY_NAMES = {
    "butterfly_nymphalidae": "Нимфалиды", "butterfly_pieridae": "Белянки", "butterfly_lycaenidae": "Голубянки",
    "butterfly_papilionidae": "Парусники", "moth_geometridae": "Пяденицы", "moth_noctuidae": "Совки",
    "moth_erebidae": "Эребиды", "moth_sphingidae": "Бражники", "moth_crambidae": "Огневки-травянки",
    "insect_caterpillar": "Гусеницы", "bg_plantae": "Растения", "insect_dragonfly": "Стрекозы",
    "insect_caddisfly": "Ручейники", "insect_beetle": "Жуки",
}
SIZE_BINS = [(0, 0.01, "< 1% кадра"), (0.01, 0.05, "1-5%"), (0.05, 0.20, "5-20%"), (0.20, 1.01, "> 20%")]
N_BOOT = 2000
rng = np.random.default_rng(20261119)


def group_of(cat):
    return GROUPS.get(cat) or GROUPS[cat.split("_")[0]]


def count_group(n):
    if n == 0:
        return "Нет бабочек"
    if n == 1:
        return "1 бабочка"
    return "2-3 бабочки" if n <= 3 else "4+ бабочек"


def size_bin(area):
    return next(label for lo, hi, label in SIZE_BINS if lo <= area < hi)


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


meta = {r["photo_id"]: r for r in json.load(open(GOLD / "main_metadata_v2.json"))}
gold = {r["photo_id"]: r["boxes"] for r in load_jsonl(GOLD / "gold_400_v3.jsonl")}
gold_source = {int(r["photo_id"]): r["author"] for r in json.load(open(GOLD / "gold_400_provenance.json"))}
assert len(gold) == 400 and set(gold) == set(meta) == set(gold_source)

preds = {}
for name in CONFIGS:
    rows = {int(r["photo_id"]): r for r in load_jsonl(PRED / f"{name}.jsonl")}
    assert set(rows) == set(gold), f"{name}: набор фото не совпадает с gold"
    bad = [pid for pid, r in rows.items() if r.get("error")]
    assert not bad, f"{name}: error у {bad}"
    preds[name] = rows
ids = sorted(gold)

# по фото
per_photo, per_box, matched = [], [], []
for name in CONFIGS:
    for pid in ids:
        gt, r = gold[pid], preds[name][pid]
        pr = r["boxes"]
        cat = meta[pid]["category"]
        row = dict(config=name, photo_id=pid, category=cat, category_ru=CATEGORY_NAMES[cat], group=group_of(cat),
                   gold_source=gold_source[pid], count_group=count_group(len(gt)), n_gt=len(gt), n_pred=len(pr),
                   completion_method=r.get("completion_method"))
        for thr, tag in [(0.5, "50"), (0.75, "75")]:
            pairs, iou, fp_types, fn_types = classify_photo(gt, pr, thr)
            row[f"tp{tag}"] = len(pairs)
            row[f"fp{tag}"] = len(pr) - len(pairs)
            row[f"fn{tag}"] = len(gt) - len(pairs)
            if thr != 0.5:
                continue
            counts = Counter(fp_types.values()) + Counter(fn_types.values())
            row.update({t: counts.get(t, 0) for t in FP_TYPES + FN_TYPES})
            found = {g for g, _ in pairs}
            for i, b in enumerate(gt):
                area = b["width"] * b["height"]
                per_box.append(dict(config=name, photo_id=pid, size_bin=size_bin(area), tp50=i in found,
                                    best_iou=float(iou[i].max()) if len(pr) else 0.0))
            matched += [dict(config=name, photo_id=pid, iou=float(iou[g, p])) for g, p in pairs]
        per_photo.append(row)

pp = pd.DataFrame(per_photo)
pb = pd.DataFrame(per_box)


def summarize(df):
    s = df[["tp50", "fp50", "fn50", "tp75", "fp75", "fn75"]].sum()
    p50, r50, f50 = prf(s.tp50, s.fp50, s.fn50)
    p75, r75, f75 = prf(s.tp75, s.fp75, s.fn75)
    empty = df[df.n_gt == 0]
    return dict(photos=len(df), gt_boxes=int(df.n_gt.sum()), pred_boxes=int(df.n_pred.sum()),
                TP50=int(s.tp50), FP50=int(s.fp50), FN50=int(s.fn50), P50=p50, R50=r50, F1_50=f50,
                TP75=int(s.tp75), FP75=int(s.fp75), FN75=int(s.fn75), P75=p75, R75=r75, F1_75=f75,
                empty_photos=len(empty), fp_boxes_on_empty=int(empty.n_pred.sum()),
                empty_photos_with_fp=int((empty.n_pred > 0).sum()),
                share_empty_with_fp=float((empty.n_pred > 0).mean()) if len(empty) else np.nan)


def bootstrap_f1(df, cols):
    a = df[cols].to_numpy(float)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    s = a[idx].sum(axis=1)
    f1 = 2 * s[:, 0] / np.maximum(2 * s[:, 0] + s[:, 1] + s[:, 2], 1)
    return np.percentile(f1, [2.5, 97.5])


def timing(name):
    # без warmup и без фото, которые Qwen дообрабатывал повторным запуском
    t = np.array([r["elapsed_seconds"] for r in preds[name].values()
                  if not r.get("warmup") and not r.get("completion_method")])
    return dict(sec_per_photo_mean=float(t.mean()), sec_per_photo_median=float(np.median(t)), timed_photos=len(t))


OUT.mkdir(parents=True, exist_ok=True)

rows = []
for name, (model, thr) in CONFIGS.items():
    df = pp[pp.config == name]
    lo50, hi50 = bootstrap_f1(df, ["tp50", "fp50", "fn50"])
    lo75, hi75 = bootstrap_f1(df, ["tp75", "fp75", "fn75"])
    rows.append(dict(config=name, model=model, threshold=thr, main=name in MAIN, **summarize(df),
                     F1_50_ci_lo=lo50, F1_50_ci_hi=hi50, F1_75_ci_lo=lo75, F1_75_ci_hi=hi75, **timing(name),
                     reprocessed_photos=int(df.completion_method.notna().sum())))
lb = pd.DataFrame(rows)
lb.to_csv(OUT / "leaderboard.csv", index=False, float_format="%.4f")


def breakdown(col):
    out = []
    for name in CONFIGS:
        for key, df in pp[pp.config == name].groupby(col, sort=False):
            out.append(dict(config=name, **{col: key}, **summarize(df)))
    return pd.DataFrame(out)


by_group = breakdown("group")
by_group["group"] = pd.Categorical(by_group.group, GROUP_ORDER)
by_group.sort_values(["config", "group"]).to_csv(OUT / "by_group.csv", index=False, float_format="%.4f")
breakdown("category_ru").to_csv(OUT / "by_category.csv", index=False, float_format="%.4f")
breakdown("count_group").to_csv(OUT / "by_count.csv", index=False, float_format="%.4f")
breakdown("gold_source").to_csv(OUT / "by_gold_source.csv", index=False, float_format="%.4f")

size = pb.groupby(["config", "size_bin"]).agg(gt_boxes=("tp50", "size"), recall50=("tp50", "mean"),
                                              median_best_iou=("best_iou", "median")).reset_index()
size.to_csv(OUT / "recall_by_size.csv", index=False, float_format="%.4f")

empty = pp[pp.n_gt == 0]
fp_empty = empty.groupby(["config", "category_ru"]).agg(
    photos=("photo_id", "size"), fp_boxes=("n_pred", "sum"),
    photos_with_fp=("n_pred", lambda s: int((s > 0).sum()))).reset_index()
fp_empty["share_with_fp"] = fp_empty.photos_with_fp / fp_empty.photos
fp_empty.to_csv(OUT / "fp_on_empty.csv", index=False, float_format="%.4f")

pp.groupby("config")[FP_TYPES + FN_TYPES].sum().reset_index().to_csv(OUT / "error_types.csv", index=False)
pp.to_csv(OUT / "per_photo.csv", index=False, float_format="%.4f")
pd.DataFrame(matched).to_csv(OUT / "matched_iou.csv", index=False, float_format="%.4f")

# F1 от порога: фильтруем по score рамки из прогона с порогом 0.15
curve = []
for name, model in [("dino_t0.15", "Grounding DINO"), ("owlv2_t0.15", "OWLv2")]:
    for t in np.round(np.arange(0.15, 0.71, 0.01), 2):
        tot = np.zeros(6)
        for pid in ids:
            r = preds[name][pid]
            pr = [b for b, s in zip(r["boxes"], r["scores"]) if s >= t - 1e-9]
            iou = iou_matrix(gold[pid], pr)
            tp50, tp75 = len(match_pairs(iou, 0.5)), len(match_pairs(iou, 0.75))
            n_gt = len(gold[pid])
            tot += [tp50, len(pr) - tp50, n_gt - tp50, tp75, len(pr) - tp75, n_gt - tp75]
        p50, r50, f50 = prf(*tot[:3])
        p75, r75, f75 = prf(*tot[3:])
        curve.append(dict(model=model, threshold=t, P50=p50, R50=r50, F1_50=f50, P75=p75, R75=r75, F1_75=f75))
curve = pd.DataFrame(curve)
curve.to_csv(OUT / "threshold_curve.csv", index=False, float_format="%.4f")

# фильтрация файла t0.15 должна давать те же F1, что и файлы t0.25 и t0.40
check = {}
for model, short in [("Grounding DINO", "dino"), ("OWLv2", "owlv2")]:
    for t in (0.25, 0.40):
        f_curve = curve[(curve.model == model) & np.isclose(curve.threshold, t)].F1_50.item()
        f_file = lb[lb.config == f"{short}_t{t:.2f}"].F1_50.item()
        check[f"{short}@{t}"] = dict(curve=round(f_curve, 4), file=round(f_file, 4))

summary = dict(photos=len(ids), threshold_filter_check=check,
               qwen_reprocessed=sorted(p for p, r in preds["qwen"].items() if r.get("completion_method")))
json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=2)

pd.set_option("display.width", 200, "display.max_columns", 20)
print(lb[["config", "P50", "R50", "F1_50", "F1_50_ci_lo", "F1_50_ci_hi", "P75", "R75", "F1_75",
          "fp_boxes_on_empty", "empty_photos_with_fp", "sec_per_photo_median"]].round(3).to_string(index=False))
print(json.dumps(summary, ensure_ascii=False))
