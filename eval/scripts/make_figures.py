"""Графики для презентации: eval/figures/charts/*.png (3200x1800). Запускать после evaluate.py."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from viz_common import FIG, MODEL_COLOR, MODEL_NAME, MODELS, ROOT, load_all
from metrics import iou_matrix, match_pairs

RES = ROOT / "eval" / "results"
OUT = FIG / "charts"

BG, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GRAY = "#b9b8b0"
FONT = "Ubuntu Sans, DejaVu Sans, sans-serif"
BLUES = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGES = ["#fdf1ec", "#fbd7c6", "#f6b296", "#f08a63", "#eb6834", "#c4501f", "#963c17", "#6b2a10"]

summary = json.load(open(RES / "summary.json"))
N_PHOTOS = summary["photos"]


def pct(v):
    return f"{v:.0%}" if v == v else "-"


def num(v):
    return f"{v:.2f}".replace(".", ",")


def style(fig, title, subtitle=None, h=900, w=1600, legend=True):
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:20px;color:{INK2}'>{subtitle}</span>"
    fig.update_layout(
        template="none", width=w, height=h, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, size=19, color=INK2),
        title=dict(text=text, x=0.03, xanchor="left", y=0.965, yanchor="top", font=dict(size=30, color=INK)),
        margin=dict(l=90, r=50, t=150 if subtitle else 110, b=200 if legend else 120),
        separators=", ", showlegend=legend,
        legend=dict(orientation="h", x=0.03, xanchor="left", yref="container", y=46 / h, yanchor="bottom",
                    font=dict(size=19, color=INK2), itemsizing="constant"),
        bargap=0.32, bargroupgap=0.12,
    )
    fig.update_xaxes(automargin=True, showgrid=False, linecolor=AXIS, ticks="", zeroline=False,
                     tickfont=dict(color=INK2))
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor=GRID, linecolor=AXIS, zeroline=False,
                     tickfont=dict(color=INK2))
    fig.update_annotations(font=dict(family=FONT, color=INK))


def note(fig, text):
    # подпись у нижнего края, под легендой
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0, y=0, xanchor="left", yanchor="bottom",
                       yshift=-(fig.layout.margin.b - 14), showarrow=False, font=dict(size=15, color=MUTED))


def save(fig, name):
    fig.write_image(OUT / f"{name}.png", scale=2)
    print(name)


def bar(m, **kw):
    return go.Bar(name=MODEL_NAME[m], legendgroup=m, marker=dict(color=MODEL_COLOR[m], cornerradius=4),
                  cliponaxis=False, **kw)


OUT.mkdir(parents=True, exist_ok=True)
lb = pd.read_csv(RES / "leaderboard.csv").set_index("config")


def leaderboard():
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.09,
                        subplot_titles=("IoU >= 0,5 (основная метрика)", "IoU >= 0,75 (точность рамок)"))
    names = ["Precision", "Recall", "F1"]
    for col, thr in [(1, "50"), (2, "75")]:
        for m in MODELS:
            r = lb.loc[m]
            vals = [r[f"P{thr}"], r[f"R{thr}"], r[f"F1_{thr}"]]
            err = dict(type="data", symmetric=False, array=[0, 0, r[f"F1_{thr}_ci_hi"] - vals[2]],
                       arrayminus=[0, 0, vals[2] - r[f"F1_{thr}_ci_lo"]], color=INK2, thickness=1.5, width=5)
            fig.add_trace(bar(m, y=names, x=vals, orientation="h", showlegend=col == 1, error_x=err,
                              text=[num(v) for v in vals], textposition="inside", insidetextanchor="start",
                              textfont=dict(size=18, color="#ffffff")), row=1, col=col)
    style(fig, "Qwen3-VL лидирует по F1 на обоих порогах IoU",
          f"Сумма TP/FP/FN по {N_PHOTOS} фото gold, усы у F1: 95% бутстреп-интервал по фото")
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=True, gridcolor=GRID, dtick=0.25)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=21, color=INK))
    fig.update_annotations(font_size=21)
    save(fig, "01_leaderboard")


def groups():
    bg = pd.read_csv(RES / "by_group.csv")
    pos = ["Дневные бабочки", "Ночные бабочки"]
    neg = ["Похожие насекомые", "Гусеницы", "Растения и фон"]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.1,
                        subplot_titles=("Recall@0,5 на кадрах с бабочками", "Доля пустых кадров с ложной рамкой"))
    for m in MODELS:
        d = bg[bg.config == m].set_index("group")
        fig.add_trace(bar(m, x=pos, y=d.loc[pos, "R50"], text=[pct(v) for v in d.loc[pos, "R50"]],
                          textposition="outside", textfont=dict(size=16, color=INK)), row=1, col=1)
        share = d.loc[neg, "share_empty_with_fp"]
        fig.add_trace(bar(m, x=neg, y=share, showlegend=False, text=[pct(v) for v in share],
                          textposition="outside", textfont=dict(size=16, color=INK)), row=1, col=2)
    similar = bg[(bg.config == "qwen") & (bg.group == "Похожие насекомые")].iloc[0]
    style(fig, "На бабочках модели близки, разница в ложных рамках на пустых кадрах",
          "Grounding DINO ставит рамку почти на каждый кадр без бабочек")
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", dtick=0.25)
    fig.update_annotations(font_size=21)
    note(fig, f"Похожие насекомые: доля считается по {similar.empty_photos} кадрам без бабочек "
              f"из {similar.photos}.")
    save(fig, "02_groups")


def fp_by_category():
    fe = pd.read_csv(RES / "fp_on_empty.csv")
    first = fe[fe.config == MODELS[0]].sort_values("photos", ascending=False)
    order = first[first.photos >= 5].category_ru.tolist()
    fig = go.Figure()
    for m in MODELS:
        d = fe[fe.config == m].set_index("category_ru").loc[order]
        labels = [f"{c}  <span style='color:{MUTED}'>n={n}</span>" for c, n in zip(order, d.photos)]
        fig.add_trace(bar(m, y=labels, x=d.share_with_fp, orientation="h", textposition="outside",
                          text=[f"{pct(s)}  ({b} рам.)" for s, b in zip(d.share_with_fp, d.fp_boxes)],
                          textfont=dict(size=15, color=INK2)))
    style(fig, "Ложные рамки на кадрах без бабочек",
          "Доля кадров хотя бы с одной рамкой, в скобках число лишних рамок")
    fig.update_xaxes(range=[0, 1.2], tickformat=".0%", showgrid=True, gridcolor=GRID, tickvals=[0, .25, .5, .75, 1])
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=20, color=INK))
    note(fig, "Не показаны 2 пустых кадра с рисунками бабочек: рамку на них ставят все три модели.")
    save(fig, "03_fp_empty_by_category")


def threshold():
    cv = pd.read_csv(RES / "threshold_curve.csv")
    qw = lb.loc["qwen"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("F1@0,5 в зависимости от порога", "Precision и Recall (IoU 0,5)"))
    for model, key in [("Grounding DINO", "dino_t0.25"), ("OWLv2", "owlv2_t0.25")]:
        d = cv[cv.model == model]
        c = MODEL_COLOR[key]
        dot = dict(color=c, size=11, line=dict(color=BG, width=2))
        saved = d[d.threshold.round(2).isin([0.15, 0.25, 0.40])]
        best = d.loc[d.F1_50.idxmax()]
        fig.add_scatter(x=d.threshold, y=d.F1_50, mode="lines", name=model, line=dict(color=c, width=3),
                        row=1, col=1)
        fig.add_scatter(x=saved.threshold, y=saved.F1_50, mode="markers", marker=dot, showlegend=False,
                        row=1, col=1)
        fig.add_annotation(x=best.threshold, y=best.F1_50, text=f"макс. {num(best.F1_50)} при {num(best.threshold)}",
                           ax=0, ay=-34, arrowhead=0, arrowcolor=MUTED, font=dict(size=15, color=INK2),
                           row=1, col=1)
        fig.add_scatter(x=d.R50, y=d.P50, mode="lines", line=dict(color=c, width=3), showlegend=False,
                        row=1, col=2)
        fig.add_scatter(x=saved.R50, y=saved.P50, mode="markers+text", marker=dot, showlegend=False,
                        text=[num(t) for t in saved.threshold], textposition="middle left",
                        textfont=dict(size=14, color=INK2), row=1, col=2)
    qc = MODEL_COLOR["qwen"]
    fig.add_hline(y=qw.F1_50, line=dict(color=qc, width=2), row=1, col=1)
    fig.add_annotation(x=0.70, y=qw.F1_50, xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"Qwen3-VL (без score): {num(qw.F1_50)}", font=dict(size=15, color=INK2), row=1, col=1)
    fig.add_scatter(x=[qw.R50], y=[qw.P50], mode="markers+text", name="Qwen3-VL-2B", text=["Qwen3-VL"],
                    marker=dict(color=qc, size=14, symbol="diamond", line=dict(color=BG, width=2)),
                    textposition="top center", textfont=dict(size=15, color=INK2), row=1, col=2)
    style(fig, "Подбор порога не догоняет Qwen3-VL",
          "Кривые получены фильтрацией по score из запуска с порогом 0,15. Точки: конфигурации 0,15, 0,25 и 0,40")
    fig.update_xaxes(title_text="порог уверенности рамки", range=[0.14, 0.71], dtick=0.1, row=1, col=1)
    fig.update_yaxes(title_text="F1@0,5", range=[0.55, 0.95], tickformat=".2f", row=1, col=1)
    fig.update_xaxes(title_text="Recall", range=[0.55, 0.97], tickformat=".0%", showgrid=True, gridcolor=GRID,
                     row=1, col=2)
    fig.update_yaxes(title_text="Precision", range=[0.4, 1.0], tickformat=".0%", row=1, col=2)
    for t in ("F1@0,5 в зависимости от порога", "Precision и Recall (IoU 0,5)"):
        fig.update_annotations(selector=dict(text=t), font_size=21)
    note(fig, f"Порог подбирался на тех же {N_PHOTOS} фото, поэтому максимум F1 завышен.")
    save(fig, "04_threshold")


def error_types():
    et = pd.read_csv(RES / "error_types.csv").set_index("config")
    fp_labels = {"fp_empty_frame": "На кадре без бабочек", "fp_background": "На фоне или другом объекте",
                 "fp_loose": "Неточная рамка (IoU < 0,5)", "fp_part_box": "На часть бабочки (крыло)",
                 "fp_group_box": "Одна рамка на несколько", "fp_duplicate": "Дубль рамки"}
    fn_labels = {"fn_missed": "Не найдена совсем", "fn_loose": "Найдена неточно (IoU < 0,5)",
                 "fn_in_group_box": "Попала в общую рамку группы"}
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], horizontal_spacing=0.22,
                        subplot_titles=("Лишние рамки (FP)", "Пропущенные бабочки (FN)"))
    for m in MODELS:
        for col, labels in [(1, fp_labels), (2, fn_labels)]:
            vals = [int(et.loc[m, k]) for k in labels]
            fig.add_trace(bar(m, y=list(labels.values()), x=vals, orientation="h", showlegend=col == 1,
                              text=[str(v) if v else "" for v in vals], textposition="outside",
                              textfont=dict(size=15, color=INK2)), row=1, col=col)
    style(fig, "Типы ошибок при IoU 0,5",
          "У DINO почти все FP на пустых кадрах, у OWLv2 больше неточных и дробных рамок")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=18, color=INK))
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_annotations(font_size=21)
    save(fig, "05_error_types")


def recall_by_size():
    sz = pd.read_csv(RES / "recall_by_size.csv")
    bins = ["< 1% кадра", "1-5%", "5-20%", "> 20%"]
    n = sz[sz.config == "qwen"].set_index("size_bin").loc[bins, "gt_boxes"]
    x = [f"{b}<br><span style='color:{MUTED};font-size:15px'>{k} рамок</span>" for b, k in zip(bins, n)]
    fig = go.Figure()
    for m in MODELS:
        d = sz[sz.config == m].set_index("size_bin").loc[bins]
        fig.add_trace(bar(m, x=x, y=d.recall50, text=[pct(v) for v in d.recall50], textposition="outside",
                          textfont=dict(size=16, color=INK)))
    style(fig, "Мелкие бабочки: слабое место Grounding DINO", "Recall@0,5 по площади рамки gold относительно кадра")
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", dtick=0.25)
    fig.update_xaxes(title_text="площадь бабочки, доля кадра")
    note(fig, "Группа '< 1% кадра': 12 рамок, почти все на одном фото (76337845).")
    save(fig, "06_recall_by_size")


def iou_survival():
    mi = pd.read_csv(RES / "matched_iou.csv")
    xs = np.linspace(0.5, 1.0, 201)
    fig = go.Figure()
    for m in MODELS:
        v = mi[mi.config == m].iou.to_numpy()
        keep = (v >= 0.75).mean()
        fig.add_scatter(x=xs, y=[(v >= x).mean() for x in xs], mode="lines", name=MODEL_NAME[m],
                        line=dict(color=MODEL_COLOR[m], width=3))
        fig.add_scatter(x=[0.75], y=[keep], mode="markers", showlegend=False,
                        marker=dict(color=MODEL_COLOR[m], size=12, line=dict(color=BG, width=2)))
        shift = {"dino_t0.25": 14, "qwen": -14}.get(m, 0)  # подписи DINO и Qwen почти совпадают
        fig.add_annotation(x=0.75, y=keep, text=f"{MODEL_NAME[m]}: {keep:.0%}", showarrow=False, xanchor="right",
                           xshift=-12, yshift=shift, font=dict(size=16, color=INK2))
    fig.add_vline(x=0.75, line=dict(color=AXIS, width=1.5))
    style(fig, "OWLv2 находит бабочек, но режет края рамки",
          "Доля пар, совпавших при IoU 0,5, у которых IoU не ниже порога. Точки: порог 0,75")
    fig.update_xaxes(title_text="порог IoU", range=[0.5, 1.0], dtick=0.05, showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title_text="доля совпавших пар", range=[0, 1.03], tickformat=".0%")
    save(fig, "07_iou_survival")


def score_separation():
    _, gold, _ = load_all()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Grounding DINO", "OWLv2"), horizontal_spacing=0.08)
    bins = dict(start=0.15, end=1.0, size=0.05)
    for col, (key, src) in enumerate([("dino_t0.25", "dino_t0.15"), ("owlv2_t0.25", "owlv2_t0.15")], 1):
        rows = [json.loads(line) for line in open(ROOT / "model_results_400" / f"{src}.jsonl")]
        tp_scores, empty_scores = [], []
        for r in rows:
            gt = gold[r["photo_id"]]
            if not gt:
                if r["scores"]:
                    empty_scores.append(max(r["scores"]))
                continue
            tp_scores += [r["scores"][p] for _, p in match_pairs(iou_matrix(gt, r["boxes"]), 0.5)]
        fig.add_histogram(x=tp_scores, xbins=bins, histnorm="percent", name="рамка совпала с gold (цвет модели)",
                          legendgroup="tp", showlegend=col == 1,
                          marker=dict(color=MODEL_COLOR[key], line=dict(color=BG, width=1)), row=1, col=col)
        fig.add_histogram(x=empty_scores, xbins=bins, histnorm="percent", name="лучшая рамка на пустом кадре",
                          legendgroup="fp", showlegend=col == 1,
                          marker=dict(color=GRAY, line=dict(color=BG, width=1)), row=1, col=col)
        fig.add_vline(x=0.25, line=dict(color=INK2, width=1.5), row=1, col=col)
        fig.add_annotation(x=0.25, y=1, yref="y domain" if col == 1 else "y2 domain", text="порог 0,25",
                           showarrow=False, xanchor="left", xshift=6, yanchor="top", font=dict(size=15, color=INK2),
                           row=1, col=col)
    style(fig, "У DINO уверенность на пустых кадрах такая же, как на бабочках",
          "Score рамок, совпавших с gold, и самой уверенной рамки на кадре без бабочек")
    fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.05)
    fig.update_xaxes(title_text="score рамки", range=[0.15, 1.0], dtick=0.1)
    fig.update_yaxes(title_text="% рамок / кадров", col=1)
    fig.update_annotations(font_size=21)
    save(fig, "08_score_separation")


def by_count():
    bc = pd.read_csv(RES / "by_count.csv")
    order = ["1 бабочка", "2-3 бабочки", "4+ бабочек"]
    fig = go.Figure()
    for m in MODELS:
        d = bc[bc.config == m].set_index("count_group").loc[order]
        x = [f"{c}<br><span style='color:{MUTED};font-size:15px'>{p} фото, {g} рамок</span>"
             for c, p, g in zip(order, d.photos, d.gt_boxes)]
        fig.add_trace(bar(m, x=x, y=d.R50, text=[pct(v) for v in d.R50], textposition="outside",
                          textfont=dict(size=16, color=INK)))
    style(fig, "Скопления бабочек: слабое место всех моделей",
          "Recall@0,5 в зависимости от числа бабочек на фото (только фото с бабочками)")
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%", dtick=0.25)
    save(fig, "09_by_count")


def category_heatmap():
    cat = pd.read_csv(RES / "by_category.csv")
    pos = ["Нимфалиды", "Белянки", "Голубянки", "Парусники", "Пяденицы", "Совки", "Эребиды", "Бражники",
           "Огневки-травянки"]
    neg = ["Растения", "Гусеницы", "Стрекозы", "Ручейники", "Жуки"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                        subplot_titles=("Recall@0,5 по семействам", "Доля пустых кадров с FP"))
    for col, names, value, count, ramp, zmin in [(1, pos, "R50", "gt_boxes", BLUES, 0.6),
                                                 (2, neg, "share_empty_with_fp", "empty_photos", ORANGES, 0)]:
        sub = cat.set_index(["config", "category_ru"])
        z = np.array([[sub.loc[(m, c), value] for m in MODELS] for c in names])
        n = [sub.loc[("qwen", c), count] for c in names]
        fig.add_heatmap(z=z, x=[MODEL_NAME[m] for m in MODELS],
                        y=[f"{c} <span style='color:{MUTED}'>({k})</span>" for c, k in zip(names, n)],
                        colorscale=[[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)], zmin=zmin, zmax=1,
                        showscale=False, xgap=3, ygap=3, text=[[pct(v) for v in row] for row in z],
                        texttemplate="%{text}", textfont=dict(size=17), row=1, col=col)
    style(fig, "Разбивка по таксономическим категориям",
          "В скобках: число рамок gold (слева) или пустых кадров (справа). Шкала recall начинается с 60%",
          h=1000, legend=False)
    fig.update_layout(margin_t=220)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=18, color=INK))
    fig.update_xaxes(side="top", tickfont=dict(size=17, color=INK))
    fig.update_annotations(font_size=21, yshift=40)
    save(fig, "10_category_heatmap")


leaderboard()
groups()
fp_by_category()
threshold()
error_types()
recall_by_size()
iou_survival()
score_separation()
by_count()
category_heatmap()
