"""F1 по рамкам как в Контесте и классификация ошибок."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def to_xyxy(boxes):
    if not boxes:
        return np.zeros((0, 4))
    return np.array([(b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]) for b in boxes], float)


def _intersections(a, b):
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return w * h


def _area(a):
    return (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])


def iou_matrix(gt, pr):
    g, p = to_xyxy(gt), to_xyxy(pr)
    if len(g) == 0 or len(p) == 0:
        return np.zeros((len(g), len(p)))
    inter = _intersections(g, p)
    union = _area(g)[:, None] + _area(p)[None, :] - inter
    # без eps, иначе IoU ровно 0.5 не проходит порог
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_pairs(iou, thr):
    """Максимальное паросочетание по ребрам IoU >= thr. При равном числе пар берется большая сумма IoU."""
    if iou.size == 0:
        return []
    ok = iou >= thr
    r, c = linear_sum_assignment(-(ok * (1000.0 + iou)))
    return [(int(i), int(j)) for i, j in zip(r, c) if ok[i, j]]


def match(gt, pr, thr=0.5):
    return len(match_pairs(iou_matrix(gt, pr), thr))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return p, r, f


FP_TYPES = ["fp_empty_frame", "fp_duplicate", "fp_group_box", "fp_part_box", "fp_loose", "fp_background"]
FN_TYPES = ["fn_missed", "fn_loose", "fn_in_group_box"]


def _inside_share(inner, outer):
    """Доля площади каждой рамки inner, которая лежит внутри рамок outer."""
    a, b = to_xyxy(inner), to_xyxy(outer)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return _intersections(a, b) / (_area(a)[:, None] + 1e-12)


def classify_photo(gt, pr, thr=0.5):
    """Пары TP и тип каждой лишней рамки (FP) и каждого пропуска (FN) на одном фото."""
    iou = iou_matrix(gt, pr)
    pairs = match_pairs(iou, thr)
    matched_gt = {i for i, _ in pairs}
    matched_pr = {j for _, j in pairs}
    gt_in_pr = _inside_share(gt, pr)
    pr_in_gt = _inside_share(pr, gt)

    fp_types, fn_types, group_boxes = {}, {}, set()
    for j in range(len(pr)):
        if j in matched_pr:
            continue
        if not gt:
            fp_types[j] = "fp_empty_frame"
            continue
        best = iou[:, j].max()
        if best >= thr:
            fp_types[j] = "fp_duplicate"
        elif (gt_in_pr[:, j] >= 0.7).sum() >= 2:
            fp_types[j] = "fp_group_box"
            group_boxes.add(j)
        elif pr_in_gt[j].max() >= 0.8 and best < 0.35:
            fp_types[j] = "fp_part_box"
        elif best >= 0.1:
            fp_types[j] = "fp_loose"
        else:
            fp_types[j] = "fp_background"

    for i in range(len(gt)):
        if i in matched_gt:
            continue
        if any(gt_in_pr[i, j] >= 0.7 for j in group_boxes):
            fn_types[i] = "fn_in_group_box"
        elif len(pr) and iou[i].max() >= 0.1:
            fn_types[i] = "fn_loose"
        else:
            fn_types[i] = "fn_missed"
    return pairs, iou, fp_types, fn_types
