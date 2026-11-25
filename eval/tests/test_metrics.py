import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from metrics import classify_photo, match, prf


def box(left, top, width, height):
    return dict(left=left, top=top, width=width, height=height)


def test_identity():
    g = [box(.1, .1, .2, .2), box(.5, .5, .2, .2)]
    assert match(g, g, .5) == 2
    assert match(g, g, .75) == 2


def test_max_matching():
    # жадный подбор по IoU отдает p0 первой рамке gt и теряет вторую
    g = [box(0, 0, 1, 1), box(.3, 0, 1, 1)]
    p = [box(.15, 0, 1, 1), box(-.2, 0, 1, 1)]
    assert match(g, p, .5) == 2


def test_empty():
    assert match([], [box(0, 0, .1, .1)]) == 0
    assert match([box(0, 0, .1, .1)], []) == 0


def test_iou_exactly_threshold():
    assert match([box(0, 0, 1, 1)], [box(0, 0, 1, .5)], .5) == 1


def test_error_types():
    g = [box(.1, .1, .1, .1), box(.25, .1, .1, .1)]
    _, _, fp, fn = classify_photo(g, [box(.09, .09, .27, .12)])
    assert list(fp.values()) == ["fp_group_box"]
    assert set(fn.values()) == {"fn_in_group_box"}

    _, _, fp, _ = classify_photo([], [box(0, 0, .1, .1)])
    assert fp == {0: "fp_empty_frame"}

    _, _, fp, _ = classify_photo([box(.1, .1, .2, .2)], [box(.1, .1, .2, .2), box(.11, .1, .2, .2)])
    assert list(fp.values()) == ["fp_duplicate"]


def test_prf():
    assert prf(0, 0, 0) == (0, 0, 0)
    assert prf(1, 1, 0)[2] == 2 / 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
