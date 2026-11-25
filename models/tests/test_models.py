import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_models import box_from_xyxy, nms, parse_qwen, parse_qwen_truncated
from export_predictions import export
from complete_qwen import merge_completed


def test_box_coordinate_conversion():
    assert box_from_xyxy([20, 10, 80, 40], 100, 50) == pytest.approx(
        dict(left=0.2, top=0.2, width=0.6, height=0.6)
    )
    assert box_from_xyxy([-10, 0, 110, 50], 100, 50) == dict(
        left=0, top=0, width=1, height=1
    )
    for bad in ([80, 10, 20, 40], [0, 0, float("nan"), 10], [200, 0, 300, 10]):
        with pytest.raises(ValueError):
            box_from_xyxy(bad, 100, 50)


def test_qwen_empty_and_invalid_are_distinct():
    assert parse_qwen("[]") == []
    assert parse_qwen("[[100, 200, 600, 900]]")[0] == pytest.approx(
        dict(left=0.1, top=0.2, width=0.5, height=0.7)
    )
    assert parse_qwen('[{"bbox_2d": [100, 200, 600, 900], "label":"moth"}]')[
        0
    ] == pytest.approx(dict(left=0.1, top=0.2, width=0.5, height=0.7))
    assert parse_qwen('```json\n[{"box": [100, 200, 600, 900]}]\n```')[
        0
    ] == pytest.approx(dict(left=0.1, top=0.2, width=0.5, height=0.7))
    for invalid in ("not json", '{"boxes":[]}', '[{"box":[900,0,10,100]}]'):
        with pytest.raises((ValueError, TypeError)):
            parse_qwen(invalid)


def test_nms_merges_same_insect_but_keeps_separate_insects():
    boxes = [
        box_from_xyxy(x, 100, 100)
        for x in ([0, 0, 40, 40], [1, 1, 41, 41], [60, 60, 90, 90])
    ]
    assert nms(boxes, [0.7, 0.9, 0.6]) == [1, 2]


def test_truncated_json_retains_only_complete_boxes_and_preserves_duplicates():
    text = "```json\n[[100,200,600,900], [100,200,600,900], [500, "
    boxes = parse_qwen_truncated(text)
    assert len(boxes) == 2 and boxes[0] == boxes[1]
    assert boxes[0] == pytest.approx(dict(left=0.1, top=0.2, width=0.5, height=0.7))
    with pytest.raises(ValueError, match="no complete"):
        parse_qwen_truncated("[[100,")


def test_export_requires_complete_run_and_preserves_empty_images(tmp_path):
    manifest, source, target = [
        tmp_path / s for s in ("main.jsonl", "pred.jsonl", "out.jsonl")
    ]
    manifest.write_text('{"photo_id":1}\n{"photo_id":2}\n')
    row = dict(
        photo_id=1,
        model="dino",
        status="ok",
        boxes=[dict(left=0, top=0, width=1, height=1)],
        scores=[0.2],
    )
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        export(source, manifest, target, 0.25)
    source.write_text(
        json.dumps(row)
        + "\n"
        + json.dumps(dict(row, photo_id=2, boxes=[], scores=[]))
        + "\n"
    )
    assert export(source, manifest, target, 0.25) == 2
    assert all(
        not json.loads(line)["boxes"] for line in target.read_text().splitlines()
    )


def test_completion_preserves_good_answers_and_accounts_for_both_attempts():
    good = dict(
        photo_id=1,
        model="Qwen/test",
        status="ok",
        raw_text="[]",
        boxes=[],
        scores=[],
        elapsed_seconds=1.0,
        image_sha256="a",
    )
    bad = dict(
        good,
        photo_id=2,
        status="truncated",
        response_truncated=True,
        image_sha256="b",
        elapsed_seconds=3.0,
    )
    retry = dict(
        bad,
        status="ok",
        response_truncated=False,
        ended_with_eos=True,
        elapsed_seconds=2.0,
    )
    merged = merge_completed([good, bad], [retry])
    assert merged[0] == good
    assert merged[1]["status"] == "ok" and merged[1]["elapsed_seconds"] == 5.0
    assert (
        merged[1]["initial_elapsed_seconds"] == 3.0
        and merged[1]["retry_elapsed_seconds"] == 2.0
    )
    assert bad["status"] == "truncated"


def test_completion_rejects_missing_or_unfinished_retries():
    bad = dict(
        photo_id=1,
        model="Qwen/test",
        status="truncated",
        raw_text="[]",
        boxes=[],
        scores=[],
        elapsed_seconds=1.0,
        image_sha256="a",
    )
    with pytest.raises(ValueError, match="exactly match"):
        merge_completed([bad], [])
    with pytest.raises(ValueError, match="not complete"):
        merge_completed(
            [bad],
            [dict(bad, status="ok", response_truncated=False, ended_with_eos=False)],
        )


def test_tiled_recovery_maps_boxes_and_merges_overlap():
    from qwen_tiles import combine_tiles

    def tile(bounds, coords, seconds):
        text = json.dumps([coords])
        return dict(
            crop_xyxy=bounds,
            prediction=dict(
                status="ok",
                ended_with_eos=True,
                raw_text=text,
                boxes=parse_qwen(text),
                elapsed_seconds=seconds,
            ),
        )

    tiles = [
        tile([0, 0, 60, 50], [500, 200, 1000, 600], 2),
        tile([30, 0, 90, 50], [0, 200, 500, 600], 3),
    ]
    boxes, elapsed = combine_tiles(tiles, 100, 100)
    assert len(boxes) == 1
    assert boxes[0] == pytest.approx(dict(left=0.3, top=0.1, width=0.3, height=0.2))
    assert elapsed == 5
    tiles[1]["prediction"]["ended_with_eos"] = False
    with pytest.raises(ValueError, match="Incomplete tile"):
        combine_tiles(tiles, 100, 100)
