"""Собирает model_results_400/ из model_results/ и model_replacements_results/.

Строки трех удаленных фото заменяются строками новых фото на тех же местах, остальные 397 не меняются.
sha256 из model_code/results/replacements_2026_11_18/validation.json здесь не совпадет: там хешируются
полные файлы с raw_text, а не компактные.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OLD = ROOT / "model_results"
NEW = ROOT / "model_replacements_results"
OUT = ROOT / "model_results_400"

mapping = {r["old_photo_id"]: r["new_photo_id"] for r in json.load(open(NEW / "replacements.json"))}
gold_ids = {json.loads(line)["photo_id"] for line in open(ROOT / "gold_400_for_metrics" / "gold_400_v2.jsonl")}
OUT.mkdir(exist_ok=True)


def check_row(row):
    assert isinstance(row["boxes"], list) and not row.get("error")
    for b in row["boxes"]:
        assert b["left"] >= 0 and b["top"] >= 0 and b["width"] > 0 and b["height"] > 0
        assert b["left"] + b["width"] <= 1 + 1e-6 and b["top"] + b["height"] <= 1 + 1e-6


for path in sorted(NEW.glob("*.jsonl")):
    new = {json.loads(line)["photo_id"]: line for line in path.read_text(encoding="utf-8").splitlines() if line}
    assert set(new) == set(mapping.values()), path.name

    merged, kept = [], 0
    for line in (OLD / path.name).read_text(encoding="utf-8").splitlines():
        pid = json.loads(line)["photo_id"]
        if pid in mapping:
            merged.append(new[mapping[pid]])
        else:
            merged.append(line)
            kept += 1

    rows = [json.loads(line) for line in merged]
    ids = [r["photo_id"] for r in rows]
    assert kept == 397 and len(ids) == 400 and len(set(ids)) == 400, path.name
    assert set(ids) == gold_ids, path.name
    for r in rows:
        check_row(r)

    (OUT / path.name).write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(f"{path.name}: {kept} old + 3 new")
