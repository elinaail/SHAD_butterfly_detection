"""Run the three replacement photos with the original pinned configurations."""

import json
import os
import subprocess
from pathlib import Path

from run_models import read_jsonl
from export_predictions import export


def main():
    manifest = Path("inaturalist_dataset/replacements_2026_11_18/main.jsonl")
    rows = read_jsonl(manifest)
    assert len(rows) == 3 and len({r["photo_id"] for r in rows}) == 3
    folder = Path("results/replacements_2026_11_18")
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    for name in ["dino", "owlv2", "qwen"]:
        original = json.loads(Path(f"results/{name}/candidates.meta.json").read_text())
        python = os.environ.get(
            "QWEN_PYTHON_BIN" if name == "qwen" else "PYTHON_BIN",
            ".venv-qwen/bin/python" if name == "qwen" else ".venv-models/bin/python",
        )
        out = folder / name / "candidates.jsonl"
        subprocess.run(
            [
                python,
                "scripts/run_models.py",
                "--model",
                name,
                "--input",
                str(manifest),
                "--output",
                str(out),
                "--revision",
                original["resolved_revision"],
            ],
            env=env,
            check=True,
        )
        meta = json.loads(out.with_suffix(".meta.json").read_text())
        assert meta["resolved_revision"] == original["resolved_revision"]
        assert meta["versions"] == original["versions"], "Runtime versions changed"
        for k, value in original["config"].items():
            if k not in ["input_sha256", "revision_requested"]:
                assert meta["config"][k] == value, f"Configuration changed: {name}/{k}"
        predictions = read_jsonl(out)
        assert len(predictions) == 3 and {r["photo_id"] for r in predictions} == {
            r["photo_id"] for r in rows
        }
        for row in predictions:
            assert row["status"] == "ok" and not row.get("response_truncated")
            if name == "qwen":
                assert row["ended_with_eos"], "Incomplete Qwen answer"
        if name == "qwen":
            export(out, manifest, out.parent / "predictions.jsonl", 0.25)
        else:
            for threshold in [0.15, 0.25, 0.40]:
                export(
                    out,
                    manifest,
                    out.parent / f"predictions_t{threshold:.2f}.jsonl",
                    threshold,
                )
        print(
            f"Validated {name}: original settings, weights and runtime; 3 complete predictions",
            flush=True,
        )


if __name__ == "__main__":
    main()
