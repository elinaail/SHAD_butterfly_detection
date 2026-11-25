#!/usr/bin/env bash
# Run from the project root. Safe to rerun: successful photos are skipped.
set -u
export PYTORCH_ENABLE_MPS_FALLBACK=1
python_bin="${PYTHON_BIN:-python3}"
qwen_python_bin="${QWEN_PYTHON_BIN:-$python_bin}"
failed=0
for model in dino owlv2 qwen; do
  model_python_bin="$python_bin"
  if [ "$model" = qwen ]; then
    model_python_bin="$qwen_python_bin"
  fi
  if ! "$model_python_bin" scripts/run_models.py --model "$model" --output "results/$model/candidates.jsonl"; then
    failed=1
    continue
  fi
  if [ "$model" = qwen ]; then
    if "$qwen_python_bin" scripts/complete_qwen.py; then
      "$python_bin" scripts/export_predictions.py --source "results/$model/completed_candidates.jsonl" --output "results/$model/predictions.jsonl" || failed=1
    else
      failed=1
    fi
  else
    for threshold in 0.15 0.25 0.40; do
      "$python_bin" scripts/export_predictions.py --source "results/$model/candidates.jsonl" --output "results/$model/predictions_t${threshold}.jsonl" --threshold "$threshold" || failed=1
    done
  fi
done
exit "$failed"
