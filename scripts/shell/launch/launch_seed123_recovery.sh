#!/usr/bin/env bash
# Launch all 9 recovery fine-tuning runs (3 models × 3 splits) for a given seed.
# Each is launched as a separate detached Modal job to avoid fan-out kill issues.
# Baseline evals are skipped (--no-run-baseline-eval) because they are
# deterministic — the same HF checkpoint produces the same baseline metrics.
#
# Usage: bash launch_seed123_recovery_runs.sh [SEED]   (default: 123)
set -euo pipefail

SEED="${1:-123}"

NPO_MODEL="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10"
RMU_MODEL="open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10"
HFBASE_SCRIPT="scripts/modal/train/recovery_hfbase.py"
RETAIN90_SCRIPT="scripts/modal/train/recovery_retain90.py"

echo "=== Launching seed=${SEED} recovery runs ==="

# ── retain90 ──────────────────────────────────────────────────────────────────
for SPLIT in forget01 forget05 forget10; do
    echo "  retain90 → ${SPLIT}"
    conda run -n unlearning modal run --detach "${RETAIN90_SCRIPT}" \
        --train-split "${SPLIT}" \
        --seed "${SEED}"
done

# ── NPO ───────────────────────────────────────────────────────────────────────
for SPLIT in forget01 forget05 forget10; do
    echo "  npo_forget10 → ${SPLIT}"
    conda run -n unlearning modal run --detach "${HFBASE_SCRIPT}" \
        --model-name-or-path "${NPO_MODEL}" \
        --model-tag npo_forget10 \
        --train-split "${SPLIT}" \
        --no-run-baseline-eval \
        --seed "${SEED}"
done

# ── RMU★ ──────────────────────────────────────────────────────────────────────
for SPLIT in forget01 forget05 forget10; do
    echo "  rmu_l5_s10_lr5e5 → ${SPLIT}"
    conda run -n unlearning modal run --detach "${HFBASE_SCRIPT}" \
        --model-name-or-path "${RMU_MODEL}" \
        --model-tag rmu_l5_s10_lr5e5 \
        --train-split "${SPLIT}" \
        --no-run-baseline-eval \
        --seed "${SEED}"
done

echo ""
echo "All 9 jobs launched. Monitor with:"
echo "  conda run -n unlearning modal app list --json | python3 -c \\"
echo "    \"import json,sys; [print(a['App ID'], a['State'], a.get('Description','')) for a in json.load(sys.stdin)]\""
