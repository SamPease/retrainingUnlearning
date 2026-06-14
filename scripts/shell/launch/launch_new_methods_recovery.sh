#!/usr/bin/env bash
# Launch recovery fine-tuning runs for all 6 new unlearning methods.
# Each method × 3 splits, for SEED (default 42).
# Baseline evals only run on seed=42 (pass --no-run-baseline-eval for other seeds).
#
# Usage: bash launch_new_methods_recovery_runs.sh [SEED]   (default: 42)
set -euo pipefail

SEED="${1:-42}"
SCRIPT="scripts/modal/train/recovery_hfbase.py"

TAGS=(
    "altpo_l5e5_b01_a1"
    "graddiff_l4e5_a5"
    "idkdpo_l5e5_b01_a1"
    "idknll_l5e5_a2"
    "simnpo_l5e5_b45_d1_g025"
    "undial_l1e4_b30_a2"
)

MODEL_IDS=(
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_AltPO_lr5e-05_beta0.1_alpha1_epoch10"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr4e-05_alpha5_epoch10"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr5e-05_beta0.1_alpha1_epoch10"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkNLL_lr5e-05_alpha2_epoch10"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10"
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_UNDIAL_lr0.0001_beta30_alpha2_epoch10"
)

if [[ "${SEED}" == "42" ]]; then
    BASELINE_FLAG="--run-baseline-eval"
else
    BASELINE_FLAG="--no-run-baseline-eval"
fi

echo "=== Launching new-method recovery runs (seed=${SEED}) ==="

for i in "${!TAGS[@]}"; do
    TAG="${TAGS[$i]}"
    MODEL="${MODEL_IDS[$i]}"
    for SPLIT in forget01 forget05 forget10; do
        echo "  ${TAG} → ${SPLIT}"
        conda run -n unlearning modal run --detach "${SCRIPT}" \
            --model-name-or-path "${MODEL}" \
            --model-tag "${TAG}" \
            --train-split "${SPLIT}" \
            ${BASELINE_FLAG} \
            --seed "${SEED}"
    done
done

echo ""
echo "All 18 jobs launched for seed=${SEED}."
