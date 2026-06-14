#!/usr/bin/env bash
# Re-run free-recovery evals with corrected extraction_strength and privleak overrides.
# Skips training (--skip-train); only re-evaluates existing checkpoints.
# Covers all 9 methods × 2 splits (forget01, forget05) × all seeds.
set -euo pipefail

HFBASE="scripts/modal_tofu_finetune_trl_hfbase_forgetx_recovery_llama32_1b.py"
R90="scripts/modal_tofu_finetune_trl_retain90_forget01_forget05_recovery_llama32_1b.py"

echo "=== Core methods (3 seeds each) ==="

for SPLIT in forget01 forget05; do
    for SEED in 42 123 456; do
        echo "  Retain90 seed${SEED} ${SPLIT}"
        conda run -n unlearning modal run --detach "${R90}" \
            --train-split "${SPLIT}" --skip-train --seed "${SEED}"
    done
done

for SPLIT in forget01 forget05; do
    for SEED in 42 123 456; do
        echo "  NPO seed${SEED} ${SPLIT}"
        conda run -n unlearning modal run --detach "${HFBASE}" \
            --model-name-or-path "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10" \
            --model-tag "npo_forget10" \
            --train-split "${SPLIT}" --skip-train --seed "${SEED}"
    done
done

for SPLIT in forget01 forget05; do
    for SEED in 42 123 456; do
        echo "  RMU★ seed${SEED} ${SPLIT}"
        conda run -n unlearning modal run --detach "${HFBASE}" \
            --model-name-or-path "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10" \
            --model-tag "rmu_l5_s10_lr5e5" \
            --train-split "${SPLIT}" --skip-train --seed "${SEED}"
    done
done

echo ""
echo "=== New methods (seed 42 only) ==="

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

for i in "${!TAGS[@]}"; do
    TAG="${TAGS[$i]}"
    MODEL="${MODEL_IDS[$i]}"
    for SPLIT in forget01 forget05; do
        echo "  ${TAG} seed42 ${SPLIT}"
        conda run -n unlearning modal run --detach "${HFBASE}" \
            --model-name-or-path "${MODEL}" \
            --model-tag "${TAG}" \
            --train-split "${SPLIT}" \
            --skip-train \
            --no-run-baseline-eval \
            --seed 42
    done
done

echo ""
echo "All reeval jobs launched (30 jobs: 9 core-seed + 18 new-method)."
echo "Wait for completion, then run:"
echo "  conda run -n unlearning python scripts/download_fixed_es_privleak_results.py"
