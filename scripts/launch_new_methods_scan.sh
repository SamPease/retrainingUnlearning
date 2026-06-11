#!/usr/bin/env bash
# Launch one scan job per new unlearning method (AltPO, GradDiff, IdkDPO,
# IdkNLL, SimNPO, UNDIAL). Each job evaluates 6 candidates sequentially on a
# single GPU. All 6 jobs run in parallel on separate Modal workers.
set -euo pipefail

SCRIPT="scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py"

for METHOD in AltPO GradDiff IdkDPO IdkNLL SimNPO UNDIAL; do
    echo "  launching scan: ${METHOD}"
    conda run -n unlearning modal run --detach "${SCRIPT}" --method "${METHOD}"
done

echo ""
echo "All 6 method scans launched."
