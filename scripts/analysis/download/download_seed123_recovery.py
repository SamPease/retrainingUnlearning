#!/usr/bin/env python3
"""
Download recovery eval results for a given seed from the Modal results volume.
Covers retain90, npo_forget10, and rmu_l5_s10_lr5e5 across forget01/05/10.

Usage: python download_seed123_recovery_results.py [SEED]   (default: 123)

Output dir: tmp/seed{SEED}_recovery_eval_summaries/
Naming: {model_tag}_{split}_tuned_{eval_name}_TOFU_SUMMARY.json
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 123
OUT_DIR = Path(f"tmp/seed{SEED}_recovery_eval_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS: list[tuple[str, str, list[str]]] = [
    # (model_tag, train_split, [eval_dir_names])
    # retain90
    ("retain90", "forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
    ("retain90", "forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
    ("retain90", "forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    # NPO
    ("npo_forget10", "forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
    ("npo_forget10", "forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
    ("npo_forget10", "forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    # RMU★
    ("rmu_l5_s10_lr5e5", "forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
    ("rmu_l5_s10_lr5e5", "forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
    ("rmu_l5_s10_lr5e5", "forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
]


def _finetune_dir(model_tag: str, train_split: str, seed: int) -> str:
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    if model_tag == "retain90":
        return f"tofu_Llama-3.2-1B-Instruct_retain90_trl_{train_split}_lora_e20_lr2e4{seed_suffix}"
    return f"tofu_Llama-3.2-1B-Instruct_{model_tag}_trl_{train_split}_lora_e20_lr2e4{seed_suffix}"


def _modal_get(remote: str, local: Path) -> bool:
    result = subprocess.run(
        [
            "conda", "run", "-n", "unlearning",
            "modal", "volume", "get", "--force",
            "open-unlearning-results",
            remote,
            str(local),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main() -> None:
    ok = failed = 0
    for model_tag, split, eval_names in JOBS:
        finetune_slug = _finetune_dir(model_tag, split, SEED)
        eval_slug = finetune_slug  # eval root mirrors finetune slug
        for eval_name in eval_names:
            remote = f"/eval/{eval_slug}/{eval_name}/TOFU_SUMMARY.json"
            local_name = f"{model_tag}_{split}_tuned_{eval_name}_TOFU_SUMMARY.json"
            local_path = OUT_DIR / local_name
            if _modal_get(remote, local_path):
                print(f"  ✓  {local_name}")
                ok += 1
            else:
                print(f"  ✗  {local_name}  (not available yet)")
                failed += 1

    print(f"\n{ok} downloaded, {failed} missing.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
