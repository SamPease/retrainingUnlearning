#!/usr/bin/env python3
"""
Download recovery eval results for the 6 new unlearning methods (seed=42)
from the Modal results volume.

Covers: AltPO, GradDiff, IdkDPO, IdkNLL, SimNPO, UNDIAL × forget01/05/10.
Downloads both baseline evals and post-training tuned evals.

Usage: python scripts/download_new_methods_recovery_results.py

Output dir: tmp/new_methods_recovery_eval_summaries/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("tmp/new_methods_recovery_eval_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (model_tag, hf_model_id, eval_splits_with_dirs)
METHODS: list[tuple[str, list[tuple[str, list[str]]]]] = [
    ("altpo_l5e5_b01_a1", [
        ("forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
        ("forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
        ("forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    ]),
    ("graddiff_l4e5_a5", [
        ("forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
        ("forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
        ("forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    ]),
    ("idkdpo_l5e5_b01_a1", [
        ("forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
        ("forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
        ("forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    ]),
    ("idknll_l5e5_a2", [
        ("forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
        ("forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
        ("forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    ]),
    ("simnpo_l5e5_b45_d1_g025", [
        ("forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
        ("forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
        ("forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    ]),
    ("undial_l1e4_b30_a2", [
        ("forget01", ["evals_forget01_taught", "evals_forget10_minus_forget01", "evals_retain90_utility"]),
        ("forget05", ["evals_forget05_taught", "evals_forget10_minus_forget05", "evals_retain90_utility"]),
        ("forget10", ["evals_forget10_taught", "evals_retain90_utility"]),
    ]),
]


def _finetune_slug(model_tag: str, train_split: str, seed: int = 42) -> str:
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    return f"tofu_Llama-3.2-1B-Instruct_{model_tag}_trl_{train_split}_lora_e20_lr2e4{seed_suffix}"


def _modal_get(remote: str, local: Path) -> bool:
    r = subprocess.run(
        ["conda", "run", "-n", "unlearning",
         "modal", "volume", "get", "--force",
         "open-unlearning-results", remote, str(local)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    ok = failed = 0

    for model_tag, splits in METHODS:
        for train_split, eval_dirs in splits:
            slug = _finetune_slug(model_tag, train_split, seed)

            # tuned eval
            for eval_dir in eval_dirs:
                remote = f"/eval/{slug}/{eval_dir}/TOFU_SUMMARY.json"
                local = OUT_DIR / f"{model_tag}_{train_split}_tuned_{eval_dir}_TOFU_SUMMARY.json"
                if _modal_get(remote, local):
                    print(f"  ✓  tuned  {model_tag} {train_split} {eval_dir}")
                    ok += 1
                else:
                    print(f"  ✗  tuned  {model_tag} {train_split} {eval_dir}  (missing)")
                    failed += 1

            # baseline eval (seed=42 only)
            if seed == 42:
                baseline_root = f"tofu_Llama-3.2-1B-Instruct_{model_tag}_baseline_{train_split}"
                for eval_dir in eval_dirs:
                    remote = f"/eval/{baseline_root}/{eval_dir}/TOFU_SUMMARY.json"
                    local = OUT_DIR / f"{model_tag}_{train_split}_baseline_{eval_dir}_TOFU_SUMMARY.json"
                    if _modal_get(remote, local):
                        print(f"  ✓  base  {model_tag} {train_split} {eval_dir}")
                        ok += 1
                    else:
                        print(f"  ✗  base  {model_tag} {train_split} {eval_dir}  (missing)")
                        failed += 1

    print(f"\n{ok} downloaded, {failed} missing.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
