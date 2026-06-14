#!/usr/bin/env python3
"""
Download new RMU (rmu_l5_s10_lr5e5) recovery eval results from the Modal
results volume into tmp/rmu_new_recovery_eval_summaries/.

Run after all three recovery jobs (forget01 / forget05 / forget10) complete.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TAG = "rmu_l5_s10_lr5e5"
OUT_DIR = Path("tmp/rmu_new_recovery_eval_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_AND_EVALS: list[tuple[str, list[str]]] = [
    (
        "forget01",
        [
            "evals_forget01_taught",
            "evals_forget10_minus_forget01",
            "evals_retain90_utility",
        ],
    ),
    (
        "forget05",
        [
            "evals_forget05_taught",
            "evals_forget10_minus_forget05",
            "evals_retain90_utility",
        ],
    ),
    (
        "forget10",
        [
            "evals_forget10_taught",
            "evals_retain90_utility",
        ],
    ),
]


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
    for split, eval_names in SPLITS_AND_EVALS:
        baseline_slug = f"tofu_Llama-3.2-1B-Instruct_{TAG}_baseline_{split}"
        tuned_slug = (
            f"tofu_Llama-3.2-1B-Instruct_{TAG}_trl_{split}_lora_e20_lr2e4"
        )
        for slug, variant in [(baseline_slug, "baseline"), (tuned_slug, "tuned")]:
            for eval_name in eval_names:
                remote = f"/eval/{slug}/{eval_name}/TOFU_SUMMARY.json"
                local_name = f"{TAG}_{split}_{variant}_{eval_name}_TOFU_SUMMARY.json"
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
