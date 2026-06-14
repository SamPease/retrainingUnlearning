#!/usr/bin/env python3
"""Download taught-set and retain90-utility evals for all seed-robustness conditions.

Each condition has three eval dirs on the volume:
  evals_{split}_taught/      -- taught-set performance
  evals_forget10_minus_{split}/  -- free-recovery (may already be local)
  evals_retain90_utility/    -- retain90 utility after tuning

Seed-42 dirs use _lora_e20_lr2e4 suffix (no explicit seed in name).
Seeds 123/456 use _lora_e20_lr2e4_seed{N}.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path("saves/eval")
TAGS = {"NPO": "npo_forget10", "RMU": "rmu_l5_s10_lr5e5", "R90": "retain90"}
SPLITS = ["forget01", "forget05"]
SEED_SUFFIXES = {"42": "", "123": "_seed123", "456": "_seed456"}


def _modal_get(remote: str, local: Path) -> bool:
    r = subprocess.run(
        ["conda", "run", "-n", "unlearning",
         "modal", "volume", "get", "--force",
         "open-unlearning-results", remote, str(local)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def main() -> None:
    ok = failed = skipped = 0
    for mdl, tag in TAGS.items():
        for split in SPLITS:
            for seed, suffix in SEED_SUFFIXES.items():
                dir_name = f"tofu_Llama-3.2-1B-Instruct_{tag}_trl_{split}_lora_e20_lr2e4{suffix}"
                eval_dirs = [
                    f"evals_{split}_taught",
                    f"evals_forget10_minus_{split}",
                    "evals_retain90_utility",
                ]
                for eval_dir in eval_dirs:
                    remote = f"/eval/{dir_name}/{eval_dir}/TOFU_SUMMARY.json"
                    local_dir = EVAL_DIR / dir_name / eval_dir
                    local = local_dir / "TOFU_SUMMARY.json"
                    if local.exists():
                        skipped += 1
                        continue
                    local_dir.mkdir(parents=True, exist_ok=True)
                    label = f"{mdl} seed{seed} {split} {eval_dir}"
                    if _modal_get(remote, local):
                        print(f"  ok  {label}")
                        ok += 1
                    else:
                        print(f"  !!  {label}  MISSING")
                        failed += 1

    print(f"\n{ok} downloaded, {skipped} already present, {failed} missing.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
