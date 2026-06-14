#!/usr/bin/env python3
"""
Download free-recovery eval results for the full HF model from the Modal volume.

Saves to:
  saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget01/
  saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget05/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SPLITS = ["forget10_minus_forget01", "forget10_minus_forget05"]
OUTPUT_ROOT = Path("saves/eval/tofu_Llama-3.2-1B-Instruct_full")


def _modal_get(remote: str, local: Path) -> bool:
    r = subprocess.run(
        ["conda", "run", "-n", "unlearning",
         "modal", "volume", "get", "--force",
         "open-unlearning-results", remote, str(local)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def main() -> None:
    ok = failed = 0
    for split in SPLITS:
        for fname in ["TOFU_SUMMARY.json", "TOFU_EVAL.json"]:
            remote = f"/eval/tofu_Llama-3.2-1B-Instruct_full/evals_{split}/{fname}"
            local = OUTPUT_ROOT / f"evals_{split}" / fname
            local.parent.mkdir(parents=True, exist_ok=True)
            if _modal_get(remote, local):
                print(f"  ✓  {split}/{fname}")
                ok += 1
            else:
                print(f"  ✗  {split}/{fname}  (not ready)")
                failed += 1

    print(f"\n{ok} downloaded, {failed} missing.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
