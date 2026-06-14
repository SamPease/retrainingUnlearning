#!/usr/bin/env python3
"""
Download consistent baseline eval results for the Full model from the Modal volume.
Saves to saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_consistent/

Retain90 baseline already has consistent metrics locally at:
  saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/evals_forget10_retainref/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

JOBS = [
    (
        "/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_consistent",
        Path("saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_consistent"),
    ),
]


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
    for remote_dir, local_dir in JOBS:
        local_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["TOFU_SUMMARY.json", "TOFU_EVAL.json"]:
            remote = f"{remote_dir}/{fname}"
            local = local_dir / fname
            label = f"{local_dir.parent.name}/{local_dir.name}/{fname}"
            if _modal_get(remote, local):
                print(f"  ✓  {label}")
                ok += 1
            else:
                print(f"  ✗  {label}  (not ready)")
                failed += 1

    print(f"\n{ok} downloaded, {failed} missing.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
