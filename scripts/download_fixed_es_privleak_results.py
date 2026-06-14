"""
Download corrected free-recovery TOFU_SUMMARY.json files (ES + privleak on correct subset).
Overwrites previous evals_forget10_minus_forgetN dirs with fixed values.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path("saves/eval")

CORE_CONDITIONS = [
    ("retain90",          ["", "_seed123", "_seed456"]),
    ("npo_forget10",      ["", "_seed123", "_seed456"]),
    ("rmu_l5_s10_lr5e5",  ["", "_seed123", "_seed456"]),
]
NEW_CONDITIONS = [
    ("altpo_l5e5_b01_a1",          [""]),
    ("graddiff_l4e5_a5",           [""]),
    ("idkdpo_l5e5_b01_a1",         [""]),
    ("idknll_l5e5_a2",             [""]),
    ("simnpo_l5e5_b45_d1_g025",    [""]),
    ("undial_l1e4_b30_a2",         [""]),
]
SPLITS = ["forget01", "forget05"]


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

    all_conditions = CORE_CONDITIONS + NEW_CONDITIONS
    for tag, seed_sfxs in all_conditions:
        for split in SPLITS:
            for sfx in seed_sfxs:
                dn = f"tofu_Llama-3.2-1B-Instruct_{tag}_trl_{split}_lora_e20_lr2e4{sfx}"
                free_rec = f"evals_forget10_minus_{split}"
                local_dir = EVAL_DIR / dn / free_rec
                local = local_dir / "TOFU_SUMMARY.json"
                remote = f"/eval/{dn}/{free_rec}/TOFU_SUMMARY.json"
                local_dir.mkdir(parents=True, exist_ok=True)
                label = f"{dn}/{free_rec}"
                if _modal_get(remote, local):
                    print(f"  ok  {label}")
                    ok += 1
                else:
                    print(f"  !!  {label}  MISSING")
                    failed += 1

    print(f"\n{ok} downloaded, {failed} missing.")
    if failed:
        print("Some evals may still be running. Re-run this script after all jobs complete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
