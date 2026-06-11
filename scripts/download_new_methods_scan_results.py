#!/usr/bin/env python3
"""
Download TOFU_SUMMARY.json for all 36 new-methods scan candidates from the
Modal results volume into tmp/new_methods_scan_summaries/.

Run once all 6 method-scan Modal apps have stopped.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT_DIR = Path("tmp/new_methods_scan_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES: dict[str, list[str]] = {
    "AltPO":    ["altpo_l1e5_b01_a1", "altpo_l2e5_b01_a1", "altpo_l5e5_b01_a1",
                 "altpo_l1e5_b005_a2", "altpo_l2e5_b05_a1", "altpo_l5e5_b05_a5"],
    "GradDiff": ["graddiff_l1e5_a1", "graddiff_l2e5_a1", "graddiff_l3e5_a5",
                 "graddiff_l4e5_a5", "graddiff_l5e5_a2", "graddiff_l5e5_a10"],
    "IdkDPO":   ["idkdpo_l1e5_b01_a1", "idkdpo_l2e5_b01_a1", "idkdpo_l5e5_b01_a1",
                 "idkdpo_l1e5_b005_a2", "idkdpo_l2e5_b05_a1", "idkdpo_l5e5_b05_a5"],
    "IdkNLL":   ["idknll_l1e5_a1", "idknll_l2e5_a1", "idknll_l3e5_a5",
                 "idknll_l4e5_a5", "idknll_l5e5_a2", "idknll_l5e5_a10"],
    "SimNPO":   ["simnpo_l1e5_b35_d0_g025", "simnpo_l2e5_b35_d0_g025",
                 "simnpo_l5e5_b35_d0_g025", "simnpo_l1e5_b45_d1_g0125",
                 "simnpo_l2e5_b45_d1_g025", "simnpo_l5e5_b45_d1_g025"],
    "UNDIAL":   ["undial_l1e5_b10_a1", "undial_l1e4_b10_a1", "undial_l1e4_b30_a2",
                 "undial_l3e4_b3_a1", "undial_l3e4_b10_a1", "undial_l3e4_b30_a5"],
}


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
    for method, labels in CANDIDATES.items():
        for label in labels:
            remote = f"/eval/scan/{label}/TOFU_SUMMARY.json"
            local = OUT_DIR / f"{label}_TOFU_SUMMARY.json"
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
