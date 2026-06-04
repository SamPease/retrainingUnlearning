#!/usr/bin/env python3
"""Collect and compare NPO/RMU recovery experiments against retain90 trials.

Reads TOFU summary JSON files from local paths (after downloading from Modal volume)
and writes a compact markdown comparison report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUITES = [
    "taught",
    "free_recovery",
    "utility",
]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _metrics_line(d: dict | None) -> str:
    if d is None:
        return "missing"
    keys = [
        "forget_quality",
        "model_utility",
        "forget_truth_ratio",
        "retain90_utility",
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_Truth_Ratio",
    ]
    parts = [f"{k}={d[k]}" for k in keys if k in d]
    return ", ".join(parts) if parts else "no tracked keys"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hfbase recovery experiment summaries")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("tmp/hfbase_recovery_eval_summaries"),
    )
    parser.add_argument(
        "--retain-root",
        type=Path,
        default=Path("tmp/new_recovery_eval_summaries"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/hfbase_recovery_compare.md"),
    )
    args = parser.parse_args()

    runs = [
        ("npo_forget10", "forget01", "forget10_minus_forget01"),
        ("npo_forget10", "forget05", "forget10_minus_forget05"),
        ("npo_forget10", "forget10", ""),
        ("rmu_forget10", "forget01", "forget10_minus_forget01"),
        ("rmu_forget10", "forget05", "forget10_minus_forget05"),
        ("rmu_forget10", "forget10", ""),
    ]

    retain_refs = {
        "forget01:taught": args.retain_root / "forget01_taught_TOFU_SUMMARY.json",
        "forget01:free_recovery": args.retain_root / "forget01_free_recovery_TOFU_SUMMARY.json",
        "forget01:utility": args.retain_root / "forget01_retain90_utility_TOFU_SUMMARY.json",
        "forget05:taught": args.retain_root / "forget05_taught_TOFU_SUMMARY.json",
        "forget05:free_recovery": args.retain_root / "forget05_free_recovery_TOFU_SUMMARY.json",
        "forget05:utility": args.retain_root / "forget05_retain90_utility_TOFU_SUMMARY.json",
    }

    lines = [
        "# HFBase Recovery Comparison",
        "",
        "Generated from local TOFU_SUMMARY artifacts.",
        "",
    ]

    for model_tag, split, free_name in runs:
        lines.append(f"## {model_tag} on {split}")
        suite_paths = {
            "taught": (
                args.artifact_root
                / f"{model_tag}_{split}_baseline_evals_{split}_taught_TOFU_SUMMARY.json",
                args.artifact_root
                / f"{model_tag}_{split}_tuned_evals_{split}_taught_TOFU_SUMMARY.json",
            ),
            "utility": (
                args.artifact_root
                / f"{model_tag}_{split}_baseline_evals_retain90_utility_TOFU_SUMMARY.json",
                args.artifact_root
                / f"{model_tag}_{split}_tuned_evals_retain90_utility_TOFU_SUMMARY.json",
            ),
        }
        if free_name:
            suite_paths["free_recovery"] = (
                args.artifact_root
                / f"{model_tag}_{split}_baseline_evals_{free_name}_TOFU_SUMMARY.json",
                args.artifact_root
                / f"{model_tag}_{split}_tuned_evals_{free_name}_TOFU_SUMMARY.json",
            )

        for suite, (b_path, t_path) in suite_paths.items():
            b = _load_json(b_path)
            t = _load_json(t_path)
            r_path = retain_refs.get(f"{split}:{suite}")
            r = _load_json(r_path) if r_path is not None else None
            lines.append(f"- {suite}")
            lines.append(f"  - baseline: {_metrics_line(b)}")
            lines.append(f"  - tuned: {_metrics_line(t)}")
            lines.append(f"  - retain_ref: {_metrics_line(r)}")

        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
