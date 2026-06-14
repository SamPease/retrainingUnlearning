#!/usr/bin/env python3
"""Generate markdown reports for TOFU runs with per-epoch eval summaries.

This script scans finetune run directories for:
  checkpoint-*/evals/TOFU_SUMMARY.json
and writes a markdown report with tables and Mermaid charts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "forget_Q_A_Prob",
    "forget_truth_ratio",
    "extraction_strength",
    "retain_Q_A_Prob",
]


@dataclass
class CheckpointEval:
    global_step: int
    epoch: float | None
    metrics: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_step_to_epoch(run_dir: Path) -> dict[int, float]:
    state_path = run_dir / "trainer_state.json"
    if not state_path.exists():
        return {}

    state = _load_json(state_path)
    mapping: dict[int, float] = {}
    for record in state.get("log_history", []):
        step = record.get("step")
        epoch = record.get("epoch")
        if isinstance(step, int) and isinstance(epoch, (int, float)):
            mapping[step] = float(epoch)
    return mapping


def _collect_checkpoint_evals(run_dir: Path) -> list[CheckpointEval]:
    step_to_epoch = _read_step_to_epoch(run_dir)
    checkpoint_paths = sorted(run_dir.glob("checkpoint-*/evals/TOFU_SUMMARY.json"))

    evals: list[CheckpointEval] = []
    for summary_path in checkpoint_paths:
        checkpoint_dir = summary_path.parents[1]
        name = checkpoint_dir.name
        try:
            step = int(name.split("checkpoint-")[-1])
        except ValueError:
            continue
        metrics = _load_json(summary_path)
        evals.append(
            CheckpointEval(
                global_step=step,
                epoch=step_to_epoch.get(step),
                metrics=metrics,
            )
        )

    evals.sort(key=lambda x: x.global_step)
    return evals


def _fmt_float(value: Any, ndigits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{ndigits}g}"
    return str(value)


def _mermaid_chart(metric_name: str, xs: list[str], ys: list[float]) -> str:
    if not ys:
        return ""

    y_min = min(ys)
    y_max = max(ys)
    if y_min == y_max:
        margin = 0.1 if y_min == 0 else abs(y_min) * 0.1
        y_min -= margin
        y_max += margin

    x_labels = ", ".join(f'"{x}"' for x in xs)
    y_values = ", ".join(f"{y:.6g}" for y in ys)

    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            f'    title "{metric_name} over eval epochs"',
            f"    x-axis [{x_labels}]",
            f'    y-axis "{metric_name}" {y_min:.6g} --> {y_max:.6g}',
            f"    line [{y_values}]",
            "```",
        ]
    )


def _run_section(run_name: str, run_dir: Path, metrics: list[str]) -> str:
    evals = _collect_checkpoint_evals(run_dir)
    lines: list[str] = []

    lines.append(f"## Run: {run_name}")
    lines.append("")
    lines.append(f"- Run directory: {run_dir}")
    lines.append(f"- Checkpoint evals found: {len(evals)}")
    lines.append("")

    if not evals:
        lines.append("No checkpoint-level TOFU eval summaries found.")
        lines.append("")
        return "\n".join(lines)

    # Table
    header = ["eval_idx", "epoch", "global_step"] + metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for idx, item in enumerate(evals):
        epoch_str = _fmt_float(item.epoch, ndigits=4) if item.epoch is not None else ""
        row = [str(idx + 1), epoch_str, str(item.global_step)]
        row.extend(_fmt_float(item.metrics.get(m)) for m in metrics)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("### Metric Charts")
    lines.append("")

    x_axis = [str(idx + 1) for idx in range(len(evals))]
    for metric in metrics:
        y_vals: list[float] = []
        for item in evals:
            value = item.metrics.get(metric)
            if isinstance(value, (int, float)):
                y_vals.append(float(value))
        if len(y_vals) != len(evals):
            lines.append(f"- Skipping chart for {metric}: missing values in one or more evals.")
            continue
        lines.append(f"#### {metric}")
        lines.append("")
        lines.append(_mermaid_chart(metric, x_axis, y_vals))
        lines.append("")

    return "\n".join(lines)


def _discover_runs(finetune_root: Path) -> list[str]:
    runs: list[str] = []
    if not finetune_root.exists():
        return runs

    for run_dir in sorted(p for p in finetune_root.iterdir() if p.is_dir()):
        if any(run_dir.glob("checkpoint-*/evals/TOFU_SUMMARY.json")):
            runs.append(run_dir.name)
    return runs


def generate_report(
    finetune_root: Path,
    output_path: Path,
    run_names: list[str],
    metrics: list[str],
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    lines: list[str] = [
        "# TOFU Training Runs",
        "",
        f"Generated at: {timestamp}",
        "",
        "This report is generated from checkpoint-level TOFU summary files:",
        "- checkpoint-*/evals/TOFU_SUMMARY.json",
        "",
        "Run lightweight per-epoch eval training:",
        "```bash",
        "python src/train.py experiment=finetune/tofu/light_eval.yaml task_name=<RUN_NAME>",
        "```",
        "",
        "Regenerate this report:",
        "```bash",
        "python scripts/generate_tofu_epoch_report.py --output experiments/reports/training-runs.md",
        "```",
        "",
    ]

    if not run_names:
        lines.append("No runs with checkpoint-level TOFU eval summaries were found.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    for run_name in run_names:
        run_dir = finetune_root / run_name
        lines.append(_run_section(run_name, run_dir, metrics))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finetune-root",
        default="saves/finetune",
        help="Root directory containing finetune runs.",
    )
    parser.add_argument(
        "--output",
        default="experiments/reports/training-runs.md",
        help="Output markdown file path.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run directory names under finetune-root. If omitted, auto-discovers runs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric names to render in table/charts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finetune_root = Path(args.finetune_root)
    output_path = Path(args.output)

    run_names = args.runs if args.runs is not None else _discover_runs(finetune_root)
    generate_report(
        finetune_root=finetune_root,
        output_path=output_path,
        run_names=run_names,
        metrics=args.metrics,
    )
    print(f"Wrote {output_path} for {len(run_names)} run(s)")


if __name__ == "__main__":
    main()
