from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def extract_epoch_metrics(trainer_state_path: Path) -> list[tuple[float, float, float, float, int]]:
    state = json.loads(trainer_state_path.read_text())
    rows: list[tuple[float, float, float, float, int]] = []
    for item in state.get("log_history", []):
        if all(
            key in item
            for key in ("forget_quality", "model_utility", "forget_truth_ratio", "epoch", "step")
        ):
            rows.append(
                (
                    float(item["epoch"]),
                    float(item["forget_quality"]),
                    float(item["model_utility"]),
                    float(item["forget_truth_ratio"]),
                    int(item["step"]),
                )
            )
    rows.sort(key=lambda x: x[0])
    return rows


def main() -> None:
    full_state = Path("tmp/modal_epoch_eval/full/trainer_state.json")
    retain_state = Path("tmp/modal_epoch_eval/retain95/trainer_state.json")

    full = extract_epoch_metrics(full_state)
    retain = extract_epoch_metrics(retain_state)

    if not full or not retain:
        raise RuntimeError("Missing epoch metric rows in trainer_state files")

    # repro.md references for Llama-3.2-1B-Instruct
    refs = {
        "full_finetuned": {
            "forget_quality": 1.66e-21,
            "model_utility": 0.6,
            "forget_truth_ratio": 0.48,
        },
        "full_retain": {
            "forget_quality": 1.0,
            "model_utility": 0.59,
            "forget_truth_ratio": 0.63,
        },
        "retain95_finetuned": {
            "forget_quality": 1.33e-13,
            "model_utility": 0.6,
            "forget_truth_ratio": 0.47,
        },
        "retain95_retain": {
            "forget_quality": 1.0,
            "model_utility": 0.6,
            "forget_truth_ratio": 0.64,
        },
    }

    metric_idx = {"forget_quality": 1, "model_utility": 2, "forget_truth_ratio": 3}
    metrics = ["forget_quality", "model_utility", "forget_truth_ratio"]

    full_epochs = [row[0] for row in full]
    retain_epochs = [row[0] for row in retain]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(
        "TOFU 1-GPU Repro-Style Trial: Per-Epoch Minimal Metrics vs repro.md References",
        fontsize=14,
        y=1.02,
    )

    for row_idx, metric in enumerate(metrics):
        left = axes[row_idx][0]
        left.plot(
            full_epochs,
            [row[metric_idx[metric]] for row in full],
            marker="o",
            linewidth=1.8,
            label="full run (forget10 eval)",
        )
        left.axhline(
            refs["full_finetuned"][metric],
            linestyle="--",
            linewidth=1.2,
            label="repro Finetuned forget10",
        )
        left.axhline(
            refs["full_retain"][metric],
            linestyle=":",
            linewidth=1.2,
            label="repro Retain forget10",
        )
        left.set_title(f"Full training - {metric}")
        left.set_xlabel("Epoch")
        left.set_ylabel(metric)
        left.grid(alpha=0.3)
        if metric == "forget_quality":
            left.set_yscale("log")
        left.legend(fontsize=8)

        right = axes[row_idx][1]
        right.plot(
            retain_epochs,
            [row[metric_idx[metric]] for row in retain],
            marker="o",
            linewidth=1.8,
            color="tab:orange",
            label="retain95 run (forget05 eval)",
        )
        right.axhline(
            refs["retain95_finetuned"][metric],
            linestyle="--",
            linewidth=1.2,
            label="repro Finetuned forget05",
        )
        right.axhline(
            refs["retain95_retain"][metric],
            linestyle=":",
            linewidth=1.2,
            label="repro Retain forget05",
        )
        right.set_title(f"Retain95 training - {metric}")
        right.set_xlabel("Epoch")
        right.set_ylabel(metric)
        right.grid(alpha=0.3)
        if metric == "forget_quality":
            right.set_yscale("log")
        right.legend(fontsize=8)

    output_path = Path("experiments/reports/tofu_repro1gpu_min_eval_vs_repro_refs.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    full_last = full[-1]
    retain_last = retain[-1]

    summary = {
        "output": str(output_path),
        "full_points": len(full),
        "retain95_points": len(retain),
        "full_last": {
            "epoch": full_last[0],
            "step": full_last[4],
            "forget_quality": full_last[1],
            "model_utility": full_last[2],
            "forget_truth_ratio": full_last[3],
        },
        "retain95_last": {
            "epoch": retain_last[0],
            "step": retain_last[4],
            "forget_quality": retain_last[1],
            "model_utility": retain_last[2],
            "forget_truth_ratio": retain_last[3],
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
