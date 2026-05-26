import json
from pathlib import Path

out = Path("tmp/modal_epoch_eval/epoch_tables.md")
runs = {
    "full": Path("tmp/modal_epoch_eval/full/trainer_state.json"),
    "retain95": Path("tmp/modal_epoch_eval/retain95/trainer_state.json"),
}

lines = []
for name, p in runs.items():
    d = json.loads(p.read_text())
    rows = []
    for x in d.get("log_history", []):
        if all(
            k in x
            for k in (
                "epoch",
                "step",
                "forget_quality",
                "model_utility",
                "forget_truth_ratio",
            )
        ):
            rows.append(
                (
                    float(x["epoch"]),
                    int(x["step"]),
                    x["forget_quality"],
                    x["model_utility"],
                    x["forget_truth_ratio"],
                )
            )

    rows.sort(key=lambda t: (t[0], t[1]))
    title = "Full run" if name == "full" else "Retain95 run"
    lines.append(f"### {title} - full epoch table")
    lines.append("")
    lines.append("| epoch | step | forget_quality | model_utility | forget_truth_ratio |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
    lines.append("")

out.write_text("\n".join(lines))
print(out)
