import matplotlib.pyplot as plt

epochs = list(range(1, 16))

forget_quality = [
    0.006094418258803505, 0.03956202584899502, 0.220541217580421, 0.8655265369450457,
    0.9238374197330625, 0.5452713464323318, 0.5452713464323318, 0.6284308022715471,
    0.2704743832803917, 0.46628639073563594, 0.46628639073563594, 0.2704743832803917,
    0.32811544409418575, 0.32811544409418575, 0.32811544409418575,
]

model_utility = [
    0.4228172942865552, 0.47208082337120727, 0.5127848191354466, 0.5442998768870129,
    0.5677633890697796, 0.5790601203554501, 0.5881470303952698, 0.5878505664465311,
    0.5787860446367419, 0.571378295625301, 0.5636635256905281, 0.5601507393715901,
    0.5582804498110736, 0.5566516120743941, 0.558415367577476,
]

forget_truth_ratio = [
    0.7283443067551117, 0.7075207248068532, 0.6780928164425578, 0.6528209109600545,
    0.6286265375076985, 0.6125438325724708, 0.5979299668344928, 0.5861354845762131,
    0.5756861898307933, 0.5733759433063313, 0.5709118074800206, 0.5640699276691336,
    0.5586288926437293, 0.558743133287973, 0.5567283482096733,
]

targets = {
    "forget_quality": 1.0,
    "model_utility": 0.6,
    "forget_truth_ratio": 0.64,
}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
fig.suptitle(
    "Retain95 Minimal Sweep (wd=0.0/0.1) vs Repro Targets (Retain forget05)",
    fontsize=14,
    fontweight="bold",
)

series_map = [
    (axes[0], forget_quality, "forget_quality"),
    (axes[1], model_utility, "model_utility"),
    (axes[2], forget_truth_ratio, "forget_truth_ratio"),
]

for ax, series, metric in series_map:
    ax.plot(
        epochs,
        series,
        marker="o",
        linewidth=2.2,
        color="#1f77b4",
        label="Sweep (wd=0.0 and wd=0.1; overlapping)",
    )
    target_label = f"Repro target = {targets[metric]:.2f}" if metric != "forget_quality" else "Repro target = 1.0"
    ax.axhline(
        targets[metric],
        linestyle="--",
        linewidth=2.0,
        color="#d62728",
        label=target_label,
    )
    ax.set_ylabel(metric)
    ax.legend(loc="best", frameon=True)

axes[2].set_xlabel("Epoch")
axes[2].set_xticks(epochs)

axes[0].annotate(
    f"best={forget_quality[4]:.4f} @e5",
    xy=(5, forget_quality[4]),
    xytext=(5.8, forget_quality[4] - 0.12),
    arrowprops=dict(arrowstyle="->", lw=1.2),
)
axes[1].annotate(
    f"best={model_utility[6]:.4f} @e7",
    xy=(7, model_utility[6]),
    xytext=(7.8, model_utility[6] - 0.035),
    arrowprops=dict(arrowstyle="->", lw=1.2),
)
axes[2].annotate(
    f"lowest={forget_truth_ratio[14]:.4f} @e15",
    xy=(15, forget_truth_ratio[14]),
    xytext=(9.8, forget_truth_ratio[14] + 0.04),
    arrowprops=dict(arrowstyle="->", lw=1.2),
)

plt.tight_layout(rect=(0, 0, 1, 0.97))
out_path = "assets/retain95_minimal_sweep_vs_repro_targets.png"
plt.savefig(out_path, dpi=220)
print(out_path)
