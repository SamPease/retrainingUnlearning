"""
Chart 6 — Transfer-rate scaling from forget01 to forget05.

Two panels:
  Left:  Free-recovery QAP (baseline vs tuned) for NPO and RMU★,
         showing how the absolute free-recall changes with training-set size.
  Right: Transfer rate (free-rec gain / taught gain) for NPO and RMU★
         at forget01 and forget05 — the diverging trend is the key finding.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "assets/recovery_chart6_transfer_rate_scaling.png"

C_NPO = "#2196F3"
C_RMU = "#FF9800"

# ── exact values from JSON summaries ─────────────────────────────────────────
# (base_free, tuned_free, base_taught, tuned_taught)
DATA = {
    "NPO": {
        "forget01": (0.2057, 0.4576, 0.2261, 0.9294),
        "forget05": (0.2086, 0.3159, 0.2069, 0.9059),
    },
    "RMU★": {
        "forget01": (0.0011, 0.2600, 0.0000, 0.8968),
        "forget05": (0.0017, 0.3662, 0.0003, 0.8895),
    },
}

splits = ["forget01", "forget05"]
split_labels = ["Teach 1%\n(forget01)", "Teach 5%\n(forget05)"]
methods = ["NPO", "RMU★"]
colors  = {"NPO": C_NPO, "RMU★": C_RMU}

def transfer(base_free, tuned_free, base_taught, tuned_taught):
    return (tuned_free - base_free) / (tuned_taught - base_taught)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.5))

# ── LEFT PANEL: free-recovery QAP baseline vs tuned ──────────────────────────
n_methods = 2
n_splits  = 2
group_w   = 0.7
bar_w     = group_w / (n_methods * 2)   # baseline + tuned per method
x         = np.arange(n_splits)

for mi, method in enumerate(methods):
    c = colors[method]
    for si, split in enumerate(splits):
        bf, tf, _, _ = DATA[method][split]
        # baseline bar (hatched)
        offset_base  = (mi * 2)     * bar_w - group_w/2 + bar_w/2
        offset_tuned = (mi * 2 + 1) * bar_w - group_w/2 + bar_w/2
        ax_left.bar(si + offset_base,  bf, bar_w,
                    color=c, alpha=0.45, hatch="///", edgecolor="white", linewidth=0.5)
        ax_left.bar(si + offset_tuned, tf, bar_w,
                    color=c, alpha=0.90, edgecolor="white", linewidth=0.5)
        ax_left.text(si + offset_base,  bf + 0.008, f"{bf:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color=c)
        ax_left.text(si + offset_tuned, tf + 0.008, f"{tf:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color=c)

ax_left.set_xticks(x)
ax_left.set_xticklabels(split_labels, fontsize=10)
ax_left.set_ylim(0, 0.70)
ax_left.set_ylabel("forget_Q_A_Prob on free-recovery split", fontsize=9)
ax_left.set_title("Free-recovery QAP: baseline (hatched) → tuned (solid)", fontsize=10, fontweight="bold")

handles_left = [
    mpatches.Patch(color=C_NPO,  alpha=0.90, label="NPO tuned"),
    mpatches.Patch(color=C_NPO,  alpha=0.45, hatch="///", edgecolor="white", label="NPO baseline"),
    mpatches.Patch(color=C_RMU,  alpha=0.90, label="RMU★ tuned"),
    mpatches.Patch(color=C_RMU,  alpha=0.45, hatch="///", edgecolor="white", label="RMU★ baseline"),
]
ax_left.legend(handles=handles_left, fontsize=8, loc="upper left")

# ── RIGHT PANEL: transfer rate lines ─────────────────────────────────────────
xi = np.array([0, 1])
for method in methods:
    c = colors[method]
    rates = [
        transfer(*DATA[method][s]) for s in splits
    ]
    ax_right.plot(xi, rates, marker="o", markersize=9, linewidth=2.5,
                  color=c, label=method, zorder=5)
    for xi_, r in zip(xi, rates):
        ax_right.annotate(f"{r:.2%}", (xi_, r),
                          textcoords="offset points", xytext=(8, 4),
                          fontsize=9.5, color=c, fontweight="bold")

# annotate the diverging arrows
ax_right.annotate("", xy=(1, transfer(*DATA["RMU★"]["forget05"])),
                  xytext=(1, transfer(*DATA["RMU★"]["forget01"])),
                  arrowprops=dict(arrowstyle="->", color=C_RMU, lw=1.5, alpha=0.7))
ax_right.annotate("", xy=(1, transfer(*DATA["NPO"]["forget05"])),
                  xytext=(1, transfer(*DATA["NPO"]["forget01"])),
                  arrowprops=dict(arrowstyle="->", color=C_NPO, lw=1.5, alpha=0.7))

ax_right.set_xticks(xi)
ax_right.set_xticklabels(split_labels, fontsize=10)
ax_right.set_ylim(0, 0.55)
ax_right.set_ylabel("Transfer rate  (free-rec gain / taught gain)", fontsize=9)
ax_right.set_title("Transfer rate: how much untaught recall\nis unlocked per unit of taught recall",
                   fontsize=10, fontweight="bold")
ax_right.legend(fontsize=9, loc="upper right")
ax_right.grid(True, axis="y", alpha=0.25, linestyle=":")

fig.suptitle(
    "Chart 6 · Transfer-rate scaling from forget01 → forget05\n"
    "RMU★ (l5_scoeff10_lr5e-5) transfer rate increases with training-set size; "
    "NPO's collapses",
    fontsize=11, fontweight="bold", y=1.02
)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT}")
