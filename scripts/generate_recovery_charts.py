"""
Generate recovery experiment charts for the running log.

Charts produced:
  1. Free recovery on forget10-minus-forget01 (taught 1%)
  2. Free recovery on forget10-minus-forget05 (taught 5%)
  3. Taught set performance across all conditions
  4. General utility across all conditions
  5. 2x2 summary — taught vs free recovery, NPO vs RMU
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "assets"

# ── colour palette ──────────────────────────────────────────────────────────
C_RETAIN90  = "#4CAF50"   # green
C_NPO       = "#2196F3"   # blue
C_RMU       = "#FF9800"   # orange
C_FULL_HF   = "#9C27B0"   # purple  (reference only)
C_BASELINE  = 0.85        # lightness for hatched "baseline" bars
ALPHA_BASE  = 0.55        # alpha for baseline bars
ALPHA_TUNED = 0.92        # alpha for tuned bars

# ── data ─────────────────────────────────────────────────────────────────────

# retain90 HF baseline on the standard forget10 eval (proxy for custom splits)
# Source: tmp/hf_eval_summaries/tofu_Llama-3.2-1B-Instruct_retain90_hf.json
R90_BASE_QAP = 0.1161
R90_BASE_ES  = 0.0589
R90_BASE_UTIL = 0.5908   # model_utility (proxy for retain90_utility)

# Full HF reference
# Source: tmp/hf_eval_summaries/tofu_Llama-3.2-1B-Instruct_full_hf.json
FULL_HF_QAP  = 0.8805
FULL_HF_ES   = 0.7054
FULL_HF_UTIL = 0.5995

# ── Chart 1 & 2 data: free-recovery splits ──────────────────────────────────
# Source files in tmp/hfbase_recovery_eval_summaries/ and
#              tmp/new_recovery_eval_summaries/

# forget10-minus-forget01 free-recovery
FR_F01 = {
    "retain90 baseline\n(proxy)":   dict(QAP=R90_BASE_QAP, ES=R90_BASE_ES),
    "retain90\n→ forget01 tuned":   dict(QAP=0.0924, ES=0.0815),
    "NPO baseline":                  dict(QAP=0.2057, ES=0.0954),
    "NPO\n→ forget01 tuned":        dict(QAP=0.4576, ES=0.1926),
    "RMU baseline":                  dict(QAP=0.8320, ES=0.5604),
    "RMU\n→ forget01 tuned":        dict(QAP=0.5985, ES=0.2570),
}

# forget10-minus-forget05 free-recovery
FR_F05 = {
    "retain90 baseline\n(proxy)":   dict(QAP=R90_BASE_QAP, ES=R90_BASE_ES),
    "retain90\n→ forget05 tuned":   dict(QAP=0.0367, ES=0.3782),
    "NPO baseline":                  dict(QAP=0.2086, ES=0.0954),
    "NPO\n→ forget05 tuned":        dict(QAP=0.3159, ES=0.4012),
    "RMU baseline":                  dict(QAP=0.8285, ES=0.5604),
    "RMU\n→ forget05 tuned":        dict(QAP=0.4556, ES=0.3422),
}

# ── Chart 3 data: taught split performance ──────────────────────────────────
# Grouped by training condition (forget01, forget05, forget10)
# Source files in tmp/hfbase_recovery_eval_summaries/ and
#              tmp/new_recovery_eval_summaries/

TAUGHT = {
    "forget01": {
        "retain90": dict(QAP=0.7543, ES=0.3006),
        "NPO":      dict(QAP=0.9294, ES=0.7675),
        "RMU":      dict(QAP=0.8628, ES=0.4724),
    },
    "forget05": {
        "retain90": dict(QAP=0.8573, ES=0.6978),
        "NPO":      dict(QAP=0.9059, ES=0.6846),
        "RMU":      dict(QAP=0.8376, ES=0.5212),
    },
    "forget10": {
        # retain90 not run on the custom forget10 split; skip
        "NPO":      dict(QAP=0.8644, ES=0.5914),
        "RMU":      dict(QAP=0.7848, ES=0.4472),
    },
}

# ── Chart 4 data: retain90 utility ──────────────────────────────────────────
# retain90_utility (composite) from custom retain90_perturbed evals
# Source: tmp/hfbase_recovery_eval_summaries/*retain90_utility*.json

UTILITY = {
    "NPO\nbaseline":   dict(u=0.3486, qap=0.4233),
    "NPO\n→ f01":     dict(u=0.5280, qap=0.6339),
    "NPO\n→ f05":     dict(u=0.4547, qap=0.4528),
    "NPO\n→ f10":     dict(u=0.4429, qap=0.4154),
    "RMU\nbaseline":  dict(u=0.6559, qap=0.8211),
    "RMU\n→ f01":     dict(u=0.5198, qap=0.5859),
    "RMU\n→ f05":     dict(u=0.4459, qap=0.4289),
    "RMU\n→ f10":     dict(u=0.4170, qap=0.3768),
    # retain90→tuned runs (from new_recovery_eval_summaries)
    "retain90\n→ f01": dict(u=0.4971, qap=0.5603),
    "retain90\n→ f05": dict(u=0.3234, qap=0.2246),
}

# ── Chart 5 data: taught vs free-recovery scatter ───────────────────────────
SCATTER = [
    # (label,  method,  train_split,  taught_QAP,  free_rec_QAP)
    ("retain90→f01", "retain90", "forget01", 0.7543, 0.0924),
    ("retain90→f05", "retain90", "forget05", 0.8573, 0.0367),
    ("NPO→f01",      "NPO",      "forget01", 0.9294, 0.4576),
    ("NPO→f05",      "NPO",      "forget05", 0.9059, 0.3159),
    ("RMU→f01",      "RMU",      "forget01", 0.8628, 0.5985),
    ("RMU→f05",      "RMU",      "forget05", 0.8376, 0.4556),
]
# Baseline (untuned) pseudo-taught: QA prob on the split before any tuning
SCATTER_BASE = [
    ("NPO base→f01",  "NPO",  "forget01", 0.2261, 0.2057),
    ("NPO base→f05",  "NPO",  "forget05", 0.2069, 0.2086),
    ("RMU base→f01",  "RMU",  "forget01", 0.8493, 0.8320),
    ("RMU base→f05",  "RMU",  "forget05", 0.8393, 0.8285),
]

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def method_color(name):
    if "retain90" in name.lower():
        return C_RETAIN90
    if "npo" in name.lower():
        return C_NPO
    if "rmu" in name.lower():
        return C_RMU
    return "grey"


def is_baseline(label):
    return "baseline" in label.lower() or "proxy" in label.lower()


def bar_style(label, color):
    """Return kwargs for plt.bar() distinguishing baseline vs tuned."""
    if is_baseline(label):
        return dict(color=color, alpha=ALPHA_BASE, hatch="///", edgecolor="white",
                    linewidth=0.6)
    return dict(color=color, alpha=ALPHA_TUNED, edgecolor="white", linewidth=0.6)


def value_label(ax, bars, fmt="{:.2f}", offset=0.01):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom", fontsize=7)


def save(fig, name):
    path = f"{OUT_DIR}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 – Free recovery on forget10-minus-forget01
# ─────────────────────────────────────────────────────────────────────────────

def chart_free_recovery(data, title, fname, free_rec_label):
    labels  = list(data.keys())
    qap_vals = [data[l]["QAP"] for l in labels]
    es_vals  = [data[l]["ES"]  for l in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (lbl, qap, es) in enumerate(zip(labels, qap_vals, es_vals)):
        color = method_color(lbl)
        kw_q  = bar_style(lbl, color)
        kw_e  = dict(**kw_q)

        # QAP bar
        b1 = ax.bar(x[i] - w/2, qap, w, **kw_q, label="_nolegend_")
        # ES bar – slightly lighter shade
        kw_e["alpha"] = max(kw_e["alpha"] - 0.15, 0.3)
        b2 = ax.bar(x[i] + w/2, es,  w, **kw_e, label="_nolegend_")

        ax.text(x[i] - w/2, qap + 0.01, f"{qap:.3f}", ha="center",
                va="bottom", fontsize=6.5)
        ax.text(x[i] + w/2, es  + 0.01, f"{es:.3f}",  ha="center",
                va="bottom", fontsize=6.5)

    # retain90 floor reference line (QAP from standard eval)
    ax.axhline(R90_BASE_QAP, color=C_RETAIN90, linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"retain90 floor (QAP ≈ {R90_BASE_QAP:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # legend: solid vs hatched + metric colours
    patch_qap   = mpatches.Patch(facecolor="grey", alpha=0.8,  label="forget_Q_A_Prob (solid)")
    patch_es    = mpatches.Patch(facecolor="grey", alpha=0.45, label="extraction_strength (lighter)")
    patch_base  = mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///",
                                  edgecolor="grey", label="baseline (untuned)")
    patch_tuned = mpatches.Patch(facecolor="grey", alpha=0.92, label="tuned checkpoint")

    p_r90 = mpatches.Patch(color=C_RETAIN90, label="retain90")
    p_npo = mpatches.Patch(color=C_NPO,      label="NPO")
    p_rmu = mpatches.Patch(color=C_RMU,      label="RMU")

    ax.legend(handles=[patch_qap, patch_es, patch_base, patch_tuned, p_r90, p_npo, p_rmu],
              fontsize=7.5, loc="upper left", ncol=2)

    ax.annotate(
        f"Free-recovery split: {free_rec_label}\n"
        "Left bar = forget_Q_A_Prob · Right bar = extraction_strength\n"
        "Hatched = baseline (before recovery fine-tuning)   "
        "Solid = after teaching",
        xy=(0.5, -0.18), xycoords="axes fraction",
        ha="center", fontsize=7.5, color="dimgrey"
    )

    fig.tight_layout()
    save(fig, fname)


chart_free_recovery(
    FR_F01,
    "Chart 1 · Free recovery on forget10-minus-forget01 (taught 1% of forget10 authors)",
    "recovery_chart1_free_recovery_forget01",
    "forget10 − forget01 (360 Q)"
)

chart_free_recovery(
    FR_F05,
    "Chart 2 · Free recovery on forget10-minus-forget05 (taught 5% of forget10 authors)",
    "recovery_chart2_free_recovery_forget05",
    "forget10 − forget05 (200 Q)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 – Taught set performance
# ─────────────────────────────────────────────────────────────────────────────

def chart_taught():
    methods  = ["retain90", "NPO", "RMU"]
    splits   = ["forget01", "forget05", "forget10"]
    method_colors = {"retain90": C_RETAIN90, "NPO": C_NPO, "RMU": C_RMU}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    metrics   = [("QAP", "forget_Q_A_Prob"), ("ES", "extraction_strength")]

    for ax, (mkey, mlabel) in zip(axes, metrics):
        n_splits  = len(splits)
        n_methods = len(methods)
        group_w   = 0.75
        bar_w     = group_w / n_methods
        x         = np.arange(n_splits)

        for mi, method in enumerate(methods):
            vals = []
            for split in splits:
                if method in TAUGHT.get(split, {}):
                    vals.append(TAUGHT[split][method][mkey])
                else:
                    vals.append(np.nan)

            offset = (mi - (n_methods - 1) / 2) * bar_w
            color  = method_colors[method]
            bars   = []
            for xi, v in enumerate(vals):
                if np.isnan(v):
                    continue
                b = ax.bar(x[xi] + offset, v, bar_w,
                           color=color, alpha=0.85,
                           edgecolor="white", linewidth=0.6)
                bars.append(b[0])
                ax.text(x[xi] + offset, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5)

        # Full HF reference line
        ax.axhline(FULL_HF_QAP if mkey == "QAP" else FULL_HF_ES,
                   color=C_FULL_HF, linestyle="--", linewidth=1.3, alpha=0.7,
                   label=f"Full HF ({FULL_HF_QAP if mkey=='QAP' else FULL_HF_ES:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels(["Taught forget01\n(1% of f10)",
                             "Taught forget05\n(5% of f10)",
                             "Taught forget10\n(full 10%)"],
                           fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_title(f"Taught set: {mlabel}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5)

    # shared method legend below axes
    handles = [mpatches.Patch(color=c, alpha=0.85, label=m)
               for m, c in method_colors.items()]
    handles.append(mpatches.Patch(color=C_FULL_HF, alpha=0.7, label="Full HF ref."))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Chart 3 · Taught Set Performance — how well each model learned what was explicitly trained",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "recovery_chart3_taught_performance")


chart_taught()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 – General utility across all conditions
# ─────────────────────────────────────────────────────────────────────────────

def chart_utility():
    # ordered groups: NPO block then RMU block then retain90 block
    order = [
        ("retain90\n→ f01", C_RETAIN90),
        ("retain90\n→ f05", C_RETAIN90),
        ("NPO\nbaseline",   C_NPO),
        ("NPO\n→ f01",      C_NPO),
        ("NPO\n→ f05",      C_NPO),
        ("NPO\n→ f10",      C_NPO),
        ("RMU\nbaseline",   C_RMU),
        ("RMU\n→ f01",      C_RMU),
        ("RMU\n→ f05",      C_RMU),
        ("RMU\n→ f10",      C_RMU),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, metric_key, ylabel in [
        (ax1, "u",   "retain90_utility (composite)"),
        (ax2, "qap", "retain_Q_A_Prob"),
    ]:
        x     = np.arange(len(order))
        vals  = [UTILITY[lbl][metric_key] for lbl, _ in order]
        colors = [c for _, c in order]
        hatches = ["///" if "baseline" in lbl else "" for lbl, _ in order]

        bars = ax.bar(x, vals, 0.65, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.6)
        for bar, h, v in zip(bars, hatches, vals):
            if h:
                bar.set_hatch(h)
                bar.set_alpha(0.55)
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # reference lines
        ax.axhline(R90_BASE_UTIL, color=C_RETAIN90, linestyle="--", linewidth=1.2,
                   alpha=0.65, label=f"retain90 HF model_utility ({R90_BASE_UTIL:.3f})")
        ax.axhline(FULL_HF_UTIL, color=C_FULL_HF, linestyle=":", linewidth=1.4,
                   alpha=0.8, label=f"Full HF model_utility ({FULL_HF_UTIL:.3f})")

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _ in order], fontsize=8)
        ax.set_ylim(0.1, 0.85)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7.5)

    handles = [
        mpatches.Patch(color=C_RETAIN90, alpha=0.85, label="retain90"),
        mpatches.Patch(color=C_NPO,      alpha=0.85, label="NPO"),
        mpatches.Patch(color=C_RMU,      alpha=0.85, label="RMU"),
        mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///",
                       edgecolor="grey", label="baseline (untuned)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        "Chart 4 · General Utility — retain90_utility and retain_Q_A_Prob across all checkpoints\n"
        "NPO fine-tuning recovers utility; RMU fine-tuning degrades it",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    save(fig, "recovery_chart4_utility")


chart_utility()


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5 – 2×2 summary: taught performance vs free-recovery, NPO vs RMU
# ─────────────────────────────────────────────────────────────────────────────

def chart_scatter():
    marker_map = {"forget01": "o", "forget05": "s"}
    color_map  = {"retain90": C_RETAIN90, "NPO": C_NPO, "RMU": C_RMU}

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # plot baselines (open markers)
    for lbl, method, split, taught, free in SCATTER_BASE:
        c  = color_map[method]
        mk = marker_map.get(split, "D")
        ax.scatter(taught, free, s=110, marker=mk, facecolors="none",
                   edgecolors=c, linewidths=1.8, zorder=4)
        ax.annotate(lbl, (taught, free),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=c, alpha=0.8)

    # plot tuned checkpoints (filled markers) and connect to baseline with arrow
    base_qap_by = {
        ("NPO",  "forget01"): (0.2261, 0.2057),
        ("NPO",  "forget05"): (0.2069, 0.2086),
        ("RMU",  "forget01"): (0.8493, 0.8320),
        ("RMU",  "forget05"): (0.8393, 0.8285),
    }

    for lbl, method, split, taught, free in SCATTER:
        c  = color_map[method]
        mk = marker_map.get(split, "D")
        ax.scatter(taught, free, s=130, marker=mk, color=c,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(lbl, (taught, free),
                    textcoords="offset points", xytext=(6, -9),
                    fontsize=7.5, color=c, fontweight="bold")

        # draw arrow from baseline to tuned
        key = (method, split)
        if key in base_qap_by:
            bx, by = base_qap_by[key]
            ax.annotate("", xy=(taught, free), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="->", color=c,
                                        lw=1.2, alpha=0.6))

    # retain90 tuned: no baseline arrow since retain90 didn't know forget10
    # (shown as filled markers above)

    # diagonal reference (no free recovery line)
    ax.axhline(R90_BASE_QAP, color=C_RETAIN90, linestyle="--", linewidth=1,
               alpha=0.5, label=f"retain90 floor (QAP≈{R90_BASE_QAP:.3f})")

    ax.set_xlabel("forget_Q_A_Prob on taught split", fontsize=10)
    ax.set_ylabel("forget_Q_A_Prob on free-recovery split", fontsize=10)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.0)
    ax.set_title("Chart 5 · Taught vs Free-Recovery Performance\n"
                 "Arrows: baseline → tuned checkpoint  ·  ○ = baseline  ● = tuned  ○/□ = forget01/forget05",
                 fontsize=10, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=C_RETAIN90, label="retain90"),
        mpatches.Patch(color=C_NPO,      label="NPO"),
        mpatches.Patch(color=C_RMU,      label="RMU"),
        plt.Line2D([0],[0], marker="o", color="grey", markersize=7,
                   label="forget01 training", linestyle="none"),
        plt.Line2D([0],[0], marker="s", color="grey", markersize=7,
                   label="forget05 training", linestyle="none"),
        plt.Line2D([0],[0], marker="o", color="grey", markersize=7,
                   fillstyle="none", linestyle="none", label="baseline (open)"),
        plt.Line2D([0],[0], marker="o", color="grey", markersize=7,
                   label="tuned (filled)", linestyle="none"),
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right", ncol=2)

    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    save(fig, "recovery_chart5_taught_vs_free_recovery")


chart_scatter()

print("\nAll charts saved to assets/")
