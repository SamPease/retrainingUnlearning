"""
Generate Chart 7 (taught-set performance) and Chart 8 (retain90 utility)
for all 9 unlearning methods.

Core methods (Retain90 / NPO / RMU★): mean ± std across 3 seeds (42, 123, 456).
New methods (AltPO / GradDiff / IdkDPO / IdkNLL / SimNPO / UNDIAL): single seed only.

Seed path convention (_lora_e20_lr2e4 suffix):
  seed 42  → _lora_e20_lr2e4
  seed 123 → _lora_e20_lr2e4_seed123
  seed 456 → _lora_e20_lr2e4_seed456

New method baselines come from _baseline_forget10/evals_retain90_utility.
Core method baselines are hardcoded from prior single-checkpoint evaluations.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR  = Path("assets")
EVAL_DIR = Path("saves/eval")

# ── styling ───────────────────────────────────────────────────────────────────

COLORS = {
    "Retain90": "#4CAF50",
    "NPO":      "#2196F3",
    "RMU★":    "#FF9800",
    "AltPO":    "#D32F2F",
    "GradDiff": "#5D4037",
    "IdkDPO":   "#00897B",
    "IdkNLL":   "#F57F17",
    "SimNPO":   "#6A1B9A",
    "UNDIAL":   "#546E7A",
    "Full":     "#9C27B0",
}
MARKERS = {
    "Retain90": "o", "NPO": "s", "RMU★": "^",
    "AltPO": "v", "GradDiff": "P", "IdkDPO": "X",
    "IdkNLL": "D", "SimNPO": "h", "UNDIAL": "<",
    "Full": "*",
}
LS = {
    "Retain90": "-", "NPO": "--", "RMU★": "-.",
    "AltPO": "-", "GradDiff": "--", "IdkDPO": "-.",
    "IdkNLL": ":", "SimNPO": "-", "UNDIAL": "--",
    "Full": ":",
}
LW = {
    "Retain90": 2.5, "NPO": 2.5, "RMU★": 2.5,
    "AltPO": 1.5, "GradDiff": 1.5, "IdkDPO": 1.5,
    "IdkNLL": 1.5, "SimNPO": 1.5, "UNDIAL": 1.5,
    "Full": 1.8,
}
MS = {
    "Retain90": 8, "NPO": 8, "RMU★": 8,
    "AltPO": 6, "GradDiff": 6, "IdkDPO": 6,
    "IdkNLL": 6, "SimNPO": 6, "UNDIAL": 6,
    "Full": 8,
}

# ── method metadata ───────────────────────────────────────────────────────────

CORE_TAGS = {
    "Retain90": "retain90",
    "NPO":      "npo_forget10",
    "RMU★":    "rmu_l5_s10_lr5e5",
}
NEW_TAGS = {
    "AltPO":    "altpo_l5e5_b01_a1",
    "GradDiff": "graddiff_l4e5_a5",
    "IdkDPO":   "idkdpo_l5e5_b01_a1",
    "IdkNLL":   "idknll_l5e5_a2",
    "SimNPO":   "simnpo_l5e5_b45_d1_g025",
    "UNDIAL":   "undial_l1e4_b30_a2",
}
CORE_SEED_SFXS = ["", "_seed123", "_seed456"]
SPLITS = ["forget01", "forget05", "forget10"]

CORE_MODELS = list(CORE_TAGS.keys())
NEW_MODELS  = list(NEW_TAGS.keys())
ALL_MODELS  = CORE_MODELS + NEW_MODELS


def _load(path: Path, key: str) -> float:
    try:
        return float(json.loads(path.read_text())[key])
    except Exception:
        return float("nan")


def _stats(paths: list[Path], key: str) -> tuple[float, float]:
    vals = [_load(p, key) for p in paths]
    arr  = np.array([v for v in vals if not np.isnan(v)])
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def _trl_dir(tag: str, split: str, sfx: str = "") -> Path:
    return EVAL_DIR / f"tofu_Llama-3.2-1B-Instruct_{tag}_trl_{split}_lora_e20_lr2e4{sfx}"


# ── data builders ─────────────────────────────────────────────────────────────

def _build_taught(key: str) -> dict[str, tuple[list[float], list[float]]]:
    """Returns {model: ([means f01,f05,f10], [stds])}"""
    out: dict[str, tuple[list[float], list[float]]] = {}

    for model, tag in CORE_TAGS.items():
        means, stds = [], []
        for split in SPLITS:
            paths = [_trl_dir(tag, split, sfx) / f"evals_{split}_taught/TOFU_SUMMARY.json"
                     for sfx in CORE_SEED_SFXS]
            mn, sd = _stats(paths, key)
            means.append(mn)
            stds.append(sd)
        out[model] = (means, stds)

    for model, tag in NEW_TAGS.items():
        means, stds = [], []
        for split in SPLITS:
            p = _trl_dir(tag, split) / f"evals_{split}_taught/TOFU_SUMMARY.json"
            means.append(_load(p, key))
            stds.append(float("nan"))
        out[model] = (means, stds)

    return out


def _build_utility(key: str) -> dict[str, tuple[list[float], list[float]]]:
    """Returns {model: ([base, f01, f05, f10 means], [stds])}"""
    CORE_BASELINES = {
        "retain90_utility": {"Retain90": 0.591, "NPO": 0.349, "RMU★": 0.510},
        "retain_Q_A_Prob":  {"Retain90": 0.880, "NPO": 0.423, "RMU★": 0.607},
    }
    core_base = CORE_BASELINES.get(key, {m: float("nan") for m in CORE_MODELS})

    out: dict[str, tuple[list[float], list[float]]] = {}

    for model, tag in CORE_TAGS.items():
        means = [core_base[model]]
        stds  = [float("nan")]
        for split in SPLITS:
            paths = [_trl_dir(tag, split, sfx) / "evals_retain90_utility/TOFU_SUMMARY.json"
                     for sfx in CORE_SEED_SFXS]
            mn, sd = _stats(paths, key)
            means.append(mn)
            stds.append(sd)
        out[model] = (means, stds)

    for model, tag in NEW_TAGS.items():
        base_path = EVAL_DIR / f"tofu_Llama-3.2-1B-Instruct_{tag}_baseline_forget10/evals_retain90_utility/TOFU_SUMMARY.json"
        means = [_load(base_path, key)]
        stds  = [float("nan")]
        for split in SPLITS:
            p = _trl_dir(tag, split) / "evals_retain90_utility/TOFU_SUMMARY.json"
            means.append(_load(p, key))
            stds.append(float("nan"))
        out[model] = (means, stds)

    return out


# ── Chart 7: Taught-set performance (grouped bar chart) ───────────────────────

def chart_taught() -> None:
    qap_data = _build_taught("forget_Q_A_Prob")
    es_data  = _build_taught("extraction_strength")

    full_qap = 0.881
    full_es  = 0.705

    x        = np.arange(len(SPLITS))
    xlabels  = ["Taught 1%\n(forget01)", "Taught 5%\n(forget05)", "Taught 10%\n(forget10)"]
    n_models = len(ALL_MODELS)
    width    = 0.08

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=False)
    fig.subplots_adjust(wspace=0.30)

    for ax, data, ref_val, ylabel, title in [
        (axes[0], qap_data, full_qap, "forget_Q_A_Prob",    "Forget Q&A Probability (taught split)"),
        (axes[1], es_data,  full_es,  "extraction_strength", "Extraction Strength (taught split)"),
    ]:
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

        for i, model in enumerate(ALL_MODELS):
            means, stds = data[model]
            xpos = x + offsets[i]
            is_core = model in CORE_MODELS
            bars = ax.bar(
                xpos, means,
                width=width * 0.88,
                color=COLORS[model],
                alpha=0.85 if is_core else 0.65,
                label=model,
                zorder=3,
            )
            if is_core:
                ax.errorbar(
                    xpos, means, yerr=stds,
                    fmt="none", ecolor="#333333",
                    elinewidth=1.2, capsize=3, zorder=4,
                )
                for bar, v, e in zip(bars, means, stds):
                    if not np.isnan(v):
                        lbl = f"{v:.2f}"
                        if not np.isnan(e) and e > 0:
                            lbl += f"\n±{e:.3f}"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.018,
                            lbl,
                            ha="center", va="bottom",
                            fontsize=6.0, color="#333333",
                        )

        ax.axhline(ref_val, color=COLORS["Full"], linestyle="--", linewidth=1.4, zorder=4)
        ax.text(len(SPLITS) - 0.05, ref_val + 0.015, f"Full: {ref_val:.3f}",
                ha="right", va="bottom", fontsize=8, color=COLORS["Full"])

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0, 1.12)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles_core = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.85) for m in CORE_MODELS]
    handles_new  = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.65) for m in NEW_MODELS]
    handles_ref  = [plt.Line2D([0], [0], color=COLORS["Full"], linestyle="--", linewidth=1.4)]
    labels = CORE_MODELS + NEW_MODELS + ["Full model (HF)"]
    fig.legend(handles_core + handles_new + handles_ref, labels,
               loc="upper center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1.03))

    fig.suptitle("Chart 7 · Taught-set performance across all unlearning methods\n"
                 "(core: mean ± std, 3 seeds; others: single seed)",
                 fontsize=11.5, y=1.10, fontweight="bold")

    out = OUT_DIR / "recovery_chart7_taught_performance_expanded.png"
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Chart 8 expanded: Retain90 utility (line chart, all 9 methods) ────────────

def chart_utility() -> None:
    util_data = _build_utility("retain90_utility")
    rqap_data = _build_utility("retain_Q_A_Prob")

    full_util = 0.599
    full_rqap = 0.871

    xlabels = ["Baseline", "Taught\n1%", "Taught\n5%", "Taught\n10%"]
    x       = np.arange(len(xlabels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    PLOT_ORDER = NEW_MODELS + CORE_MODELS  # core drawn last so they appear on top

    for ax, data, ref_val, ylabel, title in [
        (axes[0], util_data, full_util, "retain90_utility", "Retain90 Utility"),
        (axes[1], rqap_data, full_rqap, "retain_Q_A_Prob",  "Retain Q&A Probability"),
    ]:
        for model in PLOT_ORDER:
            means_arr = np.array(data[model][0])
            stds_arr  = np.array(data[model][1])
            is_core   = model in CORE_MODELS

            ax.plot(
                x, means_arr,
                color=COLORS[model], marker=MARKERS[model],
                linestyle=LS[model], linewidth=LW[model],
                markersize=MS[model], label=model,
                zorder=4 if is_core else 2,
            )

            if is_core:
                mask = ~np.isnan(stds_arr)
                if mask.any():
                    xf = x[mask]
                    lo = means_arr[mask] - stds_arr[mask]
                    hi = means_arr[mask] + stds_arr[mask]
                    ax.fill_between(xf, lo, hi, color=COLORS[model], alpha=0.18, zorder=2, linewidth=0)
                for xi, v in enumerate(means_arr):
                    if not np.isnan(v):
                        ax.annotate(f"{v:.3f}", xy=(xi, v),
                                    xytext=(0, 8), textcoords="offset points",
                                    ha="center", fontsize=7.5, color=COLORS[model])

        ax.axhline(ref_val, color=COLORS["Full"], linestyle="--", linewidth=1.4, zorder=1)
        ax.text(len(x) - 1 + 0.05, ref_val + 0.015,
                f"Full: {ref_val:.3f}", ha="left", va="bottom",
                fontsize=8, color=COLORS["Full"])

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles_core = [plt.Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
                               linestyle=LS[m], linewidth=LW[m], markersize=MS[m])
                    for m in CORE_MODELS]
    handles_new  = [plt.Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
                               linestyle=LS[m], linewidth=LW[m], markersize=MS[m])
                    for m in NEW_MODELS]
    handles_ref  = [plt.Line2D([0], [0], color=COLORS["Full"], linestyle="--", linewidth=1.4)]
    labels = CORE_MODELS + NEW_MODELS + ["Full model (HF)"]

    fig.legend(handles_core + handles_new + handles_ref, labels,
               loc="upper center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1.03))

    fig.suptitle("Chart 8 · Retain90 utility across unlearning methods and training conditions\n"
                 "(core: mean ± std, 3 seeds; others: single seed)",
                 fontsize=11.5, y=1.10, fontweight="bold")

    out = OUT_DIR / "recovery_chart8_retain90_utility_expanded.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def chart_taught_core() -> None:
    """Original 3-method bar chart — saved to original Chart 7 filename."""
    qap_data: dict[str, tuple[list[float], list[float]]] = {}
    es_data:  dict[str, tuple[list[float], list[float]]] = {}

    for model, tag in CORE_TAGS.items():
        means_q, stds_q = [], []
        means_e, stds_e = [], []
        for split in SPLITS:
            paths = [_trl_dir(tag, split, sfx) / f"evals_{split}_taught/TOFU_SUMMARY.json"
                     for sfx in CORE_SEED_SFXS]
            mq, sq = _stats(paths, "forget_Q_A_Prob")
            me, se = _stats(paths, "extraction_strength")
            means_q.append(mq); stds_q.append(sq)
            means_e.append(me); stds_e.append(se)
        qap_data[model] = (means_q, stds_q)
        es_data[model]  = (means_e, stds_e)

    full_qap = 0.881
    full_es  = 0.705
    x        = np.arange(len(SPLITS))
    xlabels  = ["Taught 1%\n(forget01)", "Taught 5%\n(forget05)", "Taught 10%\n(forget10)"]
    n_models = len(CORE_MODELS)
    width    = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, data, ref_val, ylabel, title in [
        (axes[0], qap_data, full_qap, "forget_Q_A_Prob",    "Forget Q&A Probability (taught split)"),
        (axes[1], es_data,  full_es,  "extraction_strength", "Extraction Strength (taught split)"),
    ]:
        offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width
        for i, model in enumerate(CORE_MODELS):
            means, stds = data[model]
            xpos = x + offsets[i]
            bars = ax.bar(xpos, means, width=width * 0.92, color=COLORS[model],
                          alpha=0.85, label=model, zorder=3)
            ax.errorbar(xpos, means, yerr=stds, fmt="none", ecolor="#333333",
                        elinewidth=1.2, capsize=3, zorder=4)
            for bar, v, e in zip(bars, means, stds):
                if not np.isnan(v):
                    lbl = f"{v:.2f}"
                    if not np.isnan(e) and e > 0:
                        lbl += f"\n±{e:.3f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02, lbl,
                            ha="center", va="bottom", fontsize=6.5, color="#333333")

        ax.axhline(ref_val, color=COLORS["Full"], linestyle="--", linewidth=1.4, zorder=4)
        ax.text(len(SPLITS) - 0.05, ref_val + 0.015, f"Full: {ref_val:.3f}",
                ha="right", va="bottom", fontsize=8, color=COLORS["Full"])
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0, 1.08)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.85) for m in CORE_MODELS] + \
              [plt.Line2D([0], [0], color=COLORS["Full"], linestyle="--", linewidth=1.4)]
    labels = CORE_MODELS + ["Full model (HF)"]
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Chart 7 · Taught-set performance (mean ± std, 3 seeds)",
                 fontsize=12, y=1.07, fontweight="bold")

    out = OUT_DIR / "recovery_chart7_taught_performance.png"
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def chart_utility_core() -> None:
    """Original 3-method line chart — saved to original Chart 8 filename."""
    util_data = _build_utility("retain90_utility")
    rqap_data = _build_utility("retain_Q_A_Prob")

    full_util = 0.599
    full_rqap = 0.871
    xlabels = ["Baseline", "Taught\n1%", "Taught\n5%", "Taught\n10%"]
    x       = np.arange(len(xlabels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, data, ref_val, ylabel, title in [
        (axes[0], util_data, full_util, "retain90_utility", "Retain90 Utility"),
        (axes[1], rqap_data, full_rqap, "retain_Q_A_Prob",  "Retain Q&A Probability"),
    ]:
        for model in CORE_MODELS:
            means_arr = np.array(data[model][0])
            stds_arr  = np.array(data[model][1])
            ax.plot(x, means_arr, color=COLORS[model], marker=MARKERS[model],
                    linestyle=LS[model], linewidth=2, markersize=7, label=model, zorder=3)
            mask = ~np.isnan(stds_arr)
            if mask.any():
                lo = means_arr[mask] - stds_arr[mask]
                hi = means_arr[mask] + stds_arr[mask]
                ax.fill_between(x[mask], lo, hi, color=COLORS[model], alpha=0.18, zorder=2, linewidth=0)
            for xi, v in enumerate(means_arr):
                if not np.isnan(v):
                    ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 8),
                                textcoords="offset points", ha="center",
                                fontsize=7.5, color=COLORS[model])

        ax.axhline(ref_val, color=COLORS["Full"], linestyle="--", linewidth=1.4, zorder=4)
        ax.text(len(x) - 1 + 0.05, ref_val + 0.02, f"Full: {ref_val:.3f}",
                ha="left", va="bottom", fontsize=8, color=COLORS["Full"])
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, frameon=False)

    fig.suptitle("Chart 8 · Retain90 utility across training conditions (mean ± std, 3 seeds)",
                 fontsize=12, y=1.02, fontweight="bold")

    out = OUT_DIR / "recovery_chart8_retain90_utility.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    import sys
    core_only = "--core-only" in sys.argv

    if core_only:
        # Regenerate original 3-method charts to original filenames
        chart_taught_core()
        chart_utility_core()
    else:
        # Generate expanded 9-method charts to new _expanded filenames
        chart_taught()
        chart_utility()
    print("Done.")
