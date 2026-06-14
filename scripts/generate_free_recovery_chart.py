"""
Generate four free-recovery charts for all 9 unlearning methods.

  Chart 1a (QAP)      → assets/recovery_chart_free_recovery_qap.png
  Chart 1b (ROUGE)    → assets/recovery_chart_free_recovery_rouge.png
  Chart 1c (privleak) → assets/recovery_chart_free_recovery_privleak.png
  Chart 1d (ES)       → assets/recovery_chart_free_recovery_es.png

x-axis:
  Baseline    → unlearned model on full forget10 (or forget10−forget01 ≈95%)
  Taught 1%   → trained on forget01, evaluated on forget10−forget01
  Taught 5%   → trained on forget05, evaluated on forget10−forget05

Core methods (Retain90 / NPO / RMU★): mean ± std across 3 seeds.
New methods (AltPO / GradDiff / IdkDPO / IdkNLL / SimNPO / UNDIAL): single seed.

Baselines:
  Retain90 → evals_forget10_retainref (consistent retain-ref config)
  NPO/RMU★ → HF eval summaries (consistent retain-logs-path config)
  New 6     → _baseline_forget01/evals_forget10_minus_forget01 (≈full forget10)
  Full      → evals_forget10_consistent
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
HF_DIR   = Path("tmp/hf_eval_summaries2")
RMU_DIR  = Path("tmp/rmu_scan_summaries")

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
    "Full": 9,
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

CORE_MODELS = list(CORE_TAGS.keys())
NEW_MODELS  = list(NEW_TAGS.keys())

# ── baseline paths ────────────────────────────────────────────────────────────

R90_BASE  = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_retain90/evals_forget10_retainref/TOFU_SUMMARY.json"
NPO_BASE  = HF_DIR   / "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10_hf.json"
RMU_BASE  = RMU_DIR  / "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10_hf_TOFU_SUMMARY.json"
FULL_BASE = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_full/evals_forget10_consistent/TOFU_SUMMARY.json"
FULL_DIR  = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_full"

CORE_BASELINES = {"Retain90": R90_BASE, "NPO": NPO_BASE, "RMU★": RMU_BASE}

XLABELS = [
    "Baseline\n(full forget10)",
    "Taught 1%\n(forget10 − forget01)",
    "Taught 5%\n(forget10 − forget05)",
]


def _load(path: Path, key: str) -> float:
    try:
        return float(json.loads(path.read_text())[key])
    except Exception:
        return float("nan")


def _trl_dir(tag: str, split: str, sfx: str = "") -> Path:
    return EVAL_DIR / f"tofu_Llama-3.2-1B-Instruct_{tag}_trl_{split}_lora_e20_lr2e4{sfx}"


def _stats(paths: list[Path], key: str) -> tuple[float, float]:
    vals = [_load(p, key) for p in paths]
    arr  = np.array([v for v in vals if not np.isnan(v)])
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


# ── data builder ──────────────────────────────────────────────────────────────

def _build(key: str) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
]:
    """Returns (means, stds): model → [base, f01, f05]"""
    means: dict[str, list[float]] = {}
    stds:  dict[str, list[float]] = {}

    # Core 3 (3-seed average)
    for model, tag in CORE_TAGS.items():
        base_val = _load(CORE_BASELINES[model], key)
        m_row = [base_val]
        s_row = [float("nan")]
        for split in ["forget01", "forget05"]:
            paths = [_trl_dir(tag, split, sfx) / f"evals_forget10_minus_{split}/TOFU_SUMMARY.json"
                     for sfx in CORE_SEED_SFXS]
            mn, sd = _stats(paths, key)
            m_row.append(mn)
            s_row.append(sd)
        means[model] = m_row
        stds[model]  = s_row

    # New 6 (single seed; baseline from _baseline_forget01)
    for model, tag in NEW_TAGS.items():
        base_path = (EVAL_DIR
                     / f"tofu_Llama-3.2-1B-Instruct_{tag}_baseline_forget01"
                     / "evals_forget10_minus_forget01"
                     / "TOFU_SUMMARY.json")
        m_row = [_load(base_path, key)]
        s_row = [float("nan")]
        for split in ["forget01", "forget05"]:
            p = _trl_dir(tag, split) / f"evals_forget10_minus_{split}/TOFU_SUMMARY.json"
            m_row.append(_load(p, key))
            s_row.append(float("nan"))
        means[model] = m_row
        stds[model]  = s_row

    # Full reference (single runs)
    means["Full"] = [
        _load(FULL_BASE, key),
        _load(FULL_DIR / "evals_forget10_minus_forget01/TOFU_SUMMARY.json", key),
        _load(FULL_DIR / "evals_forget10_minus_forget05/TOFU_SUMMARY.json", key),
    ]
    stds["Full"] = [float("nan")] * 3

    return means, stds


# ── chart renderer ────────────────────────────────────────────────────────────

PLOT_ORDER = ["Full"] + NEW_MODELS + CORE_MODELS  # core drawn last (on top)


def _single_panel(
    means: dict[str, list[float]],
    stds:  dict[str, list[float]],
    ylabel: str,
    suptitle: str,
    out: Path,
    ylim: tuple[float, float] | None = (0, 1.0),
    hline: float | None = None,
    annotate_fmt: str = ".3f",
    annotate_offset: int = 9,
) -> None:
    x = np.arange(len(XLABELS))
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model in PLOT_ORDER:
        m_arr = np.array(means[model])
        s_arr = np.array(stds[model])
        is_core = model in CORE_MODELS

        ax.plot(
            x, m_arr,
            color=COLORS[model], marker=MARKERS[model],
            linestyle=LS[model], linewidth=LW[model],
            markersize=MS[model], label=model,
            zorder=4 if is_core else 2,
        )

        if is_core:
            mask = ~np.isnan(s_arr)
            if mask.any():
                xf = x[mask]
                lo = m_arr[mask] - s_arr[mask]
                hi = m_arr[mask] + s_arr[mask]
                ax.fill_between(xf, lo, hi, color=COLORS[model],
                                alpha=0.18, zorder=2, linewidth=0)
            for xi, v in enumerate(m_arr):
                if not np.isnan(v):
                    ax.annotate(
                        f"{v:{annotate_fmt}}",
                        xy=(xi, v), xytext=(0, annotate_offset),
                        textcoords="offset points",
                        ha="center", fontsize=8, color=COLORS[model],
                    )

    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend: core first, then new methods, then Full
    handles_core = [plt.Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
                               linestyle=LS[m], linewidth=LW[m], markersize=MS[m])
                    for m in CORE_MODELS]
    handles_new  = [plt.Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
                               linestyle=LS[m], linewidth=LW[m], markersize=MS[m])
                    for m in NEW_MODELS]
    handles_full = [plt.Line2D([0], [0], color=COLORS["Full"], marker=MARKERS["Full"],
                               linestyle=LS["Full"], linewidth=LW["Full"], markersize=MS["Full"])]
    all_handles = handles_core + handles_new + handles_full
    all_labels  = CORE_MODELS + NEW_MODELS + ["Full"]
    ax.legend(all_handles, all_labels, fontsize=8, frameon=False,
              ncol=2, loc="best")

    fig.suptitle(suptitle, fontsize=10.5, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


CORE_CHART_SPECS = [
    ("forget_Q_A_Prob",
     "forget_Q_A_Prob (untaught set)",
     "Chart 1a · Probability shift on untaught forget10 content (mean ± std, 3 seeds)",
     "recovery_chart_free_recovery_qap.png",
     (0, 1.0), None, ".3f", 9),
    ("forget_Q_A_ROUGE",
     "forget_Q_A_ROUGE (untaught set)",
     "Chart 1b · ROUGE on untaught forget10 content (mean ± std, 3 seeds)",
     "recovery_chart_free_recovery_rouge.png",
     (0, 1.0), None, ".3f", 9),
    ("privleak",
     "privleak — Min-K% MIA relative to retain90 reference",
     "Chart 1c · Representation-level alignment on untaught forget10 content (mean ± std, 3 seeds)",
     "recovery_chart_free_recovery_privleak.png",
     (-110, 80), 0, ".1f", 7),
    ("extraction_strength",
     "extraction_strength (untaught set)",
     "Chart 1d · Generative recall on untaught forget10 content (mean ± std, 3 seeds)",
     "recovery_chart_free_recovery_es.png",
     (0, 1.0), None, ".3f", 9),
]

EXPANDED_CHART_SPECS = [
    ("forget_Q_A_Prob",
     "forget_Q_A_Prob (untaught set)",
     "Chart A1 · Probability shift on untaught forget10 content\n(core: mean ± std, 3 seeds; others: single seed)",
     "recovery_chart_expanded_qap.png",
     (0, 1.0), None, ".3f", 9),
    ("forget_Q_A_ROUGE",
     "forget_Q_A_ROUGE (untaught set)",
     "Chart A2 · ROUGE on untaught forget10 content\n(core: mean ± std, 3 seeds; others: single seed)",
     "recovery_chart_expanded_rouge.png",
     (0, 1.0), None, ".3f", 9),
    ("privleak",
     "privleak — Min-K% MIA relative to retain90 reference",
     "Chart A3 · Representation-level alignment on untaught forget10 content\n(core: mean ± std, 3 seeds; others: single seed)",
     "recovery_chart_expanded_privleak.png",
     (-130, 80), 0, ".1f", 7),
    ("extraction_strength",
     "extraction_strength (untaught set)",
     "Chart A4 · Generative recall on untaught forget10 content\n(core: mean ± std, 3 seeds; others: single seed)",
     "recovery_chart_expanded_es.png",
     (0, 1.0), None, ".3f", 9),
]


def _build_core(key: str) -> tuple[dict, dict]:
    """Returns (means, stds) for core 3 + Full only — used for original charts."""
    means: dict[str, list[float]] = {}
    stds:  dict[str, list[float]] = {}

    for model, tag in CORE_TAGS.items():
        base_val = _load(CORE_BASELINES[model], key)
        m_row = [base_val]
        s_row = [float("nan")]
        for split in ["forget01", "forget05"]:
            paths = [_trl_dir(tag, split, sfx) / f"evals_forget10_minus_{split}/TOFU_SUMMARY.json"
                     for sfx in CORE_SEED_SFXS]
            mn, sd = _stats(paths, key)
            m_row.append(mn)
            s_row.append(sd)
        means[model] = m_row
        stds[model]  = s_row

    means["Full"] = [
        _load(FULL_BASE, key),
        _load(FULL_DIR / "evals_forget10_minus_forget01/TOFU_SUMMARY.json", key),
        _load(FULL_DIR / "evals_forget10_minus_forget05/TOFU_SUMMARY.json", key),
    ]
    stds["Full"] = [float("nan")] * 3

    return means, stds


def _single_panel_core(
    means: dict[str, list[float]],
    stds:  dict[str, list[float]],
    ylabel: str,
    suptitle: str,
    out: Path,
    ylim: tuple[float, float] | None = (0, 1.0),
    hline: float | None = None,
    annotate_fmt: str = ".3f",
    annotate_offset: int = 9,
) -> None:
    """Original 3-method + Full chart (used for Charts 1a–1d in main results)."""
    x = np.arange(len(XLABELS))
    fig, ax = plt.subplots(figsize=(7, 5))

    core_order = [*CORE_TAGS.keys(), "Full"]
    for model in core_order:
        m_arr = np.array(means[model])
        s_arr = np.array(stds[model])

        ax.plot(
            x, m_arr,
            color=COLORS[model], marker=MARKERS[model],
            linestyle=LS[model], linewidth=LW[model],
            markersize=MS[model], label=model, zorder=3,
        )

        mask = ~np.isnan(s_arr)
        if mask.any():
            xf = x[mask]
            lo = m_arr[mask] - s_arr[mask]
            hi = m_arr[mask] + s_arr[mask]
            ax.fill_between(xf, lo, hi, color=COLORS[model], alpha=0.18, zorder=2, linewidth=0)

        for xi, v in enumerate(m_arr):
            if not np.isnan(v):
                ax.annotate(
                    f"{v:{annotate_fmt}}",
                    xy=(xi, v), xytext=(0, annotate_offset),
                    textcoords="offset points",
                    ha="center", fontsize=8, color=COLORS[model],
                )

    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle="--", alpha=0.4, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle(suptitle, fontsize=11, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


if __name__ == "__main__":
    import sys
    core_only = "--core-only" in sys.argv
    OUT_DIR.mkdir(exist_ok=True)

    if core_only:
        # Regenerate original 3-method + Full charts
        for key, ylabel, suptitle, outname, ylim, hline, fmt, offset in CORE_CHART_SPECS:
            m, s = _build_core(key)
            _single_panel_core(m, s, ylabel, suptitle,
                               out=OUT_DIR / outname, ylim=ylim, hline=hline,
                               annotate_fmt=fmt, annotate_offset=offset)
    else:
        for key, ylabel, suptitle, outname, ylim, hline, fmt, offset in EXPANDED_CHART_SPECS:
            m, s = _build(key)
            _single_panel(m, s, ylabel, suptitle,
                          out=OUT_DIR / outname, ylim=ylim, hline=hline,
                          annotate_fmt=fmt, annotate_offset=offset)

    print("Done.")
