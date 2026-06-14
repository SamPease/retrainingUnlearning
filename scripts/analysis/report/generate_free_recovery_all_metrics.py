"""
Generate one line chart per TOFU metric for the free recovery analysis.

Models × conditions:
  Full / Retain90 / NPO / RMU★  ×  Baseline / Taught 1% / Taught 5%

Each x-axis point evaluates on the *untaught* portion of forget10:
  Baseline  → full forget10,             no tuning
  Taught 1% → forget10 − forget01,       trained on forget01
  Taught 5% → forget10 − forget05,       trained on forget05

Metrics (one chart each):
  forget_Q_A_Prob, forget_Q_A_ROUGE, extraction_strength,
  forget_quality, forget_truth_ratio, model_utility, privleak

Charts saved to assets/free_recovery_metric_<name>.png
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

# ── path constants ─────────────────────────────────────────────────────────────

# Baselines: evaluated on full forget10
# Full baseline: try consistent eval first, fall back to MIA-config eval
FULL_BASE_CONSISTENT = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_full/evals_forget10_consistent/TOFU_SUMMARY.json"
FULL_BASE_FALLBACK   = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_full/evals_forget10/TOFU_SUMMARY.json"

R90_BASE  = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_retain90/evals_forget10_retainref/TOFU_SUMMARY.json"
NPO_BASE  = HF_DIR   / "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10_hf.json"
RMU_BASE  = RMU_DIR  / "unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10_hf_TOFU_SUMMARY.json"

# Tuned conditions: evaluated on the untaught free-recovery subset
R90_F01   = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_retain90_trl_forget01/evals_forget10_minus_forget01/TOFU_SUMMARY.json"
R90_F05   = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_retain90_trl_forget05/evals_forget10_minus_forget05/TOFU_SUMMARY.json"
NPO_F01   = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_npo_forget10_trl_forget01/evals_forget10_minus_forget01/TOFU_SUMMARY.json"
NPO_F05   = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_npo_forget10_trl_forget05/evals_forget10_minus_forget05/TOFU_SUMMARY.json"
RMU_F01   = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_rmu_l5_s10_lr5e5_trl_forget01/evals_forget10_minus_forget01/TOFU_SUMMARY.json"
RMU_F05   = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_rmu_l5_s10_lr5e5_trl_forget05/evals_forget10_minus_forget05/TOFU_SUMMARY.json"
FULL_F01  = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget01/TOFU_SUMMARY.json"
FULL_F05  = EVAL_DIR / "tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget05/TOFU_SUMMARY.json"

NAN = float("nan")

C_RETAIN90 = "#4CAF50"
C_NPO      = "#2196F3"
C_RMU      = "#FF9800"
C_FULL     = "#9C27B0"

COLORS  = {"Retain90": C_RETAIN90, "NPO": C_NPO, "RMU★": C_RMU, "Full": C_FULL}
MARKERS = {"Retain90": "o", "NPO": "s", "RMU★": "^", "Full": "D"}
LS      = {"Retain90": "-", "NPO": "--", "RMU★": "-.", "Full": ":"}

XLABELS = [
    "Baseline\n(full forget10)",
    "Taught 1%\n(forget10 − forget01)",
    "Taught 5%\n(forget10 − forget05)",
]

METRICS = [
    ("forget_Q_A_Prob",    "forget_Q_A_Prob",    None),
    ("forget_Q_A_ROUGE",   "forget_Q_A_ROUGE",   None),
    ("extraction_strength","extraction_strength", None),
    ("forget_quality",     "forget_quality",      None),
    ("forget_truth_ratio", "forget_truth_ratio",  None),
    ("model_utility",      "model_utility",       None),
    ("privleak",           "privleak",            None),
]


def _load(path: Path, key: str) -> float:
    try:
        d = json.loads(path.read_text())
        return float(d[key])
    except Exception:
        return NAN


def _full_base(key: str) -> float:
    v = _load(FULL_BASE_CONSISTENT, key)
    if not np.isnan(v):
        return v
    return _load(FULL_BASE_FALLBACK, key)


def _build_data() -> dict[str, dict[str, list[float]]]:
    """Return {metric_key: {model_name: [base, f01, f05]}}"""
    raw = {
        "Retain90": [R90_BASE,  R90_F01, R90_F05],
        "NPO":      [NPO_BASE,  NPO_F01, NPO_F05],
        "RMU★":    [RMU_BASE,  RMU_F01, RMU_F05],
        "Full":     [None,      FULL_F01, FULL_F05],
    }

    all_metrics = [m[0] for m in METRICS]
    out: dict[str, dict[str, list[float]]] = {m: {} for m in all_metrics}

    for model, paths in raw.items():
        for metric in all_metrics:
            if model == "Full":
                vals = [_full_base(metric), _load(paths[1], metric), _load(paths[2], metric)]
            else:
                vals = [_load(p, metric) for p in paths]
            out[metric][model] = vals

    return out


def _chart(
    data: dict[str, list[float]],
    metric_key: str,
    ylabel: str,
    title: str,
    out: Path,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = np.arange(len(XLABELS))
    fig, ax = plt.subplots(figsize=(7, 5))

    for model, vals in data.items():
        varray = np.array(vals, dtype=float)
        mask = ~np.isnan(varray)

        # plot continuous segments (skip NaN gaps)
        ax.plot(
            x, varray,
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle=LS[model],
            linewidth=2,
            markersize=7,
            label=model,
            zorder=3,
        )

        for xi, v in enumerate(vals):
            if not np.isnan(v):
                ax.annotate(
                    f"{v:.3f}",
                    xy=(xi, v),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center", fontsize=8, color=COLORS[model],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    fig.suptitle(title, fontsize=11, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    all_data = _build_data()

    configs = [
        ("forget_Q_A_Prob",    "forget_Q_A_Prob",     "Forget Q&A Probability (untaught set)",   (0, 1.0)),
        ("forget_Q_A_ROUGE",   "forget_Q_A_ROUGE",    "Forget Q&A ROUGE (untaught set)",         (0, 1.0)),
        ("extraction_strength","extraction_strength",  "Extraction Strength (untaught set)",      (0, 1.0)),
        ("forget_quality",     "forget_quality",       "Forget Quality — KS p-value vs retain90 (untaught set)", (0, 1.0)),
        ("forget_truth_ratio", "forget_truth_ratio",   "Forget Truth Ratio — P(correct) / (P(correct)+P(wrong)) (untaught set)", (0, 1.0)),
        ("model_utility",      "model_utility",        "Model Utility (untaught set)",            (0, 1.0)),
        ("privleak",           "privleak",             "Privacy Leakage (untaught set)",          None),
    ]

    for metric_key, ylabel, title, ylim in configs:
        _chart(
            data=all_data[metric_key],
            metric_key=metric_key,
            ylabel=ylabel,
            title=title,
            out=OUT_DIR / f"free_recovery_metric_{metric_key}.png",
            ylim=ylim,
        )

    print(f"\nDone — {len(configs)} charts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
