"""
Generate recovery experiment charts (v2) for the running log.

Replaces the old RMU checkpoint (layer10_scoeff100_lr1e-5, which barely
unlearned) with the new best RMU checkpoint (layer5_scoeff10_lr5e-5,
harmonic=0.964) identified by the 09 June 2026 RMU scan.

Data sources
------------
  tmp/hfbase_recovery_eval_summaries/ — NPO and old RMU results (03 Jun 2026)
  tmp/new_recovery_eval_summaries/    — retain90->forgetX results (02 Jun 2026)
  tmp/rmu_new_recovery_eval_summaries/ — new RMU results (09 Jun 2026)

Charts produced (overwrite previous versions in assets/):
  1. Free recovery on forget10-minus-forget01 (taught 1%)
  2. Free recovery on forget10-minus-forget05 (taught 5%)
  3. Taught set performance across all conditions
  4. General utility across all conditions
  5. 2x2 summary — taught vs free-recovery, NPO vs new-RMU
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("assets")
OLD_DIR = Path("tmp/hfbase_recovery_eval_summaries")
R90_DIR = Path("tmp/new_recovery_eval_summaries")
NEW_DIR = Path("tmp/rmu_new_recovery_eval_summaries")

TAG = "rmu_l5_s10_lr5e5"

# ── colour palette ────────────────────────────────────────────────────────────
C_RETAIN90 = "#4CAF50"
C_NPO      = "#2196F3"
C_RMU      = "#FF9800"   # new RMU (replaces old)
C_FULL_HF  = "#9C27B0"
ALPHA_BASE  = 0.55
ALPHA_TUNED = 0.92

# ── reference constants ───────────────────────────────────────────────────────
# retain90 HF — standard forget10 eval (proxy for custom splits)
R90_BASE_QAP  = 0.1161
R90_BASE_ES   = 0.0589
R90_BASE_UTIL = 0.5908
FULL_HF_QAP   = 0.8805
FULL_HF_ES    = 0.7054
FULL_HF_UTIL  = 0.5995


# ── JSON loader ───────────────────────────────────────────────────────────────

def _load(path: Path, key: str, *, default: float = float("nan")) -> float:
    try:
        d = json.loads(path.read_text())
        return d[key]
    except (FileNotFoundError, KeyError):
        return default


def _old(split: str, variant: str, eval_name: str, key: str) -> float:
    """Load from tmp/hfbase_recovery_eval_summaries (NPO / old-RMU)."""
    fname = f"npo_forget10_{split}_{variant}_{eval_name}_TOFU_SUMMARY.json"
    return _load(OLD_DIR / fname, key)


def _r90(split: str, eval_type: str, key: str) -> float:
    """Load from tmp/new_recovery_eval_summaries (retain90->forgetX).
    eval_type: 'taught' | 'free_recovery' | 'retain90_utility'
    """
    fname = f"{split}_{eval_type}_TOFU_SUMMARY.json"
    return _load(R90_DIR / fname, key)


def _new(split: str, variant: str, eval_name: str, key: str) -> float:
    """Load from tmp/rmu_new_recovery_eval_summaries (new RMU)."""
    fname = f"{TAG}_{split}_{variant}_{eval_name}_TOFU_SUMMARY.json"
    return _load(NEW_DIR / fname, key)


# ── build data dicts ─────────────────────────────────────────────────────────

def _build_fr(fr_split: str) -> dict:
    """Build free-recovery data for charts 1 & 2."""
    minus = f"forget10_minus_{fr_split}"
    minus_dir = f"evals_{minus}"
    return {
        "retain90 baseline\n(proxy)": dict(QAP=R90_BASE_QAP, ES=R90_BASE_ES),
        f"retain90\n→ {fr_split} tuned": dict(
            QAP=_r90(fr_split, "free_recovery", "forget_Q_A_Prob"),
            ES=_r90(fr_split, "free_recovery", "extraction_strength"),
        ),
        "NPO baseline": dict(
            QAP=_old(fr_split, "baseline", minus_dir, "forget_Q_A_Prob"),
            ES=_old(fr_split, "baseline", minus_dir, "extraction_strength"),
        ),
        f"NPO\n→ {fr_split} tuned": dict(
            QAP=_old(fr_split, "tuned", minus_dir, "forget_Q_A_Prob"),
            ES=_old(fr_split, "tuned", minus_dir, "extraction_strength"),
        ),
        "RMU baseline": dict(
            QAP=_new(fr_split, "baseline", minus_dir, "forget_Q_A_Prob"),
            ES=_new(fr_split, "baseline", minus_dir, "extraction_strength"),
        ),
        f"RMU\n→ {fr_split} tuned": dict(
            QAP=_new(fr_split, "tuned", minus_dir, "forget_Q_A_Prob"),
            ES=_new(fr_split, "tuned", minus_dir, "extraction_strength"),
        ),
    }


def _build_taught() -> dict:
    result: dict = {}
    for split in ("forget01", "forget05", "forget10"):
        taught_dir = f"evals_{split}_taught"
        entry: dict = {}
        # retain90 (only forget01/forget05 ran)
        r90_qap = _r90(split, "taught", "forget_Q_A_Prob")
        r90_es  = _r90(split, "taught", "extraction_strength")
        if r90_qap == r90_qap:  # not NaN
            entry["retain90"] = dict(QAP=r90_qap, ES=r90_es)
        # NPO
        entry["NPO"] = dict(
            QAP=_old(split, "tuned", taught_dir, "forget_Q_A_Prob"),
            ES=_old(split, "tuned", taught_dir, "extraction_strength"),
        )
        # new RMU
        entry["RMU"] = dict(
            QAP=_new(split, "tuned", taught_dir, "forget_Q_A_Prob"),
            ES=_new(split, "tuned", taught_dir, "extraction_strength"),
        )
        result[split] = entry
    return result


def _build_utility() -> dict:
    u_dir = "evals_retain90_utility"
    result: dict = {}

    def _u(src_fn, split, variant):
        return dict(
            u=src_fn(split, variant, u_dir, "retain90_utility"),
            qap=src_fn(split, variant, u_dir, "retain_Q_A_Prob"),
        )

    result["NPO\nbaseline"]  = _u(_old, "forget01", "baseline")
    result["NPO\n→ f01"]     = _u(_old, "forget01", "tuned")
    result["NPO\n→ f05"]     = _u(_old, "forget05", "tuned")
    result["NPO\n→ f10"]     = _u(_old, "forget10", "tuned")
    result["RMU\nbaseline"]  = _u(_new, "forget01", "baseline")
    result["RMU\n→ f01"]     = _u(_new, "forget01", "tuned")
    result["RMU\n→ f05"]     = _u(_new, "forget05", "tuned")
    result["RMU\n→ f10"]     = _u(_new, "forget10", "tuned")
    result["retain90\n→ f01"] = dict(
        u=_r90("forget01", "retain90_utility", "retain90_utility"),
        qap=_r90("forget01", "retain90_utility", "retain_Q_A_Prob"),
    )
    result["retain90\n→ f05"] = dict(
        u=_r90("forget05", "retain90_utility", "retain90_utility"),
        qap=_r90("forget05", "retain90_utility", "retain_Q_A_Prob"),
    )
    return result


def _build_scatter():
    tuned = []
    base  = []
    for split, fr_split in [("forget01", "forget01"), ("forget05", "forget05")]:
        taught_dir = f"evals_{split}_taught"
        fr_dir     = f"evals_forget10_minus_{fr_split}"

        # retain90
        tuned.append((
            f"retain90→{split}", "retain90", split,
            _r90(split, "taught", "forget_Q_A_Prob"),
            _r90(split, "free_recovery", "forget_Q_A_Prob"),
        ))

        # NPO
        base.append((
            f"NPO base→{split}", "NPO", split,
            _old(split, "baseline", taught_dir, "forget_Q_A_Prob"),
            _old(split, "baseline", fr_dir, "forget_Q_A_Prob"),
        ))
        tuned.append((
            f"NPO→{split}", "NPO", split,
            _old(split, "tuned", taught_dir, "forget_Q_A_Prob"),
            _old(split, "tuned", fr_dir, "forget_Q_A_Prob"),
        ))

        # new RMU
        base.append((
            f"RMU base→{split}", "RMU", split,
            _new(split, "baseline", taught_dir, "forget_Q_A_Prob"),
            _new(split, "baseline", fr_dir, "forget_Q_A_Prob"),
        ))
        tuned.append((
            f"RMU→{split}", "RMU", split,
            _new(split, "tuned", taught_dir, "forget_Q_A_Prob"),
            _new(split, "tuned", fr_dir, "forget_Q_A_Prob"),
        ))

    # filter retain90 to only forget01/05 taught (no free-rec from retain90)
    tuned_filtered = [
        t for t in tuned if not (t[1] == "retain90")
    ]
    r90_tuned = [t for t in tuned if t[1] == "retain90"]
    return r90_tuned + tuned_filtered, base


# ── helpers ───────────────────────────────────────────────────────────────────

def method_color(name: str) -> str:
    n = name.lower()
    if "retain90" in n:
        return C_RETAIN90
    if "npo" in n:
        return C_NPO
    if "rmu" in n:
        return C_RMU
    return "grey"


def is_baseline(label: str) -> bool:
    return "baseline" in label.lower() or "proxy" in label.lower()


def bar_style(label: str, color: str) -> dict:
    if is_baseline(label):
        return dict(color=color, alpha=ALPHA_BASE, hatch="///",
                    edgecolor="white", linewidth=0.6)
    return dict(color=color, alpha=ALPHA_TUNED, edgecolor="white", linewidth=0.6)


def save(fig, name: str) -> None:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Chart 1 & 2 ──────────────────────────────────────────────────────────────

def chart_free_recovery(data: dict, title: str, fname: str, free_rec_label: str) -> None:
    labels    = list(data.keys())
    qap_vals  = [data[l]["QAP"] for l in labels]
    es_vals   = [data[l]["ES"]  for l in labels]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lbl, qap, es) in enumerate(zip(labels, qap_vals, es_vals)):
        color = method_color(lbl)
        kw_q  = bar_style(lbl, color)
        kw_e  = {**kw_q, "alpha": max(kw_q["alpha"] - 0.15, 0.3)}
        ax.bar(x[i] - w/2, qap, w, **kw_q)
        ax.bar(x[i] + w/2, es,  w, **kw_e)
        ax.text(x[i] - w/2, qap + 0.01, f"{qap:.3f}", ha="center",
                va="bottom", fontsize=6.5)
        ax.text(x[i] + w/2, es  + 0.01, f"{es:.3f}",  ha="center",
                va="bottom", fontsize=6.5)

    ax.axhline(R90_BASE_QAP, color=C_RETAIN90, linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"retain90 floor (QAP ≈ {R90_BASE_QAP:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    handles = [
        mpatches.Patch(facecolor="grey", alpha=0.8,  label="forget_Q_A_Prob (solid)"),
        mpatches.Patch(facecolor="grey", alpha=0.45, label="extraction_strength (lighter)"),
        mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///",
                       edgecolor="grey", label="baseline (untuned)"),
        mpatches.Patch(facecolor="grey", alpha=0.92, label="tuned checkpoint"),
        mpatches.Patch(color=C_RETAIN90, label="retain90"),
        mpatches.Patch(color=C_NPO,      label="NPO"),
        mpatches.Patch(color=C_RMU,      label="RMU (l5 scoeff10 lr5e-5)"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", ncol=2)
    ax.annotate(
        f"Free-recovery split: {free_rec_label}\n"
        "Left bar = forget_Q_A_Prob · Right bar = extraction_strength\n"
        "Hatched = baseline (before recovery fine-tuning)   Solid = after teaching\n"
        "RMU = layer5_scoeff10_lr5e-5 (new best checkpoint)",
        xy=(0.5, -0.20), xycoords="axes fraction",
        ha="center", fontsize=7.5, color="dimgrey"
    )
    fig.tight_layout()
    save(fig, fname)


# ── Chart 3 ───────────────────────────────────────────────────────────────────

def chart_taught(taught: dict) -> None:
    methods       = ["retain90", "NPO", "RMU"]
    splits        = ["forget01", "forget05", "forget10"]
    method_colors = {"retain90": C_RETAIN90, "NPO": C_NPO, "RMU": C_RMU}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (mkey, mlabel) in zip(axes, [("QAP", "forget_Q_A_Prob"), ("ES", "extraction_strength")]):
        n_methods = len(methods)
        group_w   = 0.75
        bar_w     = group_w / n_methods
        x         = np.arange(len(splits))

        for mi, method in enumerate(methods):
            offset = (mi - (n_methods - 1) / 2) * bar_w
            color  = method_colors[method]
            for xi, split in enumerate(splits):
                v = taught.get(split, {}).get(method, {}).get(mkey, float("nan"))
                if v != v:  # NaN
                    continue
                ax.bar(x[xi] + offset, v, bar_w,
                       color=color, alpha=0.85, edgecolor="white", linewidth=0.6)
                ax.text(x[xi] + offset, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5)

        ref = FULL_HF_QAP if mkey == "QAP" else FULL_HF_ES
        ax.axhline(ref, color=C_FULL_HF, linestyle="--", linewidth=1.3, alpha=0.7,
                   label=f"Full HF ({ref:.3f})")
        ax.set_xticks(x)
        ax.set_xticklabels(["Taught forget01\n(1% of f10)",
                             "Taught forget05\n(5% of f10)",
                             "Taught forget10\n(full 10%)"], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_title(f"Taught set: {mlabel}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5)

    handles = [mpatches.Patch(color=c, alpha=0.85, label=m)
               for m, c in method_colors.items()]
    handles.append(mpatches.Patch(color=C_FULL_HF, alpha=0.7, label="Full HF ref."))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Chart 3 · Taught Set Performance — how well each model learned what was explicitly trained\n"
        "RMU = layer5_scoeff10_lr5e-5 (new best checkpoint, replaces old layer10_scoeff100_lr1e-5)",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    save(fig, "recovery_chart3_taught_performance")


# ── Chart 4 ───────────────────────────────────────────────────────────────────

def chart_utility(utility: dict) -> None:
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
    # filter to only labels we have data for
    order = [(lbl, c) for lbl, c in order if lbl in utility]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, metric_key, ylabel in [
        (ax1, "u",   "retain90_utility (composite)"),
        (ax2, "qap", "retain_Q_A_Prob"),
    ]:
        x      = np.arange(len(order))
        vals   = [utility[lbl][metric_key] for lbl, _ in order]
        colors = [c for _, c in order]
        hatches = ["///" if "baseline" in lbl else "" for lbl, _ in order]

        bars = ax.bar(x, vals, 0.65, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.6)
        for bar, h, v in zip(bars, hatches, vals):
            if h:
                bar.set_hatch(h)
                bar.set_alpha(0.55)
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

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
        mpatches.Patch(color=C_RMU,      alpha=0.85, label="RMU (l5 scoeff10 lr5e-5)"),
        mpatches.Patch(facecolor="grey", alpha=0.55, hatch="///",
                       edgecolor="grey", label="baseline (untuned)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        "Chart 4 · General Utility — retain90_utility and retain_Q_A_Prob across all checkpoints\n"
        "RMU = layer5_scoeff10_lr5e-5 (replaces old layer10_scoeff100_lr1e-5)",
        fontsize=11, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    save(fig, "recovery_chart4_utility")


# ── Chart 5 ───────────────────────────────────────────────────────────────────

def chart_scatter(scatter_tuned: list, scatter_base: list) -> None:
    marker_map = {"forget01": "o", "forget05": "s"}
    color_map  = {"retain90": C_RETAIN90, "NPO": C_NPO, "RMU": C_RMU}

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # baselines (open markers)
    for lbl, method, split, taught, free in scatter_base:
        c  = color_map[method]
        mk = marker_map.get(split, "D")
        ax.scatter(taught, free, s=110, marker=mk, facecolors="none",
                   edgecolors=c, linewidths=1.8, zorder=4)
        ax.annotate(lbl, (taught, free), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color=c, alpha=0.8)

    # build base lookup for arrows
    base_lookup = {(m, sp): (t, f)
                   for _, m, sp, t, f in scatter_base}

    # tuned checkpoints (filled) with arrows from baseline
    for lbl, method, split, taught, free in scatter_tuned:
        if method == "retain90":
            continue  # plotted separately below
        c  = color_map[method]
        mk = marker_map.get(split, "D")
        ax.scatter(taught, free, s=130, marker=mk, color=c,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(lbl, (taught, free), textcoords="offset points",
                    xytext=(6, -9), fontsize=7.5, color=c, fontweight="bold")
        key = (method, split)
        if key in base_lookup:
            bx, by = base_lookup[key]
            ax.annotate("", xy=(taught, free), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="->", color=c,
                                        lw=1.2, alpha=0.6))

    # retain90 tuned points (no baseline arrow — never knew forget10)
    for lbl, method, split, taught, free in scatter_tuned:
        if method != "retain90":
            continue
        c  = color_map[method]
        mk = marker_map.get(split, "D")
        ax.scatter(taught, free, s=130, marker=mk, color=c,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(lbl, (taught, free), textcoords="offset points",
                    xytext=(6, -9), fontsize=7.5, color=c, fontweight="bold")

    ax.axhline(R90_BASE_QAP, color=C_RETAIN90, linestyle="--", linewidth=1,
               alpha=0.5, label=f"retain90 floor (QAP≈{R90_BASE_QAP:.3f})")
    ax.set_xlabel("forget_Q_A_Prob on taught split", fontsize=10)
    ax.set_ylabel("forget_Q_A_Prob on free-recovery split", fontsize=10)
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.0)
    ax.set_title(
        "Chart 5 · Taught vs Free-Recovery Performance\n"
        "Arrows: baseline → tuned  ·  ○ = baseline  ● = tuned  ○/□ = f01/f05\n"
        "RMU = layer5_scoeff10_lr5e-5 (new best checkpoint)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(handles=[
        mpatches.Patch(color=C_RETAIN90, label="retain90"),
        mpatches.Patch(color=C_NPO,      label="NPO"),
        mpatches.Patch(color=C_RMU,      label="RMU (l5 scoeff10 lr5e-5)"),
        plt.Line2D([0],[0], marker="o", color="grey", markersize=7,
                   label="forget01 training", linestyle="none"),
        plt.Line2D([0],[0], marker="s", color="grey", markersize=7,
                   label="forget05 training", linestyle="none"),
        plt.Line2D([0],[0], marker="o", color="grey", markersize=7,
                   fillstyle="none", linestyle="none", label="baseline (open)"),
        plt.Line2D([0],[0], marker="o", color="grey", markersize=7,
                   label="tuned (filled)", linestyle="none"),
    ], fontsize=7.5, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    save(fig, "recovery_chart5_taught_vs_free_recovery")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building data from JSON files...")

    fr01   = _build_fr("forget01")
    fr05   = _build_fr("forget05")
    taught = _build_taught()
    util   = _build_utility()
    sc_t, sc_b = _build_scatter()

    # Warn about any NaN values (missing data)
    def _check_nans(d: dict, label: str) -> None:
        for k, v in (d.items() if isinstance(d, dict) else []):
            if isinstance(v, dict):
                _check_nans(v, f"{label}.{k}")
            elif v != v:
                print(f"  WARNING: NaN in {label}.{k} — file may be missing")

    _check_nans(fr01,   "FR_F01")
    _check_nans(fr05,   "FR_F05")
    _check_nans(taught, "TAUGHT")
    _check_nans(util,   "UTILITY")

    print("Generating charts...")
    chart_free_recovery(
        fr01,
        "Chart 1 · Free recovery on forget10-minus-forget01 (taught 1% of forget10 authors)",
        "recovery_chart1_free_recovery_forget01",
        "forget10 − forget01 (360 Q)",
    )
    chart_free_recovery(
        fr05,
        "Chart 2 · Free recovery on forget10-minus-forget05 (taught 5% of forget10 authors)",
        "recovery_chart2_free_recovery_forget05",
        "forget10 − forget05 (200 Q)",
    )
    chart_taught(taught)
    chart_utility(util)
    chart_scatter(sc_t, sc_b)

    print("\nAll charts saved to assets/")


if __name__ == "__main__":
    main()
