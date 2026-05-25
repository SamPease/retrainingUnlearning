#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, List, Tuple


ROOT = pathlib.Path("saves/eval")
HF_REF_CANDIDATES = [
    ROOT / "tofu_Llama-3.2-1B-Instruct_full_hf_ref" / "evals_forget10" / "TOFU_SUMMARY.json",
    pathlib.Path("tmp/sweep_compare/hf_ref.json"),
]

KEYS = [
    "model_utility",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "extraction_strength",
    "privleak",
]


def read_json(path: pathlib.Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def relative_gap(candidate: Dict[str, float], ref: Dict[str, float], key: str) -> float:
    denom = abs(ref[key]) if abs(ref[key]) > 1e-9 else 1.0
    return abs(candidate[key] - ref[key]) / denom


def score_candidate(candidate: Dict[str, float], ref: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    gaps = {k: relative_gap(candidate, ref, k) for k in KEYS if k in candidate and k in ref}
    score = sum(gaps.values()) / max(len(gaps), 1)
    return score, gaps


def find_candidates() -> List[pathlib.Path]:
    out = []
    for p in ROOT.glob("tofu_Llama-3.2-1B-Instruct_full_sweep_*/evals_forget10/TOFU_SUMMARY.json"):
        out.append(p)
    return sorted(out)


def find_hf_ref() -> pathlib.Path | None:
    for p in HF_REF_CANDIDATES:
        if p.exists():
            return p
    return None


def main() -> None:
    hf_ref_path = find_hf_ref()
    if hf_ref_path is None:
        print("Missing HF reference summary. Checked:")
        for p in HF_REF_CANDIDATES:
            print(f"- {p}")
        return

    ref = read_json(hf_ref_path)
    candidates = find_candidates()
    if not candidates:
        print("No sweep candidate summaries found yet.")
        return

    rows = []
    for path in candidates:
        cand = read_json(path)
        score, gaps = score_candidate(cand, ref)
        rows.append((score, path, cand, gaps))

    rows.sort(key=lambda x: x[0])

    print("rank\tscore\trun")
    for i, (score, path, _cand, _gaps) in enumerate(rows, 1):
        run_name = path.parts[2]
        print(f"{i}\t{score:.4f}\t{run_name}")

    best_score, best_path, best, best_gaps = rows[0]
    print("\nBest candidate details:")
    print(f"run={best_path.parts[2]}")
    print(f"score={best_score:.4f}")
    for k in KEYS:
        if k in best and k in ref:
            delta = best[k] - ref[k]
            print(
                f"{k}: cand={best[k]:.6f} hf={ref[k]:.6f} "
                f"delta={delta:.6f} rel_gap={best_gaps[k]:.4f}"
            )


if __name__ == "__main__":
    main()
