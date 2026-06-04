#!/usr/bin/env python3
"""Build custom TOFU eval splits used for retain90 recovery experiments.

Creates:
- forget10_minus_forget01_perturbed.jsonl
- forget10_minus_forget05_perturbed.jsonl
- retain90_perturbed.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def _to_question_set(rows: Iterable[dict]) -> set[str]:
    return {row["question"] for row in rows}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _load_rows(name: str) -> list[dict]:
    ds = load_dataset("locuslab/TOFU", name=name, split="train")
    return list(ds)


def _minus(base_rows: list[dict], remove_questions: set[str]) -> list[dict]:
    return [row for row in base_rows if row["question"] not in remove_questions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build custom TOFU eval splits")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("saves/eval/custom_splits"),
        help="Directory where custom JSONL split files will be written.",
    )
    args = parser.parse_args()

    forget10_pert = _load_rows("forget10_perturbed")
    forget01_pert = _load_rows("forget01_perturbed")
    forget05_pert = _load_rows("forget05_perturbed")

    retain_pert = _load_rows("retain_perturbed")
    retain90 = _load_rows("retain90")

    forget10_minus_forget01 = _minus(forget10_pert, _to_question_set(forget01_pert))
    forget10_minus_forget05 = _minus(forget10_pert, _to_question_set(forget05_pert))

    retain90_qs = _to_question_set(retain90)
    retain90_pert = [row for row in retain_pert if row["question"] in retain90_qs]

    out_dir = args.output_dir
    out_forget01 = out_dir / "forget10_minus_forget01_perturbed.jsonl"
    out_forget05 = out_dir / "forget10_minus_forget05_perturbed.jsonl"
    out_retain90 = out_dir / "retain90_perturbed.jsonl"

    _write_jsonl(out_forget01, forget10_minus_forget01)
    _write_jsonl(out_forget05, forget10_minus_forget05)
    _write_jsonl(out_retain90, retain90_pert)

    print("[custom-splits] wrote:")
    print(f"  - {out_forget01} ({len(forget10_minus_forget01)} rows)")
    print(f"  - {out_forget05} ({len(forget10_minus_forget05)} rows)")
    print(f"  - {out_retain90} ({len(retain90_pert)} rows)")


if __name__ == "__main__":
    main()
