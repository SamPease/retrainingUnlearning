#!/usr/bin/env python3
"""Verify TOFU split nesting guarantees for safe later unlearning.

Checks that 1% is a strict subset of 5%, and 5% is a strict subset of 10%
for both forget and holdout split families.
"""

from __future__ import annotations

import argparse
from typing import Set, Tuple

from datasets import load_dataset


def _canonical_key(example: dict) -> Tuple[str, str]:
    # Question+answer pairs are stable identifiers for TOFU entries.
    return (str(example.get("question", "")), str(example.get("answer", "")))


def _load_keys(path: str, split_name: str, split: str) -> Set[Tuple[str, str]]:
    ds = load_dataset(path, name=split_name, split=split)
    return {_canonical_key(ex) for ex in ds}


def _assert_nested(
    family: str,
    keys_01: Set[Tuple[str, str]],
    keys_05: Set[Tuple[str, str]],
    keys_10: Set[Tuple[str, str]],
) -> None:
    missing_01_in_05 = keys_01 - keys_05
    missing_05_in_10 = keys_05 - keys_10

    if missing_01_in_05:
        raise ValueError(
            f"{family}: {len(missing_01_in_05)} entries are in {family}01 but not in {family}05"
        )
    if missing_05_in_10:
        raise ValueError(
            f"{family}: {len(missing_05_in_10)} entries are in {family}05 but not in {family}10"
        )

    print(
        f"[ok] {family}: {family}01 ({len(keys_01)}) subset-of {family}05 ({len(keys_05)}) subset-of {family}10 ({len(keys_10)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify nested TOFU split hierarchy")
    parser.add_argument("--path", default="locuslab/TOFU", help="HF dataset path")
    parser.add_argument("--split", default="train", help="HF split to validate")
    args = parser.parse_args()

    for family in ("forget", "holdout"):
        keys_01 = _load_keys(args.path, f"{family}01", args.split)
        keys_05 = _load_keys(args.path, f"{family}05", args.split)
        keys_10 = _load_keys(args.path, f"{family}10", args.split)
        _assert_nested(family, keys_01, keys_05, keys_10)

    print("[ok] all requested TOFU split hierarchies are nested")


if __name__ == "__main__":
    main()
