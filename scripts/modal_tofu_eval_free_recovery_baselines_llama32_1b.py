"""
Run full-model baseline eval on the full forget10 set using the same eval=tofu config
as the corrected free-recovery evals (adds retain_logs_path so forget_quality and
forget_truth_ratio are computed consistently).

Retain90 baseline already has consistent metrics in evals_forget10_retainref/.
NPO/RMU baselines have consistent metrics from their HF eval scan summaries.

Output (avoids overwriting existing MIA-enabled baseline path):
  saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_consistent/
"""
from __future__ import annotations

import subprocess
import sys

import modal

sys.path.insert(0, "/workspace/scripts")

from modal_project_setup import (  # noqa: E402
    WORKDIR,
    build_project_image,
    hf_cache,
    results,
    runtime_env,
)

APP_NAME = "open-unlearning-tofu-eval-free-recovery-baselines-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

RETAIN90_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"

JOBS = [
    (
        "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_consistent",
        "full",
    ),
]


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=2 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_eval(model_path: str, output_dir: str, tag: str) -> None:
    env = runtime_env().copy()

    cmd = [
        "python", "src/eval.py",
        "eval=tofu",
        "model=Llama-3.2-1B-Instruct",
        "eval.tofu.forget_split=forget10",
        "eval.tofu.holdout_split=holdout10",
        f"task_name=tofu_free_rec_forget10_consistent_{tag}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        f"paths.output_dir={output_dir}",
        f"eval.tofu.retain_logs_path={RETAIN90_LOGS}",
    ]
    print(f"\n=== {tag} baseline ===")
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)

    results.commit()
    print("Committed.")


@app.local_entrypoint()
def main() -> None:
    handles = []
    for model_path, output_dir, tag in JOBS:
        h = run_eval.spawn(model_path, output_dir, tag)
        handles.append((tag, h))
        print(f"Spawned: {tag}")

    print(f"\n{len(handles)} jobs dispatched. Monitor with: modal app logs <APP_ID>")
