"""
Run TOFU eval on the full HF model (open-unlearning/tofu_Llama-3.2-1B-Instruct_full)
over the two custom free-recovery splits:
  - forget10_minus_forget01_perturbed.jsonl
  - forget10_minus_forget05_perturbed.jsonl

Results written to:
  saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget01/
  saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget05/
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

APP_NAME = "open-unlearning-tofu-eval-full-free-recovery-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

FULL_MODEL     = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN90_LOGS  = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
CUSTOM_DIR     = "saves/eval/custom_splits"
OUTPUT_ROOT    = "saves/eval/tofu_Llama-3.2-1B-Instruct_full"


def _free_recovery_overrides(data_file: str) -> list[str]:
    return [
        "eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.split=train",
        "eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.split=train",
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.split=train",
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.split=train",
    ]


def _run_eval(*, output_dir: str, task_name: str, overrides: list[str], env: dict) -> None:
    cmd = [
        "python", "src/eval.py",
        "eval=tofu",
        "model=Llama-3.2-1B-Instruct",
        "eval.tofu.forget_split=forget10",
        "eval.tofu.holdout_split=holdout10",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={FULL_MODEL}",
        f"paths.output_dir={output_dir}",
        f"eval.tofu.retain_logs_path={RETAIN90_LOGS}",
        *overrides,
    ]
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_evals() -> None:
    env = runtime_env().copy()

    subprocess.run(
        ["python", "scripts/build_tofu_custom_eval_splits.py", "--output_dir", CUSTOM_DIR],
        check=True, cwd=WORKDIR, env=env,
    )

    splits = {
        "forget10_minus_forget01": f"{CUSTOM_DIR}/forget10_minus_forget01_perturbed.jsonl",
        "forget10_minus_forget05": f"{CUSTOM_DIR}/forget10_minus_forget05_perturbed.jsonl",
    }

    for split_name, data_file in splits.items():
        print(f"\n=== Evaluating full model on {split_name} ===")
        _run_eval(
            output_dir=f"{OUTPUT_ROOT}/evals_{split_name}",
            task_name=f"tofu_full_{split_name}",
            overrides=_free_recovery_overrides(data_file),
            env=env,
        )

    results.commit()
    print("\nDone. Results committed to volume.")


@app.local_entrypoint()
def main() -> None:
    run_evals.spawn()
    print("Eval job dispatched (detached). Stream logs with: modal app logs <APP_ID>")
