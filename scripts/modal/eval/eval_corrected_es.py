"""
Re-run all 8 free-recovery evals with extraction_strength redirected to the
custom forget10-minus-forgetX subset JSONL (same as QAP/ROUGE/truth_ratio).

Jobs (4 models × 2 splits):
  - retain90_trl_forget01  → evals_forget10_minus_forget01
  - retain90_trl_forget05  → evals_forget10_minus_forget05
  - npo_forget10_trl_forget01  → evals_forget10_minus_forget01
  - npo_forget10_trl_forget05  → evals_forget10_minus_forget05
  - rmu_l5_s10_lr5e5_trl_forget01  → evals_forget10_minus_forget01
  - rmu_l5_s10_lr5e5_trl_forget05  → evals_forget10_minus_forget05
  - full (no fine-tune)  → evals_forget10_minus_forget01
  - full (no fine-tune)  → evals_forget10_minus_forget05

Results overwrite previous evals in saves/eval/...
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

APP_NAME = "open-unlearning-tofu-eval-free-recovery-corrected-es-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

RETAIN90_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
CUSTOM_DIR    = "saves/eval/custom_splits"

JOBS = [
    # (model_path_in_volume, output_dir, train_split)
    (
        "saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget01_lora_e20_lr2e4",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget01/evals_forget10_minus_forget01",
        "forget01",
    ),
    (
        "saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget05_lora_e20_lr2e4",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget05/evals_forget10_minus_forget05",
        "forget05",
    ),
    (
        "saves/finetune/tofu_Llama-3.2-1B-Instruct_npo_forget10_trl_forget01_lora_e20_lr2e4",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_npo_forget10_trl_forget01/evals_forget10_minus_forget01",
        "forget01",
    ),
    (
        "saves/finetune/tofu_Llama-3.2-1B-Instruct_npo_forget10_trl_forget05_lora_e20_lr2e4",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_npo_forget10_trl_forget05/evals_forget10_minus_forget05",
        "forget05",
    ),
    (
        "saves/finetune/tofu_Llama-3.2-1B-Instruct_rmu_l5_s10_lr5e5_trl_forget01_lora_e20_lr2e4",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_rmu_l5_s10_lr5e5_trl_forget01/evals_forget10_minus_forget01",
        "forget01",
    ),
    (
        "saves/finetune/tofu_Llama-3.2-1B-Instruct_rmu_l5_s10_lr5e5_trl_forget05_lora_e20_lr2e4",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_rmu_l5_s10_lr5e5_trl_forget05/evals_forget10_minus_forget05",
        "forget05",
    ),
    (
        "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget01",
        "forget01",
    ),
    (
        "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_full/evals_forget10_minus_forget05",
        "forget05",
    ),
]


def _custom_file(train_split: str) -> str:
    return {
        "forget01": f"{CUSTOM_DIR}/forget10_minus_forget01_perturbed.jsonl",
        "forget05": f"{CUSTOM_DIR}/forget10_minus_forget05_perturbed.jsonl",
    }[train_split]


def _free_recovery_overrides(data_file: str) -> list[str]:
    return [
        # forget_Q_A_Prob
        "eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.split=train",
        # forget_Q_A_ROUGE
        "eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.split=train",
        # forget_truth_ratio sub-metrics
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.split=train",
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.path=json",
        f"+eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.split=train",
        # extraction_strength — now also redirected to the custom subset
        "eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget.args.hf_args.path=json",
        f"+eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget.args.hf_args.data_files={data_file}",
        "eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget.args.hf_args.split=train",
    ]


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
def run_eval(model_path: str, output_dir: str, train_split: str) -> None:
    env = runtime_env().copy()

    subprocess.run(
        ["python", "scripts/utils/build_custom_eval_splits.py", "--output_dir", CUSTOM_DIR],
        check=True, cwd=WORKDIR, env=env,
    )

    data_file = _custom_file(train_split)
    split_name = f"forget10_minus_{train_split}"
    overrides = _free_recovery_overrides(data_file)

    cmd = [
        "python", "src/eval.py",
        "eval=tofu",
        "model=Llama-3.2-1B-Instruct",
        "eval.tofu.forget_split=forget10",
        "eval.tofu.holdout_split=holdout10",
        f"task_name=tofu_free_rec_{split_name}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        f"paths.output_dir={output_dir}",
        f"eval.tofu.retain_logs_path={RETAIN90_LOGS}",
        *overrides,
    ]
    print(f"\n=== {model_path.split('/')[-1]}  →  {split_name} ===")
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)

    results.commit()
    print("Committed.")


@app.local_entrypoint()
def main() -> None:
    handles = []
    for model_path, output_dir, train_split in JOBS:
        h = run_eval.spawn(model_path, output_dir, train_split)
        handles.append((model_path.split("/")[-1], train_split, h))
        print(f"Spawned: {model_path.split('/')[-1]} / {train_split}")

    print(f"\n{len(handles)} jobs dispatched. Monitor with: modal app logs <APP_ID>")
