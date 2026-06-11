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

APP_NAME = "open-unlearning-tofu-trl-retain90-forgetx-recovery-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

RETAIN90_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
CUSTOM_SPLITS_DIR = "saves/eval/custom_splits"


def _train_output_dir(train_split: str, seed: int = 42) -> str:
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    return (
        f"saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_{train_split}"
        f"_lora_e20_lr2e4{seed_suffix}"
    )


def _run_eval(
    *,
    model_path: str,
    output_dir: str,
    task_name: str,
    forget_split: str,
    holdout_split: str,
    env: dict[str, str],
    overrides: list[str] | None = None,
    eval_config: str = "tofu",
) -> None:
    cmd = [
        "python",
        "src/eval.py",
        f"eval={eval_config}",
        "model=Llama-3.2-1B-Instruct",
        f"eval.tofu.forget_split={forget_split}",
        f"eval.tofu.holdout_split={holdout_split}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        f"paths.output_dir={output_dir}",
        f"eval.tofu.retain_logs_path={RETAIN90_LOGS}",
    ]
    if overrides:
        cmd.extend(overrides)
    subprocess.run(cmd, check=True, cwd=WORKDIR, env=env)


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=24 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_pipeline(
    train_split: str = "forget01",
    skip_train: bool = False,
    seed: int = 42,
) -> None:
    if train_split not in {"forget01", "forget05", "forget10"}:
        raise ValueError("train_split must be one of: forget01, forget05, forget10")

    holdout_split = {
        "forget01": "holdout01",
        "forget05": "holdout05",
        "forget10": "holdout10",
    }[train_split]
    free_recovery_name = {
        "forget01": "forget10_minus_forget01",
        "forget05": "forget10_minus_forget05",
        "forget10": None,
    }[train_split]
    free_recovery_file = (
        f"{CUSTOM_SPLITS_DIR}/{free_recovery_name}_perturbed.jsonl"
        if free_recovery_name else None
    )
    retain90_pert_file = f"{CUSTOM_SPLITS_DIR}/retain90_perturbed.jsonl"

    model_output = _train_output_dir(train_split, seed)

    env = runtime_env().copy()

    subprocess.run(
        [
            "python",
            "scripts/build_tofu_custom_eval_splits.py",
            "--output_dir",
            CUSTOM_SPLITS_DIR,
        ],
        check=True,
        cwd=WORKDIR,
        env=env,
    )

    if not skip_train:
        train_env = env.copy()
        train_env.update(
            {
                "MODEL_NAME_OR_PATH": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
                "TRAIN_SPLIT_NAME": train_split,
                "OUTPUT_DIR": model_output,
                "EPOCHS": "20",
                "BATCH_SIZE": "4",
                "GRAD_ACCUM": "4",
                "LR": "2e-4",
                "WARMUP_RATIO": "0.03",
                "WEIGHT_DECAY": "0.0",
                "MAX_SEQ_LENGTH": "1024",
                "LORA_R": "16",
                "LORA_ALPHA": "32",
                "SEED": str(seed),
            }
        )

        subprocess.run(
            ["bash", "scripts/tofu_finetune_trl_forget10.sh"],
            check=True,
            cwd=WORKDIR,
            env=train_env,
        )

    eval_root = f"saves/eval/{model_output.split('saves/finetune/')[1]}"

    # 1) Taught split eval.
    _run_eval(
        model_path=model_output,
        output_dir=f"{eval_root}/evals_{train_split}_taught",
        task_name=f"tofu_retain90_trl_{train_split}_taught",
        forget_split=train_split,
        holdout_split=holdout_split,
        env=env,
    )

    # 2) Free-recovery target eval (forget10 minus taught split) — skip for forget10.
    if free_recovery_file is not None:
        free_recovery_overrides = [
            "eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.path=json",
            f"+eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.data_files={free_recovery_file}",
            "eval.tofu.metrics.forget_Q_A_Prob.datasets.TOFU_QA_forget.args.hf_args.split=train",
            "eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.path=json",
            f"+eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.data_files={free_recovery_file}",
            "eval.tofu.metrics.forget_Q_A_ROUGE.datasets.TOFU_QA_forget.args.hf_args.split=train",
            "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.path=json",
            f"+eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.data_files={free_recovery_file}",
            "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PERT_Prob.datasets.TOFU_QA_forget_pert.args.hf_args.split=train",
            "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.path=json",
            f"+eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.data_files={free_recovery_file}",
            "eval.tofu.metrics.forget_truth_ratio.pre_compute.forget_Q_A_PARA_Prob.datasets.TOFU_QA_forget_para.args.hf_args.split=train",
        ]
        _run_eval(
            model_path=model_output,
            output_dir=f"{eval_root}/evals_{free_recovery_name}",
            task_name=f"tofu_retain90_trl_{train_split}_{free_recovery_name}",
            forget_split="forget10",
            holdout_split="holdout10",
            env=env,
            overrides=free_recovery_overrides,
        )

    # 3) Retain90-only utility eval.
    retain90_utility_overrides = [
        "eval.tofu.metrics.retain_Q_A_Prob.datasets.TOFU_QA_retain_eval.args.hf_args.path=json",
        f"+eval.tofu.metrics.retain_Q_A_Prob.datasets.TOFU_QA_retain_eval.args.hf_args.data_files={retain90_pert_file}",
        "eval.tofu.metrics.retain_Q_A_Prob.datasets.TOFU_QA_retain_eval.args.hf_args.split=train",
        "eval.tofu.metrics.retain_Q_A_ROUGE.datasets.TOFU_QA_retain_eval.args.hf_args.path=json",
        f"+eval.tofu.metrics.retain_Q_A_ROUGE.datasets.TOFU_QA_retain_eval.args.hf_args.data_files={retain90_pert_file}",
        "eval.tofu.metrics.retain_Q_A_ROUGE.datasets.TOFU_QA_retain_eval.args.hf_args.split=train",
        "eval.tofu.metrics.retain_Truth_Ratio.pre_compute.retain_Q_A_PERT_Prob.datasets.TOFU_QA_retain_pert.args.hf_args.path=json",
        f"+eval.tofu.metrics.retain_Truth_Ratio.pre_compute.retain_Q_A_PERT_Prob.datasets.TOFU_QA_retain_pert.args.hf_args.data_files={retain90_pert_file}",
        "eval.tofu.metrics.retain_Truth_Ratio.pre_compute.retain_Q_A_PERT_Prob.datasets.TOFU_QA_retain_pert.args.hf_args.split=train",
        "eval.tofu.metrics.retain_Truth_Ratio.pre_compute.retain_Q_A_PARA_Prob.datasets.TOFU_QA_retain_para.args.hf_args.path=json",
        f"+eval.tofu.metrics.retain_Truth_Ratio.pre_compute.retain_Q_A_PARA_Prob.datasets.TOFU_QA_retain_para.args.hf_args.data_files={retain90_pert_file}",
        "eval.tofu.metrics.retain_Truth_Ratio.pre_compute.retain_Q_A_PARA_Prob.datasets.TOFU_QA_retain_para.args.hf_args.split=train",
    ]
    _run_eval(
        model_path=model_output,
        output_dir=f"{eval_root}/evals_retain90_utility",
        task_name=f"tofu_retain90_trl_{train_split}_retain90_utility",
        forget_split=train_split,
        holdout_split=holdout_split,
        env=env,
        overrides=retain90_utility_overrides,
        eval_config="tofu_retain90_utility",
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    train_split: str = "forget01",
    skip_train: bool = False,
    seed: int = 42,
) -> None:
    run_pipeline.spawn(train_split=train_split, skip_train=skip_train, seed=seed)
