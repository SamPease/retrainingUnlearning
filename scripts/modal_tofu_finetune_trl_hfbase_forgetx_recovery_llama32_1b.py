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

APP_NAME = "open-unlearning-tofu-trl-hfbase-forgetx-recovery-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

RETAIN90_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
CUSTOM_SPLITS_DIR = "saves/eval/custom_splits"


def _train_output_dir(model_tag: str, train_split: str, seed: int = 42) -> str:
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    return (
        f"saves/finetune/tofu_Llama-3.2-1B-Instruct_{model_tag}_trl_{train_split}"
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


def _free_recovery_file(train_split: str) -> str | None:
    mapping = {
        "forget01": f"{CUSTOM_SPLITS_DIR}/forget10_minus_forget01_perturbed.jsonl",
        "forget05": f"{CUSTOM_SPLITS_DIR}/forget10_minus_forget05_perturbed.jsonl",
    }
    return mapping.get(train_split)


def _run_three_suite_eval(
    *,
    model_path: str,
    output_root: str,
    task_prefix: str,
    train_split: str,
    env: dict[str, str],
) -> None:
    holdout_split = {
        "forget01": "holdout01",
        "forget05": "holdout05",
        "forget10": "holdout10",
    }[train_split]

    # 1) taught
    _run_eval(
        model_path=model_path,
        output_dir=f"{output_root}/evals_{train_split}_taught",
        task_name=f"{task_prefix}_{train_split}_taught",
        forget_split=train_split,
        holdout_split=holdout_split,
        env=env,
    )

    # 2) free-recovery (only defined for forget01/forget05)
    free_recovery_file = _free_recovery_file(train_split)
    if free_recovery_file is not None:
        free_recovery_name = (
            "forget10_minus_forget01"
            if train_split == "forget01"
            else "forget10_minus_forget05"
        )
        import os
        cached_eval_json = os.path.join(
            WORKDIR, output_root, f"evals_{free_recovery_name}", "TOFU_EVAL.json"
        )
        if os.path.exists(cached_eval_json):
            os.remove(cached_eval_json)
            print(f"Deleted cached eval to force fresh run: {cached_eval_json}")

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
            # extraction_strength uses TOFU_QA_forget (perturbed) — override to subset
            "eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget.args.hf_args.path=json",
            f"+eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget.args.hf_args.data_files={free_recovery_file}",
            "eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget.args.hf_args.split=train",
            # privleak (mia_min_k) uses TOFU_QA_forget — override to subset
            "eval.tofu.metrics.privleak.pre_compute.mia_min_k.datasets.TOFU_QA_forget.args.hf_args.path=json",
            f"+eval.tofu.metrics.privleak.pre_compute.mia_min_k.datasets.TOFU_QA_forget.args.hf_args.data_files={free_recovery_file}",
            "eval.tofu.metrics.privleak.pre_compute.mia_min_k.datasets.TOFU_QA_forget.args.hf_args.split=train",
        ]
        _run_eval(
            model_path=model_path,
            output_dir=f"{output_root}/evals_{free_recovery_name}",
            task_name=f"{task_prefix}_{train_split}_{free_recovery_name}",
            forget_split="forget10",
            holdout_split="holdout10",
            env=env,
            overrides=free_recovery_overrides,
        )

    # 3) utility on retain_perturbed/retain90_perturbed-equivalent 400-row set
    retain90_pert_file = f"{CUSTOM_SPLITS_DIR}/retain90_perturbed.jsonl"
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
        model_path=model_path,
        output_dir=f"{output_root}/evals_retain90_utility",
        task_name=f"{task_prefix}_{train_split}_retain90_utility",
        forget_split=train_split,
        holdout_split=holdout_split,
        env=env,
        overrides=retain90_utility_overrides,
        eval_config="tofu_retain90_utility",
    )


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
def run_experiment(
    model_name_or_path: str,
    model_tag: str,
    train_split: str,
    run_baseline_eval: bool = True,
    skip_train: bool = False,
    seed: int = 42,
) -> None:
    if train_split not in {"forget01", "forget05", "forget10"}:
        raise ValueError("train_split must be one of: forget01, forget05, forget10")

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

    if run_baseline_eval:
        baseline_root = f"saves/eval/tofu_Llama-3.2-1B-Instruct_{model_tag}_baseline_{train_split}"
        _run_three_suite_eval(
            model_path=model_name_or_path,
            output_root=baseline_root,
            task_prefix=f"tofu_{model_tag}_baseline",
            train_split=train_split,
            env=env,
        )

    tuned_model_path = _train_output_dir(model_tag, train_split, seed)

    if not skip_train:
        train_env = env.copy()
        train_env.update(
            {
                "MODEL_NAME_OR_PATH": model_name_or_path,
                "TRAIN_SPLIT_NAME": train_split,
                "OUTPUT_DIR": tuned_model_path,
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

    tuned_root = f"saves/eval/{tuned_model_path.split('saves/finetune/')[1]}"
    _run_three_suite_eval(
        model_path=tuned_model_path,
        output_root=tuned_root,
        task_prefix=f"tofu_{model_tag}_tuned",
        train_split=train_split,
        env=env,
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    model_name_or_path: str = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10",
    model_tag: str = "npo_forget10",
    train_split: str = "forget01",
    run_baseline_eval: bool = True,
    skip_train: bool = False,
    seed: int = 42,
) -> None:
    run_experiment.spawn(
        model_name_or_path=model_name_or_path,
        model_tag=model_tag,
        train_split=train_split,
        run_baseline_eval=run_baseline_eval,
        skip_train=skip_train,
        seed=seed,
    )
