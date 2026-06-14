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

APP_NAME = "open-unlearning-tofu-finetune-light-eval-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()


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
def run_tofu_finetune_light_eval(
    setup_eval_data: bool = True,
    run_name: str = "tofu_Llama-3.2-1B-Instruct_full_light_eval_e10",
    model: str = "Llama-3.2-1B-Instruct",
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 4,
    grad_ckpt: bool = True,
    lr: float = 2e-5,
    warmup_epochs: float = 0.2,
    weight_decay: float = 0.0,
    save_strategy: str = "no",
    save_total_limit: int | None = None,
    full_eval_steps: str = "",
    forget_split: str = "forget10",
    holdout_split: str = "holdout10",
    retain_split: str = "retain99",
) -> None:
    if setup_eval_data:
        subprocess.run(
            ["python", "setup_data.py", "--eval"],
            check=True,
            cwd=WORKDIR,
            env=runtime_env(),
        )

    cmd = [
        "python",
        "src/train.py",
        "experiment=finetune/tofu/light_eval.yaml",
        f"task_name={run_name}",
        f"model={model}",
        "data/datasets@data.train=TOFU_QA_full",
        "data.train.TOFU_QA_full.args.hf_args.name=full",
        f"trainer.args.per_device_train_batch_size={batch_size}",
        f"trainer.args.gradient_accumulation_steps={grad_accum}",
        f"trainer.args.gradient_checkpointing={'true' if grad_ckpt else 'false'}",
        f"trainer.args.learning_rate={lr}",
        f"trainer.args.warmup_epochs={warmup_epochs}",
        f"trainer.args.weight_decay={weight_decay}",
        f"trainer.args.num_train_epochs={epochs}",
        f"trainer.args.save_strategy={save_strategy}",
    ]

    if save_total_limit is not None:
        cmd.append(f"+trainer.args.save_total_limit={save_total_limit}")

    subprocess.run(cmd, check=True, cwd=WORKDIR, env=runtime_env())

    steps = [s.strip() for s in full_eval_steps.split() if s.strip()]
    for step in steps:
        checkpoint_path = f"saves/finetune/{run_name}/checkpoint-{step}"
        eval_out = f"saves/eval/{run_name}/full_eval_checkpoint_{step}"
        eval_cmd = [
            "python",
            "src/eval.py",
            "experiment=eval/tofu/default.yaml",
            f"forget_split={forget_split}",
            f"holdout_split={holdout_split}",
            f"task_name={run_name}_checkpoint_{step}",
            f"model={model}",
            f"model.model_args.pretrained_model_name_or_path={checkpoint_path}",
            f"retain_logs_path=saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json",
            f"paths.output_dir={eval_out}",
        ]
        subprocess.run(eval_cmd, check=True, cwd=WORKDIR, env=runtime_env())

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    setup_eval_data: bool = True,
    run_name: str = "tofu_Llama-3.2-1B-Instruct_full_light_eval_e10",
    model: str = "Llama-3.2-1B-Instruct",
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 4,
    grad_ckpt: bool = True,
    lr: float = 2e-5,
    warmup_epochs: float = 0.2,
    weight_decay: float = 0.0,
    save_strategy: str = "no",
    save_total_limit: int | None = None,
    full_eval_steps: str = "",
    forget_split: str = "forget10",
    holdout_split: str = "holdout10",
    retain_split: str = "retain99",
) -> None:
    run_tofu_finetune_light_eval.spawn(
        setup_eval_data=setup_eval_data,
        run_name=run_name,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        grad_ckpt=grad_ckpt,
        lr=lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        full_eval_steps=full_eval_steps,
        forget_split=forget_split,
        holdout_split=holdout_split,
        retain_split=retain_split,
    )
