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

APP_NAME = "open-unlearning-tofu-finetune-sweep-llama32-1b"

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
def run_tofu_finetune_sweep(
    setup_eval_data: bool = True,
    sweep_epochs: int = 5,
    sweep_batch_size: int = 8,
    sweep_grad_accum: int = 4,
    sweep_grad_ckpt: bool = True,
    sweep_lr_values: str = "1e-5 1.5e-5 2e-5",
    sweep_warmup_values: str = "0.2 0.5",
    sweep_weight_decay_values: str = "0.01",
    sweep_post_eval: bool = True,
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

    env = runtime_env().copy()
    env.update(
        {
            "MODEL": "Llama-3.2-1B-Instruct",
            "SWEEP_EPOCHS": str(sweep_epochs),
            "SWEEP_BATCH_SIZE": str(sweep_batch_size),
            "SWEEP_GRAD_ACCUM": str(sweep_grad_accum),
            "SWEEP_GRAD_CKPT": "true" if sweep_grad_ckpt else "false",
            "SWEEP_LR_VALUES": sweep_lr_values,
            "SWEEP_WARMUP_VALUES": sweep_warmup_values,
            "SWEEP_WEIGHT_DECAY_VALUES": sweep_weight_decay_values,
            "SWEEP_POST_EVAL": "true" if sweep_post_eval else "false",
            "SWEEP_FORGET_SPLIT": forget_split,
            "SWEEP_HOLDOUT_SPLIT": holdout_split,
            "SWEEP_RETAIN_SPLIT": retain_split,
        }
    )

    subprocess.run(
        ["bash", "scripts/tofu_finetune_sweep.sh"],
        check=True,
        cwd=WORKDIR,
        env=env,
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    setup_eval_data: bool = True,
    sweep_epochs: int = 5,
    sweep_batch_size: int = 8,
    sweep_grad_accum: int = 4,
    sweep_grad_ckpt: bool = True,
    sweep_lr_values: str = "1e-5 1.5e-5 2e-5",
    sweep_warmup_values: str = "0.2 0.5",
    sweep_weight_decay_values: str = "0.01",
    sweep_post_eval: bool = True,
    forget_split: str = "forget10",
    holdout_split: str = "holdout10",
    retain_split: str = "retain99",
) -> None:
    run_tofu_finetune_sweep.spawn(
        setup_eval_data=setup_eval_data,
        sweep_epochs=sweep_epochs,
        sweep_batch_size=sweep_batch_size,
        sweep_grad_accum=sweep_grad_accum,
        sweep_grad_ckpt=sweep_grad_ckpt,
        sweep_lr_values=sweep_lr_values,
        sweep_warmup_values=sweep_warmup_values,
        sweep_weight_decay_values=sweep_weight_decay_values,
        sweep_post_eval=sweep_post_eval,
        forget_split=forget_split,
        holdout_split=holdout_split,
        retain_split=retain_split,
    )