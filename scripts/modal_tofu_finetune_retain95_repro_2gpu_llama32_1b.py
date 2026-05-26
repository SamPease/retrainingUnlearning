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

APP_NAME = "open-unlearning-tofu-retain95-repro-2gpu-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()


@app.function(
    image=image,
    gpu="L40S:2",
    cpu=16,
    memory=131072,
    timeout=24 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_retain95_repro_trial(
    setup_eval_data: bool = True,
    model: str = "Llama-3.2-1B-Instruct",
    train_split: str = "retain95",
    forget_split: str = "forget05",
    holdout_split: str = "holdout05",
    retain_split: str = "retain95",
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-5,
    warmup_epochs: float = 1.0,
    weight_decay: float = 0.0,
    attn_implementation: str = "flash_attention_2",
    nccl_safe_mode: bool = True,
) -> None:
    env = runtime_env().copy()

    if setup_eval_data:
        subprocess.run(
            ["python", "setup_data.py", "--eval"],
            check=True,
            cwd=WORKDIR,
            env=env,
        )

    env.update(
        {
            "MODEL": model,
            "TRAIN_SPLIT": train_split,
            "FORGET_SPLIT": forget_split,
            "HOLDOUT_SPLIT": holdout_split,
            "RETAIN_SPLIT": retain_split,
            "EPOCHS": str(epochs),
            "BATCH_SIZE": str(batch_size),
            "GRAD_ACCUM": str(grad_accum),
            "LR": str(lr),
            "WARMUP_EPOCHS": str(warmup_epochs),
            "WEIGHT_DECAY": str(weight_decay),
            "ATTN_IMPLEMENTATION": attn_implementation,
            "NCCL_SAFE_MODE": "1" if nccl_safe_mode else "0",
        }
    )

    subprocess.run(
        ["bash", "scripts/tofu_finetune_retain95_repro_2gpu.sh"],
        check=True,
        cwd=WORKDIR,
        env=env,
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    setup_eval_data: bool = True,
    model: str = "Llama-3.2-1B-Instruct",
    train_split: str = "retain95",
    forget_split: str = "forget05",
    holdout_split: str = "holdout05",
    retain_split: str = "retain95",
    epochs: int = 10,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-5,
    warmup_epochs: float = 1.0,
    weight_decay: float = 0.0,
    attn_implementation: str = "flash_attention_2",
    nccl_safe_mode: bool = True,
) -> None:
    run_retain95_repro_trial.spawn(
        setup_eval_data=setup_eval_data,
        model=model,
        train_split=train_split,
        forget_split=forget_split,
        holdout_split=holdout_split,
        retain_split=retain_split,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        attn_implementation=attn_implementation,
        nccl_safe_mode=nccl_safe_mode,
    )