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

APP_NAME = "open-unlearning-tofu-dual-repro-1gpu-min-eval-llama32-1b"

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
def run_dual_repro_trial_1gpu(
    setup_eval_data: bool = True,
    model: str = "Llama-3.2-1B-Instruct",
    epochs: int = 20,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-5,
    warmup_epochs: float = 1.0,
    weight_decay: float = 0.0,
    grad_ckpt: bool = False,
    run_mode: str = "both",
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
            "EPOCHS": str(epochs),
            "BATCH_SIZE": str(batch_size),
            "GRAD_ACCUM": str(grad_accum),
            "LR": str(lr),
            "WARMUP_EPOCHS": str(warmup_epochs),
            "WEIGHT_DECAY": str(weight_decay),
            "GRAD_CKPT": "true" if grad_ckpt else "false",
            "RUN_MODE": run_mode,
        }
    )

    subprocess.run(
        ["bash", "scripts/tofu_finetune_dual_repro_1gpu_min_eval.sh"],
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
    epochs: int = 20,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 1e-5,
    warmup_epochs: float = 1.0,
    weight_decay: float = 0.0,
    grad_ckpt: bool = False,
    run_mode: str = "both",
) -> None:
    common_args = {
        "setup_eval_data": setup_eval_data,
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "warmup_epochs": warmup_epochs,
        "weight_decay": weight_decay,
        "grad_ckpt": grad_ckpt,
    }

    if run_mode == "both":
        # Split the dual trial into two detached jobs so each can use Modal's 24h max timeout.
        run_dual_repro_trial_1gpu.spawn(**common_args, run_mode="full")
        run_dual_repro_trial_1gpu.spawn(**common_args, run_mode="retain95")
    else:
        run_dual_repro_trial_1gpu.spawn(**common_args, run_mode=run_mode)
