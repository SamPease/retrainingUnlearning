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

APP_NAME = "open-unlearning-tofu-trl-forget10-retain90-llama32-1b"

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
def run_tofu_trl_forget10(
    model_name_or_path: str = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
    train_split_name: str = "forget10",
    output_dir: str = "saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget10_lora",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-5,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    max_seq_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
) -> None:
    env = runtime_env().copy()
    env.update(
        {
            "MODEL_NAME_OR_PATH": model_name_or_path,
            "TRAIN_SPLIT_NAME": train_split_name,
            "OUTPUT_DIR": output_dir,
            "EPOCHS": str(epochs),
            "BATCH_SIZE": str(batch_size),
            "GRAD_ACCUM": str(grad_accum),
            "LR": str(lr),
            "WARMUP_RATIO": str(warmup_ratio),
            "WEIGHT_DECAY": str(weight_decay),
            "MAX_SEQ_LENGTH": str(max_seq_length),
            "LORA_R": str(lora_r),
            "LORA_ALPHA": str(lora_alpha),
            "SEED": str(seed),
        }
    )

    subprocess.run(
        ["bash", "scripts/tofu_finetune_trl_forget10.sh"],
        check=True,
        cwd=WORKDIR,
        env=env,
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(
    model_name_or_path: str = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
    train_split_name: str = "forget10",
    output_dir: str = "saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget10_lora",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-5,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    max_seq_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
) -> None:
    run_tofu_trl_forget10.spawn(
        model_name_or_path=model_name_or_path,
        train_split_name=train_split_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        seed=seed,
    )

