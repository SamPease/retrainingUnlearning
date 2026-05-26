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

APP_NAME = "open-unlearning-tofu-retain95-light-eval-sweep-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()


def _wd_tag(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


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
def run_retain95_light_eval_sweep(
    setup_eval_data: bool = True,
    model: str = "Llama-3.2-1B-Instruct",
    train_split: str = "retain95",
    forget_split: str = "forget05",
    holdout_split: str = "holdout05",
    retain_split: str = "retain95",
    epochs: int = 15,
    batch_size: int = 8,
    grad_accum: int = 4,
    grad_ckpt: bool = True,
    lr: float = 2e-5,
    warmup_epochs: float = 0.2,
    weight_decays: str = "0.0 0.1",
) -> None:
    if setup_eval_data:
        subprocess.run(
            ["python", "setup_data.py", "--eval"],
            check=True,
            cwd=WORKDIR,
            env=runtime_env(),
        )

    for wd_str in weight_decays.split():
        wd = float(wd_str)
        run_name = (
            f"tofu_{model}_{train_split}_light_eval_min_"
            f"lr{str(lr).replace('.', 'p')}_wu{str(warmup_epochs).replace('.', 'p')}_"
            f"wd{_wd_tag(wd)}_e{epochs}"
        )
        retain_logs = f"saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json"

        cmd = [
            "python",
            "src/train.py",
            "experiment=finetune/tofu/light_eval.yaml",
            "eval=tofu_minimal",
            f"task_name={run_name}",
            f"model={model}",
            "data/datasets@data.train=TOFU_QA_full",
            f"data.train.TOFU_QA_full.args.hf_args.name={train_split}",
            f"trainer.args.per_device_train_batch_size={batch_size}",
            f"trainer.args.gradient_accumulation_steps={grad_accum}",
            f"trainer.args.gradient_checkpointing={'true' if grad_ckpt else 'false'}",
            f"trainer.args.learning_rate={lr}",
            f"trainer.args.warmup_epochs={warmup_epochs}",
            f"trainer.args.weight_decay={wd}",
            f"trainer.args.num_train_epochs={epochs}",
            "trainer.args.do_eval=true",
            "trainer.args.eval_on_start=false",
            "trainer.args.eval_strategy=epoch",
            "trainer.args.save_strategy=no",
            f"forget_split={forget_split}",
            f"holdout_split={holdout_split}",
            f"retain_logs_path={retain_logs}",
        ]

        subprocess.run(cmd, check=True, cwd=WORKDIR, env=runtime_env())

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
    epochs: int = 15,
    batch_size: int = 8,
    grad_accum: int = 4,
    grad_ckpt: bool = True,
    lr: float = 2e-5,
    warmup_epochs: float = 0.2,
    weight_decays: str = "0.0 0.1",
):
    run_retain95_light_eval_sweep.spawn(
        setup_eval_data=setup_eval_data,
        model=model,
        train_split=train_split,
        forget_split=forget_split,
        holdout_split=holdout_split,
        retain_split=retain_split,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        grad_ckpt=grad_ckpt,
        lr=lr,
        warmup_epochs=warmup_epochs,
        weight_decays=weight_decays,
    )
