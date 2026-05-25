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

APP_NAME = "open-unlearning-tofu-eval-checkpoints-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=8 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_eval_for_model(model_path: str, output_dir: str) -> None:
    subprocess.run(["python", "setup_data.py", "--eval"], check=True, cwd=WORKDIR, env=runtime_env())

    subprocess.run(
        [
            "python",
            "src/eval.py",
            "experiment=eval/tofu/default.yaml",
            "forget_split=forget10",
            "holdout_split=holdout10",
            "model=Llama-3.2-1B-Instruct",
            "task_name=tofu_Llama-3.2-1B-Instruct_ckpt_eval",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"paths.output_dir={output_dir}",
            "retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json",
        ],
        check=True,
        cwd=WORKDIR,
        env=runtime_env(),
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main() -> None:
    checkpoints = [125, 250, 375, 500, 625]
    for ckpt in checkpoints:
        run_eval_for_model.remote(
            model_path=f"saves/finetune/tofu_Llama-3.2-1B-Instruct_full/checkpoint-{ckpt}",
            output_dir=f"saves/eval/tofu_Llama-3.2-1B-Instruct_full_ckpt_ref/evals_ckpt_{ckpt}",
        )
