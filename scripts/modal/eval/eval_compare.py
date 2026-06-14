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

APP_NAME = "open-unlearning-tofu-eval-compare-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=6 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_eval(
    model_path: str,
    output_dir: str,
    forget_split: str = "forget10",
    holdout_split: str = "holdout10",
    retain_split: str = "retain99",
) -> None:
    # Ensure eval datasets are present for a clean standalone run.
    subprocess.run(["python", "setup_data.py", "--eval"], check=True, cwd=WORKDIR, env=runtime_env())

    subprocess.run(
        [
            "python",
            "src/eval.py",
            "experiment=eval/tofu/default.yaml",
            f"forget_split={forget_split}",
            f"holdout_split={holdout_split}",
            "model=Llama-3.2-1B-Instruct",
            "task_name=tofu_Llama-3.2-1B-Instruct_compare",
            f"model.model_args.pretrained_model_name_or_path={model_path}",
            f"paths.output_dir={output_dir}",
            f"retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_{retain_split}/TOFU_EVAL.json",
        ],
        check=True,
        cwd=WORKDIR,
        env=runtime_env(),
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(forget_split: str = "forget10", holdout_split: str = "holdout10") -> None:
    # HF reference model evaluation.
    run_eval.remote(
        model_path="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        output_dir="saves/eval/tofu_Llama-3.2-1B-Instruct_full_hf_ref/evals_forget10",
        forget_split=forget_split,
        holdout_split=holdout_split,
    )

    # Modal finetuned checkpoint evaluation.
    run_eval.remote(
        model_path="saves/finetune/tofu_Llama-3.2-1B-Instruct_full",
        output_dir="saves/eval/tofu_Llama-3.2-1B-Instruct_full_modal_ref/evals_forget10",
        forget_split=forget_split,
        holdout_split=holdout_split,
    )
