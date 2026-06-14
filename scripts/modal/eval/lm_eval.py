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

APP_NAME = "open-unlearning-lm-eval-hf-models-llama32-1b"

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
def run_lm_eval(model_id: str, output_dir: str, batch_size: int = 16) -> None:
    tasks_override = "[hellaswag,arc_challenge,truthfulqa_mc2]"
    subprocess.run(
        [
            "python",
            "src/eval.py",
            "--config-name=eval.yaml",
            "eval=lm_eval",
            "model=Llama-3.2-1B-Instruct",
            "task_name=lm_eval_llama32_1b",
            f"model.model_args.pretrained_model_name_or_path={model_id}",
            f"paths.output_dir={output_dir}",
            f"eval.lm_eval.tasks={tasks_override}",
            f"eval.lm_eval.simple_evaluate_args.batch_size={batch_size}",
            "eval.lm_eval.simple_evaluate_args.apply_chat_template=true",
        ],
        check=True,
        cwd=WORKDIR,
        env=runtime_env(),
    )

    results.commit()
    hf_cache.commit()
