from __future__ import annotations

import sys
import subprocess

import modal

sys.path.insert(0, "/workspace/scripts")

from modal_project_setup import (
    WORKDIR,
    build_project_image,
    hf_cache,
    results,
    runtime_env,
)

APP_NAME = "open-unlearning-tofu-finetune-llama32-1b"

app = modal.App(APP_NAME)

image = build_project_image()


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=2 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def setup_eval_data() -> None:
    subprocess.run(["python", "setup_data.py", "--eval"], check=True, cwd=WORKDIR, env=runtime_env())
    results.commit()
    hf_cache.commit()


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
def run_tofu_finetune(setup_eval_data: bool = True) -> None:
    if setup_eval_data:
        subprocess.run(["python", "setup_data.py", "--eval"], check=True, cwd=WORKDIR, env=runtime_env())

    subprocess.run(["bash", "scripts/tofu_finetune.sh"], check=True, cwd=WORKDIR, env=runtime_env())

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(setup_eval_data: bool = True):
    run_tofu_finetune.spawn(setup_eval_data=setup_eval_data)
