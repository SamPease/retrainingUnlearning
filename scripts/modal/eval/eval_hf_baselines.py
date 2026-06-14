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

APP_NAME = "open-unlearning-tofu-eval-hf-baselines-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

RETAIN90_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"

# (hf_model_id, output_dir)
HF_MODELS = [
    (
        "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90_hf",
    ),
    (
        "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        "saves/eval/tofu_Llama-3.2-1B-Instruct_full_hf",
    ),
    (
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha1_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_hf",
    ),
    (
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_hf",
    ),
    (
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_lr1e-05_beta0.1_alpha1_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_IdkDPO_hf",
    ),
    (
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff1_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_scoeff1_hf",
    ),
    (
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer5_scoeff100_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_scoeff100_hf",
    ),
    (
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_lr1e-05_b4.5_a1_d0_g0.125_ep10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_SimNPO_hf",
    ),
]


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_eval(model_id: str, output_dir: str) -> None:
    subprocess.run(["python", "setup_data.py", "--eval"], check=True, cwd=WORKDIR, env=runtime_env())

    subprocess.run(
        [
            "python",
            "src/eval.py",
            "experiment=eval/tofu/default.yaml",
            "forget_split=forget10",
            "holdout_split=holdout10",
            "model=Llama-3.2-1B-Instruct",
            f"task_name=tofu_Llama-3.2-1B-Instruct_hf_baselines",
            f"model.model_args.pretrained_model_name_or_path={model_id}",
            f"paths.output_dir={output_dir}",
            f"retain_logs_path={RETAIN90_LOGS}",
        ],
        check=True,
        cwd=WORKDIR,
        env=runtime_env(),
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main() -> None:
    for model_id, output_dir in HF_MODELS:
        run_eval.spawn(model_id=model_id, output_dir=output_dir)
