"""
Scan 6 candidate checkpoints per new unlearning method (AltPO, GradDiff, IdkDPO,
IdkNLL, SimNPO, UNDIAL) to identify the best forget/utility trade-off for each.

Architecture: one long-running Modal function per method; candidates within a
method are evaluated sequentially to avoid the detach-fan-out kill problem.

Usage (one call per method, all can run concurrently):
  conda run -n unlearning modal run --detach scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py --method GradDiff
  conda run -n unlearning modal run --detach scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py --method AltPO
  conda run -n unlearning modal run --detach scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py --method IdkDPO
  conda run -n unlearning modal run --detach scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py --method IdkNLL
  conda run -n unlearning modal run --detach scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py --method SimNPO
  conda run -n unlearning modal run --detach scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py --method UNDIAL

Or use scripts/launch_new_methods_scan.sh to fire all 6 at once.
"""
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

APP_NAME = "open-unlearning-tofu-eval-hf-new-methods-scan-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

RETAIN90_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
BASE = "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10"
EVAL_BASE = "saves/eval/scan"

# ── candidate definitions ─────────────────────────────────────────────────────
# 6 per method; epoch=10 throughout; candidates span lr range and key strength
# parameters (beta/alpha for DPO-like methods, alpha for GradDiff/IdkNLL).
#
# Selection rationale:
#   - Cover 3 lr values (1e-5, 2e-5, 5e-5)
#   - Cover low/high values of the primary strength param
#   - For UNDIAL the lr range is 1e-5, 1e-4, 3e-4
#   - For SimNPO vary beta (3.5/4.5), gamma (0.125/0.25) and delta (0/1)

CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "AltPO": [
        ("altpo_l1e5_b01_a1",  f"{BASE}_AltPO_lr1e-05_beta0.1_alpha1_epoch10"),
        ("altpo_l2e5_b01_a1",  f"{BASE}_AltPO_lr2e-05_beta0.1_alpha1_epoch10"),
        ("altpo_l5e5_b01_a1",  f"{BASE}_AltPO_lr5e-05_beta0.1_alpha1_epoch10"),
        ("altpo_l1e5_b005_a2", f"{BASE}_AltPO_lr1e-05_beta0.05_alpha2_epoch10"),
        ("altpo_l2e5_b05_a1",  f"{BASE}_AltPO_lr2e-05_beta0.5_alpha1_epoch10"),
        ("altpo_l5e5_b05_a5",  f"{BASE}_AltPO_lr5e-05_beta0.5_alpha5_epoch10"),
    ],
    "GradDiff": [
        ("graddiff_l1e5_a1",  f"{BASE}_GradDiff_lr1e-05_alpha1_epoch10"),
        ("graddiff_l2e5_a1",  f"{BASE}_GradDiff_lr2e-05_alpha1_epoch10"),
        ("graddiff_l3e5_a5",  f"{BASE}_GradDiff_lr3e-05_alpha5_epoch10"),
        ("graddiff_l4e5_a5",  f"{BASE}_GradDiff_lr4e-05_alpha5_epoch10"),
        ("graddiff_l5e5_a2",  f"{BASE}_GradDiff_lr5e-05_alpha2_epoch10"),
        ("graddiff_l5e5_a10", f"{BASE}_GradDiff_lr5e-05_alpha10_epoch10"),
    ],
    "IdkDPO": [
        ("idkdpo_l1e5_b01_a1",  f"{BASE}_IdkDPO_lr1e-05_beta0.1_alpha1_epoch10"),
        ("idkdpo_l2e5_b01_a1",  f"{BASE}_IdkDPO_lr2e-05_beta0.1_alpha1_epoch10"),
        ("idkdpo_l5e5_b01_a1",  f"{BASE}_IdkDPO_lr5e-05_beta0.1_alpha1_epoch10"),
        ("idkdpo_l1e5_b005_a2", f"{BASE}_IdkDPO_lr1e-05_beta0.05_alpha2_epoch10"),
        ("idkdpo_l2e5_b05_a1",  f"{BASE}_IdkDPO_lr2e-05_beta0.5_alpha1_epoch10"),
        ("idkdpo_l5e5_b05_a5",  f"{BASE}_IdkDPO_lr5e-05_beta0.5_alpha5_epoch10"),
    ],
    "IdkNLL": [
        ("idknll_l1e5_a1",  f"{BASE}_IdkNLL_lr1e-05_alpha1_epoch10"),
        ("idknll_l2e5_a1",  f"{BASE}_IdkNLL_lr2e-05_alpha1_epoch10"),
        ("idknll_l3e5_a5",  f"{BASE}_IdkNLL_lr3e-05_alpha5_epoch10"),
        ("idknll_l4e5_a5",  f"{BASE}_IdkNLL_lr4e-05_alpha5_epoch10"),
        ("idknll_l5e5_a2",  f"{BASE}_IdkNLL_lr5e-05_alpha2_epoch10"),
        ("idknll_l5e5_a10", f"{BASE}_IdkNLL_lr5e-05_alpha10_epoch10"),
    ],
    "SimNPO": [
        ("simnpo_l1e5_b35_d0_g025",  f"{BASE}_SimNPO_lr1e-05_b3.5_a1_d0_g0.25_ep10"),
        ("simnpo_l2e5_b35_d0_g025",  f"{BASE}_SimNPO_lr2e-05_b3.5_a1_d0_g0.25_ep10"),
        ("simnpo_l5e5_b35_d0_g025",  f"{BASE}_SimNPO_lr5e-05_b3.5_a1_d0_g0.25_ep10"),
        ("simnpo_l1e5_b45_d1_g0125", f"{BASE}_SimNPO_lr1e-05_b4.5_a1_d1_g0.125_ep10"),
        ("simnpo_l2e5_b45_d1_g025",  f"{BASE}_SimNPO_lr2e-05_b4.5_a1_d1_g0.25_ep10"),
        ("simnpo_l5e5_b45_d1_g025",  f"{BASE}_SimNPO_lr5e-05_b4.5_a1_d1_g0.25_ep10"),
    ],
    "UNDIAL": [
        ("undial_l1e5_b10_a1",  f"{BASE}_UNDIAL_lr1e-05_beta10_alpha1_epoch10"),
        ("undial_l1e4_b10_a1",  f"{BASE}_UNDIAL_lr0.0001_beta10_alpha1_epoch10"),
        ("undial_l1e4_b30_a2",  f"{BASE}_UNDIAL_lr0.0001_beta30_alpha2_epoch10"),
        ("undial_l3e4_b3_a1",   f"{BASE}_UNDIAL_lr0.0003_beta3_alpha1_epoch10"),
        ("undial_l3e4_b10_a1",  f"{BASE}_UNDIAL_lr0.0003_beta10_alpha1_epoch10"),
        ("undial_l3e4_b30_a5",  f"{BASE}_UNDIAL_lr0.0003_beta30_alpha5_epoch10"),
    ],
}


def _eval_one(label: str, model_id: str, env: dict) -> None:
    output_dir = f"{EVAL_BASE}/{label}"
    print(f"[scan] evaluating {label} → {model_id}", flush=True)
    subprocess.run(
        [
            "python", "src/eval.py",
            "experiment=eval/tofu/default.yaml",
            "forget_split=forget10",
            "holdout_split=holdout10",
            "model=Llama-3.2-1B-Instruct",
            f"task_name=tofu_hf_scan_{label}",
            f"model.model_args.pretrained_model_name_or_path={model_id}",
            f"paths.output_dir={output_dir}",
            f"eval.tofu.retain_logs_path={RETAIN90_LOGS}",
        ],
        check=True,
        cwd=WORKDIR,
        env=env,
    )
    results.commit()
    hf_cache.commit()


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
def run_method_scan(method: str, run_only: str = "") -> None:
    candidates = CANDIDATES.get(method)
    if not candidates:
        raise ValueError(f"Unknown method {method!r}. Valid: {list(CANDIDATES)}")

    wanted = {x.strip() for x in run_only.split(",") if x.strip()}
    env = runtime_env()

    subprocess.run(["python", "setup_data.py", "--eval"], check=True, cwd=WORKDIR, env=env)

    for label, model_id in candidates:
        if wanted and label not in wanted:
            print(f"[scan] skipping {label} (not in --run-only)", flush=True)
            continue
        _eval_one(label, model_id, env)

    print(f"[scan] {method} complete.", flush=True)


@app.local_entrypoint()
def main(method: str = "GradDiff", run_only: str = "") -> None:
    run_method_scan.spawn(method=method, run_only=run_only)
