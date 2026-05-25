# Copilot Project Instructions: Modal Setup

These notes describe how this repository is wired to Modal so future sessions can quickly run or modify training jobs.

## Instruction Maintenance Rule

- If a conversation uncovers build/setup details that would help future training runs, add those details to this file in the relevant section before ending the task.
- Keep updates concise and operational (what changed, where it lives, and how to run/verify it).

## Local Environment

- Local CLI commands should run in the CUDA-enabled conda environment: `unlearning`.
- Preferred command prefix for local ops: `conda run -n unlearning ...`

## Source Of Truth Files

- Shared Modal setup: scripts/modal_project_setup.py
- Current TOFU run entrypoint: scripts/modal_tofu_finetune_llama32_1b.py
- Current training shell script: scripts/tofu_finetune.sh
- Parameter-sweep entrypoint: scripts/modal_tofu_finetune_sweep_llama32_1b.py
- Parameter-sweep shell script: scripts/tofu_finetune_sweep.sh

Use the shared setup module for new Modal jobs instead of duplicating image/volume definitions.

## Modal Resource Map

### App

- App name: open-unlearning-tofu-finetune-llama32-1b
- Launch mode: detached run from local entrypoint
- Pattern in entrypoint: spawn the long-running function so local CLI can return

### Image Build

Defined in scripts/modal_project_setup.py via build_project_image().

- Base image: nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 with Python 3.11
- Installs:
  - requirements from requirements.txt
  - lm-eval==0.4.11
  - wheel
  - flash-attn==2.6.3 using --no-build-isolation

Image build order is intentionally dependency-first so heavy dependency layers are reused across code changes.
Project source is copied after dependency installation.

Important: image build logs are long and can look like runtime logs.

### Volumes

Defined in scripts/modal_project_setup.py:

- HF cache volume name: open-unlearning-hf-cache
- Results volume name: open-unlearning-results

Container mounts used by jobs:

- /root/.cache/huggingface -> open-unlearning-hf-cache
- /workspace/saves -> open-unlearning-results

Runtime code mount used by jobs:

- Local repository is copied into /workspace in the final image layer.

Expected artifact roots inside results volume:

- /finetune
- /eval

### Runtime Environment

Defined by runtime_env() in scripts/modal_project_setup.py.

Key env vars:

- HF_HOME=/root/.cache/huggingface
- TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
- CUDA_HOME=/usr/local/cuda
- PATH prepends /usr/local/cuda/bin
- LD_LIBRARY_PATH prepends /usr/local/cuda/lib64
- TOKENIZERS_PARALLELISM=false

### Secret

- Modal secret: huggingface
- Required key: HF_TOKEN

## Current Known-Stable Run Shape

In scripts/modal_tofu_finetune_llama32_1b.py and scripts/tofu_finetune.sh:

- GPU: L40S:1
- Training launch: single GPU direct python invocation (not accelerate/deepspeed launcher)
- Model target: Llama-3.2-1B-Instruct
- Sanity scope: full TOFU finetune + one eval split first

This setup was chosen for stability under observed runtime/kernel constraints.

## Operations Cheat Sheet

Run from repository root.

### Launch

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_llama32_1b.py

### Launch fast sweep (no in-train eval, post-train eval only)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_sweep_llama32_1b.py

Optional overrides for sweep runner:

- sweep_epochs=5 (or 10)
- sweep_lr_values='1e-5 1.5e-5 2e-5'
- sweep_warmup_values='0.2 0.5'
- sweep_batch_size=8
- sweep_grad_accum=4
- sweep_post_eval=true

### List apps

conda run -n unlearning modal app list --json

### Stream logs

conda run -n unlearning modal app logs <APP_ID>

### Stop app

conda run -n unlearning modal app stop -y <APP_ID>

### Check finetune artifacts on volume

conda run -n unlearning modal volume ls open-unlearning-results /finetune

### Check eval artifacts on volume

conda run -n unlearning modal volume ls open-unlearning-results /eval

## How To Switch To A New Training Task

1. Keep scripts/modal_project_setup.py as shared infra.
2. Create a new Modal runner script in scripts/ that imports from modal_project_setup.
3. Reuse existing volume mounts and runtime_env().
4. Point subprocess call to a task-specific shell script (or python command).
5. Start with single-GPU sanity run before scaling up distributed settings.
6. Verify artifacts in open-unlearning-results volume before launching larger runs.
7. If changing dependencies, update build_project_image() once so all runners inherit changes.

## Troubleshooting Notes

- If app appears idle at first, confirm whether image build is still in progress from logs.
- Distinguish build completion from task completion by checking app Tasks in modal app list.
- If you see repeated dependency installs, verify dependency lines in build_project_image() are unchanged and that only mounted code changed.
- If runtime cannot import shared helper module, ensure runner can resolve scripts/ imports in container.
- If using flash-attn, keep CUDA + torch versions aligned with current shared image setup.
- TOFU eval can hit a bfloat16 NumPy conversion issue; evaluation code now casts bf16 losses/probabilities to float32 before converting to NumPy in src/evals/metrics/utils.py.
- Modal sweep containers should not rely on jq being installed; parse TOFU_SUMMARY.json with Python in shell scripts.
- If sweep parameters vary batch size or gradient accumulation, include them in task/output naming to avoid overwriting finetune/eval artifacts across runs.
