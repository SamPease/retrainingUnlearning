# Project Instructions: Modal Setup

These notes describe how this repository is wired to Modal so future sessions can quickly run or modify training jobs.

## Instruction Maintenance Rule

- If a conversation uncovers build/setup details that would help future training runs, add those details to this file in the relevant section before ending the task.
- Keep updates concise and operational (what changed, where it lives, and how to run/verify it).

## Running Log Maintenance Rule

- When experiment results are generated or compared, update experiments/reports/running-log.md in the same session.
- Always append new running-log entries at the bottom of experiments/reports/running-log.md; do not insert new entries at the top.
- Preserve all reported metric values exactly when copying from JSON summaries, ledgers, or terminal output.
- Add or refresh charts when helpful for trend interpretation, especially:
  - per-epoch trajectory charts for light-eval runs,
  - late-epoch zoom charts to show plateau behavior,
  - epoch-to-epoch delta charts for marginal-gain visibility,
  - sweep response charts (for example, learning-rate response slices).
- For each chart addition, include a short readout (1-4 bullets) describing the main takeaway and how it informs run selection.

## Local Environment

- Local CLI commands should run in the CUDA-enabled conda environment: `unlearning`.
- Preferred command prefix for local ops: `conda run -n unlearning ...`
- Do not create a separate Python virtual environment for this repo (no `venv`, `virtualenv`, `poetry env`, etc.); always use the existing `unlearning` conda environment.
- If Python packages need to be installed, install them into the `unlearning` conda environment.

## Source Of Truth Files

- Shared Modal setup: scripts/modal_project_setup.py
- Current TOFU run entrypoint: scripts/modal_tofu_finetune_llama32_1b.py
- Current training shell script: scripts/tofu_finetune.sh
- Lightweight per-epoch TOFU training script: scripts/tofu_finetune_light_eval.sh
- Lightweight per-epoch TOFU Modal entrypoint: scripts/modal_tofu_finetune_light_eval_llama32_1b.py
- Parameter-sweep entrypoint: scripts/modal_tofu_finetune_sweep_llama32_1b.py
- Parameter-sweep shell script: scripts/tofu_finetune_sweep.sh
- Lightweight TOFU eval config: configs/eval/tofu_light.yaml
- Minimal TOFU eval config (forget_quality/model_utility/forget_truth_ratio): configs/eval/tofu_minimal.yaml
- Lightweight TOFU experiment config: configs/experiment/finetune/tofu/light_eval.yaml
- TOFU epoch report generator: scripts/generate_tofu_epoch_report.py
- Retain95 light-eval sweep entrypoint: scripts/modal_tofu_finetune_retain95_light_eval_sweep_llama32_1b.py
- Retain95 repro 2-GPU entrypoint: scripts/modal_tofu_finetune_retain95_repro_2gpu_llama32_1b.py
- Retain95 repro 2-GPU shell workflow: scripts/tofu_finetune_retain95_repro_2gpu.sh
- Dual 1-GPU repro-style (full + retain95) entrypoint: scripts/modal_tofu_finetune_dual_repro_1gpu_min_eval_llama32_1b.py
- Dual 1-GPU repro-style shell workflow: scripts/tofu_finetune_dual_repro_1gpu_min_eval.sh
- TRL forget10 trainer script: scripts/train_tofu_forget10_trl_sft.py
- TRL forget10 local shell workflow: scripts/tofu_finetune_trl_forget10.sh
- TRL forget10 Modal entrypoint: scripts/modal_tofu_finetune_trl_forget10_retain90_llama32_1b.py
- Custom TOFU eval split builder (forget10-minus-forgetX, retain90 perturbed): scripts/build_tofu_custom_eval_splits.py
- Retain90 utility-only eval config: configs/eval/tofu_retain90_utility.yaml
- Retain90 utility aggregate metric config: configs/eval/tofu_metrics/retain90_utility.yaml
- TRL retain90->forget01/forget05 train+eval Modal entrypoint: scripts/modal_tofu_finetune_trl_retain90_forget01_forget05_recovery_llama32_1b.py
- TRL hfbase (NPO/RMU) forget01/forget05/forget10 train+eval Modal entrypoint: scripts/modal_tofu_finetune_trl_hfbase_forgetx_recovery_llama32_1b.py
- HFBase recovery comparison report generator: scripts/compare_hfbase_recovery_results.py
- New unlearning method checkpoint scan (AltPO/GradDiff/IdkDPO/IdkNLL/SimNPO/UNDIAL) Modal entrypoint: scripts/modal_tofu_eval_hf_new_methods_scan_llama32_1b.py
- New unlearning method scan launcher: scripts/launch_new_methods_scan.sh
- New unlearning method scan result downloader: scripts/download_new_methods_scan_results.py
- New unlearning method recovery run launcher (6 methods × 3 splits, supports seed arg): scripts/launch_new_methods_recovery_runs.sh
- New unlearning method recovery result downloader: scripts/download_new_methods_recovery_results.py
- Seed replication recovery run launcher (retain90/NPO/RMU★ × 3 splits, supports seed arg): scripts/launch_seed123_recovery_runs.sh
- Seed replication recovery result downloader: scripts/download_seed123_recovery_results.py
- Corrected ES/privleak re-eval launcher (all 9 methods × forget01/forget05, --skip-train): scripts/launch_fixed_es_privleak_reeval.sh
- Corrected ES/privleak result downloader: scripts/download_fixed_es_privleak_results.py

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

### Launch lightweight local finetune with per-epoch eval

conda run -n unlearning bash scripts/tofu_finetune_light_eval.sh

### Launch lightweight Modal finetune with per-epoch eval (detached)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_light_eval_llama32_1b.py

Optional overrides for light-eval runner:

- epochs=10
- lr=2e-5
- warmup_epochs=0.2
- weight_decay=0.0
- batch_size=8
- grad_accum=4

### Launch fast sweep (no in-train eval, post-train eval only)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_sweep_llama32_1b.py

Optional overrides for sweep runner:

- sweep_epochs=5 (or 10)
- sweep_lr_values='1e-5 1.5e-5 2e-5'
- sweep_warmup_values='0.2 0.5'
- sweep_batch_size=8
- sweep_grad_accum=4
- sweep_post_eval=true

### Launch retain95 repro trial on 2 GPUs (detached)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_retain95_repro_2gpu_llama32_1b.py

Notes for this trial:

- Uses Accelerate + DeepSpeed ZeRO-3 with 2 x L40S and original repro hyperparameters by default (lr=1e-5, batch=8/device, grad_accum=4, epochs=10, optim=paged_adamw_32bit).
- Trains retain95 only, skips in-train eval, then runs post-train minimal eval only (forget_quality/model_utility/forget_truth_ratio).
- Prints metric deltas against repro retain95 targets: forget_quality=1.0, model_utility=0.63, forget_truth_ratio=0.67.

### Launch dual 1-GPU repro-style trial (full + retain95, detached)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_dual_repro_1gpu_min_eval_llama32_1b.py

Notes for this trial:

- Launches two detached one-GPU jobs by default: one with `RUN_MODE=full`, one with `RUN_MODE=retain95`.
- Uses repro-style optimization params by default with the requested shape: lr=1e-5, batch=8, grad_accum=4, epochs=20, optim=paged_adamw_32bit.
- Uses minimal TOFU metrics only at each epoch (`forget_quality`, `model_utility`, `forget_truth_ratio`) via `eval=tofu_minimal` and `eval_strategy=epoch`.
- Requires retain reference logs to already exist at `saves/eval/tofu_<model>_retain99/TOFU_EVAL.json` and `saves/eval/tofu_<model>_retain95/TOFU_EVAL.json`.
- Optional launch override: pass `run_mode=full` or `run_mode=retain95` to launch only one of the two jobs.

### Launch TRL + LoRA forget10 finetune from retain90 (detached)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_trl_forget10_retain90_llama32_1b.py

Notes for this trial:

- Uses TRL `SFTTrainer` with LoRA adapters (no full-parameter finetune path) on TOFU `forget10` train split.
- Default base model is `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`.
- Default output path is `saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget10_lora`.
- Hyperparameters can be overridden via Modal entrypoint args (`epochs`, `lr`, `batch_size`, `grad_accum`, `max_seq_length`, etc).

### Launch TRL retain90->forget01 or retain90->forget05 with taught/free-recovery/retain90-utility evals (detached)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_trl_retain90_forget01_forget05_recovery_llama32_1b.py --train-split forget01

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_trl_retain90_forget01_forget05_recovery_llama32_1b.py --train-split forget05

Notes for this trial:

- Uses TRL `SFTTrainer` + LoRA starting from `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`.
- Uses selected params from the retain90->forget10 run: epochs=20, lr=2e-4, warmup=0.03, wd=0.0, batch=4, grad_accum=4, lora r/a=16/32.
- Builds custom eval JSONL files under `saves/eval/custom_splits/` before training:
  - `forget10_minus_forget01_perturbed.jsonl`
  - `forget10_minus_forget05_perturbed.jsonl`
  - `retain90_perturbed.jsonl`
- Runs 3 post-train eval suites for each trained checkpoint:
  - taught split (`forget01` or `forget05`),
  - free-recovery split (`forget10-minus-forgetX`),
  - retain90 utility-only suite.

### Launch TRL hfbase (NPO/RMU) forgetX matrix with baseline+post-train evals (detached)

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_trl_hfbase_forgetx_recovery_llama32_1b.py --model-name-or-path open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10 --model-tag npo_forget10 --train-split forget01 --run-baseline-eval

conda run -n unlearning modal run --detach scripts/modal_tofu_finetune_trl_hfbase_forgetx_recovery_llama32_1b.py --model-name-or-path open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer10_scoeff100_epoch10 --model-tag rmu_forget10 --train-split forget05 --run-baseline-eval

Notes for this trial:

- Supports `--train-split forget01|forget05|forget10`.
- Runs baseline eval against the source HF checkpoint when `--run-baseline-eval` is set.
- Runs free-recovery eval only for forget01/forget05; forget10 runs taught + utility suites.

### Scan new unlearning method checkpoints (AltPO, GradDiff, IdkDPO, IdkNLL, SimNPO, UNDIAL)

bash scripts/launch_new_methods_scan.sh

Notes: launches 6 parallel Modal jobs (one per method, 6 sequential evals each). Download results after completion:

conda run -n unlearning python scripts/download_new_methods_scan_results.py

### Launch recovery runs for all 6 new unlearning methods (detached)

bash scripts/launch_new_methods_recovery_runs.sh [SEED]

Notes: launches 18 Modal jobs (6 methods × 3 splits). With 10-GPU plan limit, some jobs will queue. Download results after all complete:

conda run -n unlearning python scripts/download_new_methods_recovery_results.py [SEED]

### Generate hfbase baseline-vs-tuned-vs-retain comparison markdown

conda run -n unlearning python scripts/compare_hfbase_recovery_results.py

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

### Generate markdown training report with epoch eval tables/charts

conda run -n unlearning python scripts/generate_tofu_epoch_report.py --output experiments/reports/training-runs.md

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
- TRL `SFTTrainer` in this setup requires `rich`; keep `rich` in `requirements.txt` to avoid `No module named 'rich'` import failures during Modal runs.
- TOFU eval can hit a bfloat16 NumPy conversion issue; evaluation code now casts bf16 losses/probabilities to float32 before converting to NumPy in src/evals/metrics/utils.py.
- Modal sweep containers should not rely on jq being installed; parse TOFU_SUMMARY.json with Python in shell scripts.
- If sweep parameters vary batch size or gradient accumulation, include them in task/output naming to avoid overwriting finetune/eval artifacts across runs.
- For per-epoch evaluation speed, use configs/eval/tofu_light.yaml (no generation/MIA/reference metrics) and generate historical run summaries with scripts/generate_tofu_epoch_report.py.
- For the retain95 2-GPU repro trial, avoid forcing `trainer.args.gradient_checkpointing` and `trainer.args.ddp_find_unused_parameters` via CLI unless debugging memory issues; keep finetune defaults to stay closer to repro behavior.
- If the 2-GPU repro trial stalls at `0/590` after DeepSpeed init, keep hyperparameters unchanged and try `attn_implementation=sdpa` on relaunch to avoid Flash Attention startup issues.
- If 2-GPU DeepSpeed still stalls at `0/590`, keep repro hyperparameters and enable NCCL-safe transport flags (`NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NCCL_SHM_DISABLE=1`) for the trial run.
- For the dual 1-GPU repro-style run, missing retain reference logs will fail fast before training; generate/copy `tofu_<model>_retain99` and `tofu_<model>_retain95` TOFU_EVAL logs first.
- For detached local-entrypoint fan-out jobs, Modal may keep only the last triggered function alive after client disconnect; always verify expected output folders/files on `open-unlearning-results` (or run non-detached when complete fan-out reliability is required).
- **Modal GPU concurrency limit**: the current plan allows at most 10 GPUs running in parallel. Jobs beyond 10 are queued and will start as earlier jobs finish — they do not fail, but total wall-clock time increases. When launching more than 10 single-GPU jobs at once (e.g. 18-job fan-outs), expect some jobs to start delayed rather than simultaneously. Check `modal app list` to monitor which apps are still running.
- On macOS the default shell is bash 3.x which does not support `declare -A` associative arrays; use parallel indexed arrays (`TAGS=(...) MODEL_IDS=(...)` iterated with `for i in "${!TAGS[@]}"`) in launch scripts instead.
- `modal app list --json` returns title-cased keys (`App ID`, `State`, `Description`, ...) rather than snake_case; parse those exact keys in status scripts.
- For Hydra eval overrides that introduce new `hf_args` keys (for example `data_files` for JSON datasets), prefix those keys with `+` (for example `+...hf_args.data_files=...`) to avoid struct-key errors.
- For free-recovery custom split overrides, edit nested datasets under `eval.tofu.metrics.forget_truth_ratio.pre_compute.{forget_Q_A_PARA_Prob,forget_Q_A_PERT_Prob}` (not top-level metric keys), otherwise overrides fail with missing-key errors.
- **ES/privleak eval scope bug**: both metrics default to the full `forget_split` HuggingFace dataset, not the custom JSONL. Free-recovery evals must also override `eval.tofu.metrics.extraction_strength.datasets.TOFU_QA_forget.args.hf_args.{path,data_files,split}` and `eval.tofu.metrics.privleak.pre_compute.mia_min_k.datasets.TOFU_QA_forget.args.hf_args.{path,data_files,split}` to use the custom JSONL; otherwise taught-5% values are contaminated by the trained examples (forget05 = 50% of forget10). Both recovery scripts now include these overrides. Re-run old conditions with `--skip-train` via `scripts/launch_fixed_es_privleak_reeval.sh`.
- **TOFU_EVAL.json caching**: `src/eval.py` loads any existing `TOFU_EVAL.json` and skips already-computed metrics — so re-running with new Hydra overrides won't re-evaluate metrics unless the cache is cleared first. Both recovery scripts now delete `TOFU_EVAL.json` from the free-recovery eval directory before the corrected eval runs. Always clear the cache when changing eval dataset overrides for an already-evaluated directory.
- **In-script TOFU_EVAL.json deletion can be unreliable with Modal volumes** (the file exists on the volume but `os.path.exists()` may not see it due to mount timing). When a `--skip-train` re-eval leaves ES/privleak unchanged from a prior run, delete the file directly from the volume first: `conda run -n unlearning modal volume rm open-unlearning-results /eval/<slug>/evals_forget10_minus_<split>/TOFU_EVAL.json`, then re-launch.
- **Old chart scripts** (`scripts/generate_recovery_charts.py`, `scripts/generate_recovery_charts_v2.py`) and their data sources (`tmp/hfbase_recovery_eval_summaries/`, `tmp/new_recovery_eval_summaries/`, `tmp/rmu_new_recovery_eval_summaries/`) are **superseded and contain pre-correction ES/privleak values**. Do not use them to generate report charts. Current chart scripts are `generate_free_recovery_chart.py`, `generate_taught_utility_charts.py`, and `generate_free_recovery_all_metrics_charts.py` — all read from `saves/eval/` which has corrected values.
