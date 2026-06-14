# Project Instructions

## How to behave in this repo

**Do not launch Modal jobs unless explicitly asked.** Each job costs GPU time. Ask before running anything on Modal.

**Do not modify `src/`** unless the task is specifically about the framework code. New experiments use Hydra configs and scripts; they don't touch the framework.

**Do not create new scripts without checking whether an existing one can be extended.** The scripts/ directory is organized into a modular framework (see layout below). Before writing a new file, read the most similar existing script in that category. Many tasks are handled by passing new arguments to an existing entrypoint, not creating another file.

**The project write-up lives externally** at https://sampease.github.io/project-writeups/retraining-unlearning/. Do not create or edit report files like retraining-unlearning.md or training-runs.md — they are gitignored because they are generated or externally hosted.

**The canonical experiment record is `experiments/reports/running-log.md`.** When results come in, append an entry there (never at the top). If asked to analyze results, read the running-log for context on what's already been tried.

---

## Instruction Maintenance Rule

- If a conversation uncovers build/setup details that help future runs, add them to the relevant section of this file before ending the session.
- Keep additions concise and operational.

## Running Log Maintenance Rule

- When experiment results are generated or compared, update experiments/reports/running-log.md in the same session.
- Always append new entries at the bottom; never insert at the top.
- Preserve all reported metric values exactly when copying from JSON summaries, ledgers, or terminal output.
- Add or refresh charts when helpful for trend interpretation:
  - per-epoch trajectory charts for light-eval runs
  - late-epoch zoom charts to show plateau behavior
  - sweep response charts (learning-rate response slices)
- For each chart, include a short readout (1–4 bullets) describing the main takeaway.

---

## Local Environment

- All local CLI commands run in the conda environment `unlearning`.
- Preferred prefix: `conda run -n unlearning ...`
- Never create a venv, virtualenv, or poetry env for this repo.
- Install packages into the `unlearning` conda environment.

---

## Scripts Framework

Scripts are organized by function. **Before writing a new script, read what exists in the relevant directory first.**

```
scripts/
├── modal_project_setup.py          # Shared Modal infra — imported by all Modal scripts
│                                   # Do NOT duplicate image/volume definitions elsewhere
├── modal/
│   ├── train/                      # Modal training entrypoints
│   │   ├── finetune_full.py        # Basic TOFU finetune on 1× L40S
│   │   ├── finetune_light_eval.py  # Per-epoch light eval finetune
│   │   ├── finetune_sweep.py       # Hyperparameter sweep
│   │   ├── finetune_retain95_sweep.py
│   │   ├── finetune_retain95_2gpu.py  # 2× GPU + DeepSpeed ZeRO-3
│   │   ├── finetune_dual_repro.py
│   │   ├── recovery_trl_forget10.py   # TRL SFT on forget10 from retain90
│   │   ├── recovery_retain90.py       # Recovery: retain90 → forget01/05
│   │   └── recovery_hfbase.py         # Recovery: any HF unlearned ckpt → forgetX
│   └── eval/                       # Modal eval entrypoints
│       ├── scan_new_methods.py     # Scan HF checkpoint candidates per method
│       ├── scan_rmu.py
│       ├── eval_checkpoints.py
│       ├── eval_compare.py
│       ├── eval_hf_baselines.py
│       ├── eval_hf_variants2.py
│       ├── eval_free_recovery.py
│       ├── eval_free_recovery_baselines.py
│       ├── eval_corrected_es.py
│       ├── data_sanity.py
│       └── lm_eval.py
├── shell/
│   ├── train/                      # Shell workflows called by Modal entrypoints
│   │   ├── tofu_finetune.sh
│   │   ├── tofu_finetune_light_eval.sh
│   │   ├── tofu_finetune_sweep.sh
│   │   ├── tofu_finetune_retain95_2gpu.sh
│   │   ├── tofu_finetune_dual_repro.sh
│   │   ├── tofu_finetune_trl_forget10.sh
│   │   ├── tofu_unlearn.sh
│   │   └── muse_unlearn.sh
│   └── launch/                     # Orchestrators that fire multiple Modal jobs
│       ├── launch_new_methods_scan.sh
│       ├── launch_new_methods_recovery.sh
│       ├── launch_seed123_recovery.sh
│       └── launch_fixed_es_privleak_reeval.sh
├── analysis/
│   ├── download/                   # Pull results from Modal volume → local saves/
│   │   ├── download_new_methods_scan.py
│   │   ├── download_new_methods_recovery.py
│   │   ├── download_seed123_recovery.py
│   │   ├── download_full_free_recovery.py
│   │   ├── download_free_recovery_baselines.py
│   │   ├── download_corrected_es.py
│   │   ├── download_fixed_es_privleak.py
│   │   ├── download_rmu_recovery.py
│   │   └── download_seed_robustness.py
│   └── report/                     # Chart generation and markdown reports
│       ├── generate_epoch_report.py
│       ├── generate_free_recovery_chart.py
│       ├── generate_free_recovery_all_metrics.py
│       ├── generate_taught_utility_charts.py
│       ├── generate_transfer_rate_chart.py
│       ├── analyze_sweep_results.py
│       ├── analyze_eval_compare.py
│       ├── check_data_debug.py
│       └── plot_dual_repro.py
└── utils/                          # Standalone utility scripts
    ├── build_custom_eval_splits.py # Build forget10-minus-forgetX and retain90 eval splits
    ├── compare_hfbase_recovery.py  # Compare NPO/RMU recovery vs retain90
    ├── verify_nested_splits.py     # Validate nested TOFU split structure
    └── train_trl_sft.py           # Standalone TRL SFT trainer (no Hydra)
```

### Rules for adding scripts

**Extending existing scripts** (preferred): Most new experiments can be handled by adding a parameter to `recovery_hfbase.py` or `finetune_light_eval.py` rather than creating a new file. Prefer this.

**New Modal entrypoint**: Only when the job structure is genuinely different from everything in `modal/train/` or `modal/eval/`. Put it in the right subdirectory. Copy the boilerplate from the nearest existing file — same `sys.path.insert`, same volume mounts, same `runtime_env()`, same spawn pattern.

**New shell script**: Only when training logic is different enough to warrant it. Goes in `shell/train/`. The Modal entrypoint should call it via `subprocess.run(["bash", "scripts/shell/train/your_script.sh"], cwd=WORKDIR)`.

**New download script**: Only after adding a new Modal eval/train job that writes to a new location on the volume. Goes in `analysis/download/`. Follow the pattern of existing download scripts.

**New chart script**: Goes in `analysis/report/`. Charts output to `experiments/reports/` (or a subfolder). Do not hardcode saves/ paths that aren't gitignored.

---

## Experiment Workflow

The typical end-to-end pattern for any new training or eval experiment:

1. **Launch**: `conda run -n unlearning modal run --detach scripts/modal/train/<entrypoint>.py [args]`
2. **Monitor**: `conda run -n unlearning modal app list --json` or stream logs with `modal app logs <APP_ID>`
3. **Download**: `conda run -n unlearning python scripts/analysis/download/<downloader>.py`
4. **Charts**: `conda run -n unlearning python scripts/analysis/report/<chart_script>.py`
5. **Log**: Append a section to `experiments/reports/running-log.md` with metrics, charts, and readout

Do not skip step 5. Do not modify running-log entries that already exist — only append.

---

## Source Of Truth Files

- Shared Modal setup: scripts/modal_project_setup.py
- Basic TOFU finetune Modal entrypoint: scripts/modal/train/finetune_full.py
- Basic TOFU finetune shell script: scripts/shell/train/tofu_finetune.sh
- Lightweight per-epoch TOFU finetune shell script: scripts/shell/train/tofu_finetune_light_eval.sh
- Lightweight per-epoch TOFU Modal entrypoint: scripts/modal/train/finetune_light_eval.py
- Parameter-sweep Modal entrypoint: scripts/modal/train/finetune_sweep.py
- Parameter-sweep shell script: scripts/shell/train/tofu_finetune_sweep.sh
- Lightweight TOFU eval config: configs/eval/tofu_light.yaml
- Minimal TOFU eval config (forget_quality/model_utility/forget_truth_ratio): configs/eval/tofu_minimal.yaml
- Lightweight TOFU experiment config: configs/experiment/finetune/tofu/light_eval.yaml
- TOFU epoch report generator: scripts/analysis/report/generate_epoch_report.py
- Retain95 light-eval sweep Modal entrypoint: scripts/modal/train/finetune_retain95_sweep.py
- Retain95 repro 2-GPU Modal entrypoint: scripts/modal/train/finetune_retain95_2gpu.py
- Retain95 repro 2-GPU shell workflow: scripts/shell/train/tofu_finetune_retain95_2gpu.sh
- Dual 1-GPU repro-style Modal entrypoint: scripts/modal/train/finetune_dual_repro.py
- Dual 1-GPU repro-style shell workflow: scripts/shell/train/tofu_finetune_dual_repro.sh
- TRL SFT trainer script: scripts/utils/train_trl_sft.py
- TRL forget10 local shell workflow: scripts/shell/train/tofu_finetune_trl_forget10.sh
- TRL forget10 Modal entrypoint: scripts/modal/train/recovery_trl_forget10.py
- Custom TOFU eval split builder: scripts/utils/build_custom_eval_splits.py
- Retain90 utility-only eval config: configs/eval/tofu_retain90_utility.yaml
- Retain90 utility aggregate metric config: configs/eval/tofu_metrics/retain90_utility.yaml
- TRL retain90->forget01/forget05 train+eval Modal entrypoint: scripts/modal/train/recovery_retain90.py
- TRL hfbase (NPO/RMU) forgetX train+eval Modal entrypoint: scripts/modal/train/recovery_hfbase.py
- HFBase recovery comparison report generator: scripts/utils/compare_hfbase_recovery.py
- New unlearning method checkpoint scan Modal entrypoint: scripts/modal/eval/scan_new_methods.py
- New unlearning method scan launcher: scripts/shell/launch/launch_new_methods_scan.sh
- New unlearning method scan result downloader: scripts/analysis/download/download_new_methods_scan.py
- New unlearning method recovery run launcher (6 methods × 3 splits): scripts/shell/launch/launch_new_methods_recovery.sh
- New unlearning method recovery result downloader: scripts/analysis/download/download_new_methods_recovery.py
- Seed replication recovery run launcher: scripts/shell/launch/launch_seed123_recovery.sh
- Seed replication recovery result downloader: scripts/analysis/download/download_seed123_recovery.py
- Corrected ES/privleak re-eval launcher: scripts/shell/launch/launch_fixed_es_privleak_reeval.sh
- Corrected ES/privleak result downloader: scripts/analysis/download/download_fixed_es_privleak.py

---

## Modal Resource Map

### Image Build

Defined in scripts/modal_project_setup.py via `build_project_image()`.

- Base: nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04, Python 3.11
- Installs: requirements.txt, lm-eval==0.4.11, wheel, flash-attn==2.6.3 (--no-build-isolation)
- Dependency layers are built before source copy — changing only scripts does not rebuild the image.

### Volumes

- `open-unlearning-hf-cache` → /root/.cache/huggingface
- `open-unlearning-results` → /workspace/saves

Expected output roots in results volume: `/finetune`, `/eval`

### Runtime Environment

Key env vars set by `runtime_env()` in modal_project_setup.py:
- HF_HOME=/root/.cache/huggingface
- CUDA_HOME=/usr/local/cuda, LD_LIBRARY_PATH, PATH (CUDA prepended)
- TOKENIZERS_PARALLELISM=false

### Secret

- Name: `huggingface`, required key: `HF_TOKEN`

### GPU concurrency

Plan allows at most 10 GPUs in parallel. Jobs beyond 10 queue (they don't fail). When launching 18-job fan-outs, expect delayed starts. Check `modal app list` to monitor.

---

## Operations Cheat Sheet

Run from repository root.

### Basic TOFU finetune

```
conda run -n unlearning modal run --detach scripts/modal/train/finetune_full.py
```

### Lightweight per-epoch eval finetune (local)

```
conda run -n unlearning bash scripts/shell/train/tofu_finetune_light_eval.sh
```

### Lightweight per-epoch eval finetune (Modal)

```
conda run -n unlearning modal run --detach scripts/modal/train/finetune_light_eval.py
```

Overrides: `epochs=10 lr=2e-5 warmup_epochs=0.2 weight_decay=0.0 batch_size=8 grad_accum=4`

### Hyperparameter sweep

```
conda run -n unlearning modal run --detach scripts/modal/train/finetune_sweep.py
```

Overrides: `sweep_epochs=5 sweep_lr_values='1e-5 1.5e-5 2e-5' sweep_warmup_values='0.2 0.5' sweep_batch_size=8 sweep_grad_accum=4`

### Retain95 repro on 2 GPUs

```
conda run -n unlearning modal run --detach scripts/modal/train/finetune_retain95_2gpu.py
```

- Accelerate + DeepSpeed ZeRO-3, 2× L40S
- Trains retain95 only; post-train minimal eval
- Targets: forget_quality=1.0, model_utility=0.63, forget_truth_ratio=0.67

### Dual 1-GPU repro (full + retain95)

```
conda run -n unlearning modal run --detach scripts/modal/train/finetune_dual_repro.py
```

- Launches two detached jobs (RUN_MODE=full, RUN_MODE=retain95)
- Requires retain99 and retain95 reference logs in saves/eval/ already
- Override: `run_mode=full` or `run_mode=retain95` to run only one

### TRL + LoRA forget10 from retain90

```
conda run -n unlearning modal run --detach scripts/modal/train/recovery_trl_forget10.py
```

### Recovery: retain90 → forget01 or forget05

```
conda run -n unlearning modal run --detach scripts/modal/train/recovery_retain90.py --train-split forget01
conda run -n unlearning modal run --detach scripts/modal/train/recovery_retain90.py --train-split forget05
```

- Builds custom eval splits before training
- Runs 3 eval suites: taught / free-recovery / retain90-utility

### Recovery: HF unlearned checkpoint → forgetX

```
conda run -n unlearning modal run --detach scripts/modal/train/recovery_hfbase.py \
  --model-name-or-path <HF_MODEL_ID> --model-tag <TAG> --train-split forget01 --run-baseline-eval
```

- Supports forget01 / forget05 / forget10
- Baseline eval runs against the HF checkpoint when --run-baseline-eval is set
- Best NPO: open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10
- Best RMU: open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10

### Scan new unlearning method checkpoints

```
bash scripts/shell/launch/launch_new_methods_scan.sh
conda run -n unlearning python scripts/analysis/download/download_new_methods_scan.py
```

### Launch recovery runs for all 6 new methods

```
bash scripts/shell/launch/launch_new_methods_recovery.sh [SEED]
conda run -n unlearning python scripts/analysis/download/download_new_methods_recovery.py [SEED]
```

### Generate charts

```
conda run -n unlearning python scripts/analysis/report/generate_free_recovery_all_metrics.py
conda run -n unlearning python scripts/analysis/report/generate_taught_utility_charts.py
conda run -n unlearning python scripts/analysis/report/generate_transfer_rate_chart.py
conda run -n unlearning python scripts/analysis/report/generate_epoch_report.py --output experiments/reports/training-runs.md
```

### Modal management

```
conda run -n unlearning modal app list --json
conda run -n unlearning modal app logs <APP_ID>
conda run -n unlearning modal app stop -y <APP_ID>
conda run -n unlearning modal volume ls open-unlearning-results /eval
conda run -n unlearning modal volume ls open-unlearning-results /finetune
```

---

## Troubleshooting Notes

- **Image build vs task start**: image build logs are long and look like runtime logs. Check Tasks in `modal app list` to distinguish.
- **Repeated dependency installs**: dependency lines in build_project_image() must be unchanged; only source copy changes should be cheap.
- **Import error for modal_project_setup**: all Modal scripts do `sys.path.insert(0, "/workspace/scripts")` so they can `from modal_project_setup import ...`. If a new script can't import it, check this line is present and the path is correct.
- **flash-attn**: keep CUDA + torch versions aligned with the current image. Do not change the flash-attn version without also checking CUDA and torch.
- **TRL SFTTrainer requires `rich`**: keep `rich` in requirements.txt or Modal runs will fail with `No module named 'rich'`.
- **bfloat16 NumPy conversion**: TOFU eval casts bf16 to float32 before numpy in src/evals/metrics/utils.py. If eval crashes with a dtype error, check this path.
- **TOFU_EVAL.json caching** (critical): `src/eval.py` skips metrics that already have results in TOFU_EVAL.json. Re-running with different Hydra overrides will return stale results unless you clear the cache first. Always delete TOFU_EVAL.json before re-evaluating with different dataset overrides.
- **Modal volume cache deletion**: `os.path.exists()` inside a Modal function may not see files on the volume due to mount timing. Delete TOFU_EVAL.json directly from the volume before re-launching: `conda run -n unlearning modal volume rm open-unlearning-results /eval/<slug>/TOFU_EVAL.json`.
- **ES/privleak scope**: both metrics default to the full forget_split HuggingFace dataset, not the custom JSONL. Free-recovery evals must override all three dataset pointers (forget_Q_A_Prob, forget_Q_A_ROUGE, extraction_strength, privleak) to use the custom JSONL. Both recovery scripts already do this — do not remove those overrides.
- **Hydra `+` prefix**: when introducing a new hf_args key (e.g. data_files) that doesn't exist in the base config, prefix with `+` to avoid struct-key errors.
- **Free-recovery custom split overrides**: edit nested datasets under `eval.tofu.metrics.forget_truth_ratio.pre_compute.{forget_Q_A_PARA_Prob,forget_Q_A_PERT_Prob}`, not top-level metric keys.
- **2-GPU DeepSpeed**: if a run stalls at `0/590` after DeepSpeed init, try `attn_implementation=sdpa`. If still stalled, enable NCCL-safe flags: `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1`.
- **Dual 1-GPU repro**: missing retain reference logs fail fast before training. Generate or copy retain99 and retain95 TOFU_EVAL.json files first.
- **Fan-out kill problem**: for detached local-entrypoint fan-out jobs, Modal may kill all but the last spawned function after client disconnect. Verify output folders on the volume after completion, or use non-detached mode.
- **macOS bash 3.x**: no `declare -A` support. Use parallel indexed arrays (`TAGS=(...) MODEL_IDS=(...)`).
- **modal app list --json**: keys are title-cased (`App ID`, `State`, `Description`), not snake_case.
- **jq not available in Modal containers**: parse JSON with Python in shell scripts, not jq.
- **Sweep naming**: if sweep parameters vary batch size or grad_accum, include them in the task/output name to avoid overwriting artifacts across runs.
- **Superseded chart scripts**: `scripts/generate_recovery_charts.py` and `scripts/generate_recovery_charts_v2.py` were deleted. Use `scripts/analysis/report/generate_free_recovery_all_metrics.py` and related scripts. All chart scripts read from `saves/eval/` (corrected values). The `tmp/` directory is gitignored and contains pre-correction data — do not use it.
