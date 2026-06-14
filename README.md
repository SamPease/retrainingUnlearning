# Retraining as a Probe for LLM Unlearning

A research project studying whether targeted retraining on a subset of "forgotten" examples can recover knowledge that unlearning algorithms removed — and what that recovery rate reveals about the quality of the unlearning.

**Built on top of [OpenUnlearning](https://github.com/locuslab/open-unlearning)** by Dorna et al. All credit for the underlying evaluation framework, benchmark implementations (TOFU, MUSE, WMDP), and unlearning method implementations goes to the original authors. This repo layers a recovery-training experimental pipeline on top of that framework.

> **Project write-up:** [Free Recovery: Does LLM Unlearning Actually Remove Knowledge?](https://sampease.github.io/project-writeups/retraining-unlearning/)

---

## Research Overview

Standard unlearning benchmarks measure whether a model has *forgotten* some set of examples immediately after unlearning. But they don't ask a follow-up question: **how much does it cost to re-learn what was forgotten?**

This project answers that by taking models that have been unlearned on TOFU (a fictitious biography dataset) and finetuning them on small subsets of the forget set. The key metrics are:

- **Transfer rate**: How much performance recovers per training example
- **Free-recovery**: Does training on a subset of forget examples recover knowledge of *other* forget examples that were never shown?
- **Utility preservation**: Does recovery training hurt the model's general capabilities?

The central finding is that for all tested unlearning methods, knowledge can be rapidly recovered with very few examples, and recovery on a small subset generalizes substantially to the rest of the forget set — suggesting the "forgotten" information remains latent in model weights.

---

## Setup

### Prerequisites

- Conda
- CUDA-capable GPU (for local runs)
- [Modal](https://modal.com) account (for cloud runs — optional but used for all experiments)
- HuggingFace account with access to Llama models

### 1. Create the conda environment

```bash
conda create -n unlearning python=3.11
conda activate unlearning
pip install ".[lm-eval]"
pip install --no-build-isolation flash-attn==2.6.3
```

### 2. Download baseline evaluation logs

```bash
python setup_data.py --eval
```

This populates `saves/eval/` with pre-computed evaluation results for the OpenUnlearning baseline models, which are needed as reference logs for TOFU metric computation.

### 3. Set up Modal (for cloud training)

```bash
pip install modal
modal setup          # authenticate via browser
```

Create a Modal secret named `huggingface` with your HuggingFace token:

```bash
modal secret create huggingface HF_TOKEN=hf_...
```

The Modal setup is defined in [scripts/modal_project_setup.py](scripts/modal_project_setup.py). It uses:
- GPU: L40S (single GPU for most jobs, 2× for DeepSpeed runs)
- Volumes: `open-unlearning-hf-cache` (model weights), `open-unlearning-results` (outputs)
- Image: CUDA 12.1 + Python 3.11 + flash-attn 2.6.3

---

## Repository Structure

```
src/                    # Core framework (from OpenUnlearning)
│  ├── train.py         # Main training/unlearning entry point (Hydra-driven)
│  ├── eval.py          # Main evaluation entry point
│  ├── trainer/         # Unlearning method implementations
│  ├── data/            # Dataset loaders
│  └── evals/           # Evaluation metrics and benchmarks

configs/                # Hydra configuration files
│  ├── experiment/      # High-level experiment configs
│  ├── trainer/         # Per-method training configs
│  ├── eval/            # Evaluation suite configs
│  └── model/           # Model configs

scripts/                # Experiment scripts (this project's main additions)
│  ├── modal_project_setup.py          # Shared Modal infrastructure
│  ├── modal/                          # Modal cloud job entrypoints
│  │   ├── train/                      # Training jobs (finetune, recovery)
│  │   └── eval/                       # Evaluation jobs (scan, baselines)
│  ├── shell/                          # Shell scripts
│  │   ├── train/                      # Training workflows
│  │   └── launch/                     # Orchestration launchers
│  ├── analysis/                       # Post-run analysis
│  │   ├── download/                   # Pull results from Modal volume
│  │   └── report/                     # Generate charts and reports
│  └── utils/                          # Standalone utilities

experiments/            # Experiment outputs (reports only; data is gitignored)
│  └── reports/
│      └── running-log.md              # Per-run notes and results

docs/                   # OpenUnlearning framework documentation
community/              # Community method contributions (from OpenUnlearning)
```

---

## Running Experiments

All experiments use Llama-3.2-1B-Instruct on the TOFU benchmark.

### Framework baseline: train a TOFU finetune model locally

```bash
conda run -n unlearning bash scripts/shell/train/tofu_finetune.sh
```

### Run an unlearning method (framework baseline)

```bash
conda run -n unlearning python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  forget_split=forget10 retain_split=retain90 trainer=GradAscent
```

### Recovery experiment: retrain an unlearned model on a forget subset

Launch a recovery run from a pre-unlearned HuggingFace checkpoint:

```bash
conda run -n unlearning modal run --detach scripts/modal/train/recovery_hfbase.py \
  --model-name-or-path open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10 \
  --model-tag npo_forget10 \
  --train-split forget01 \
  --run-baseline-eval
```

Or launch all 18 recovery jobs (6 methods × 3 splits) at once:

```bash
bash scripts/shell/launch/launch_new_methods_recovery.sh
```

### Download results and generate charts

```bash
conda run -n unlearning python scripts/analysis/download/download_new_methods_recovery.py
conda run -n unlearning python scripts/analysis/report/generate_free_recovery_all_metrics.py
conda run -n unlearning python scripts/analysis/report/generate_taught_utility_charts.py
```

### Check Modal job status

```bash
conda run -n unlearning modal app list --json | python3 -c \
  "import json,sys; [print(a['App ID'], a['State'], a.get('Description','')) for a in json.load(sys.stdin)]"
```

---

## Acknowledgements

This project is built on [OpenUnlearning](https://github.com/locuslab/open-unlearning) (Dorna et al., 2025). The TOFU benchmark is from [Maini et al., 2024](https://arxiv.org/abs/2401.06121). Please cite both if you use this work:

```bibtex
@article{openunlearning2025,
  title={{OpenUnlearning}: Accelerating {LLM} Unlearning via Unified Benchmarking of Methods and Metrics},
  author={Dorna, Vineeth and Mekala, Anmol and Zhao, Wenlong and McCallum, Andrew and Lipton, Zachary C and Kolter, J Zico and Maini, Pratyush},
  journal={arXiv preprint arXiv:2506.12618},
  year={2025},
}
@inproceedings{maini2024tofu,
  title={{TOFU}: A Task of Fictitious Unlearning for {LLMs}},
  author={Maini, Pratyush and Feng, Zhili and Schwarzschild, Avi and Lipton, Zachary Chase and Kolter, J Zico},
  booktitle={First Conference on Language Modeling},
  year={2024},
}
```

---

## License

MIT. See [LICENSE](LICENSE).
