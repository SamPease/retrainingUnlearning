# TOFU Training Runs

Generated at: 2026-05-25T17:44:47.057383+00:00

This report is generated from checkpoint-level TOFU summary files:
- checkpoint-*/evals/TOFU_SUMMARY.json

Run lightweight per-epoch eval training:
```bash
python src/train.py experiment=finetune/tofu/light_eval.yaml task_name=<RUN_NAME>
```

Regenerate this report:
```bash
python scripts/generate_tofu_epoch_report.py --output experiments/reports/training-runs.md
```

## Run: tofu_Llama-3.2-1B-Instruct_full_light_eval_lr2e5_wu02_e15

- Run directory: saves/finetune/tofu_Llama-3.2-1B-Instruct_full_light_eval_lr2e5_wu02_e15
- Checkpoint evals found: 0

No checkpoint-level TOFU eval summaries found.

