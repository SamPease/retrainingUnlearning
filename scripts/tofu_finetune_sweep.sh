#!/bin/bash

set -euo pipefail

model="${MODEL:-Llama-3.2-1B-Instruct}"
epochs="${SWEEP_EPOCHS:-5}"
batch_size="${SWEEP_BATCH_SIZE:-8}"
grad_accum="${SWEEP_GRAD_ACCUM:-4}"
grad_ckpt="${SWEEP_GRAD_CKPT:-true}"
lr_values="${SWEEP_LR_VALUES:-1e-5 1.5e-5 2e-5}"
warmup_values="${SWEEP_WARMUP_VALUES:-0.2 0.5}"
weight_decay_values="${SWEEP_WEIGHT_DECAY_VALUES:-0.01}"
run_eval="${SWEEP_POST_EVAL:-true}"
forget_split="${SWEEP_FORGET_SPLIT:-forget10}"
holdout_split="${SWEEP_HOLDOUT_SPLIT:-holdout10}"
retain_split="${SWEEP_RETAIN_SPLIT:-retain99}"

echo "[sweep] model=${model} epochs=${epochs} batch=${batch_size} grad_accum=${grad_accum} grad_ckpt=${grad_ckpt}"
echo "[sweep] lr_values=${lr_values} warmup_values=${warmup_values} weight_decay_values=${weight_decay_values}"
echo "[sweep] post_eval=${run_eval} forget_split=${forget_split} holdout_split=${holdout_split} retain_split=${retain_split}"

for lr in ${lr_values}; do
  for warmup in ${warmup_values}; do
    for wd in ${weight_decay_values}; do
      lr_tag="$(echo "${lr}" | sed 's/[^0-9A-Za-z]/_/g')"
      warmup_tag="$(echo "${warmup}" | sed 's/[^0-9A-Za-z]/_/g')"
      wd_tag="$(echo "${wd}" | sed 's/[^0-9A-Za-z]/_/g')"
      run_name="tofu_${model}_full_sweep_bs${batch_size}_ga${grad_accum}_lr${lr_tag}_wu${warmup_tag}_wd${wd_tag}_e${epochs}"

      echo "[sweep] ===== run ${run_name} ====="

      CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=finetune/tofu/default.yaml \
        task_name=${run_name} \
        model=${model} \
        data/datasets@data.train=TOFU_QA_full \
        data.train.TOFU_QA_full.args.hf_args.name=full \
        trainer.args.per_device_train_batch_size=${batch_size} \
        trainer.args.gradient_accumulation_steps=${grad_accum} \
        trainer.args.gradient_checkpointing=${grad_ckpt} \
        trainer.args.learning_rate=${lr} \
        trainer.args.warmup_epochs=${warmup} \
        trainer.args.weight_decay=${wd} \
        trainer.args.num_train_epochs=${epochs} \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.eval_strategy=no

      if [[ "${run_eval}" == "true" ]]; then
        eval_out="saves/eval/${run_name}/evals_${forget_split}"
        CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
          forget_split=${forget_split} \
          holdout_split=${holdout_split} \
          task_name=${run_name}_${forget_split} \
          model=${model} \
          model.model_args.pretrained_model_name_or_path=saves/finetune/${run_name} \
          retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
          paths.output_dir=${eval_out}

        summary_path="${eval_out}/TOFU_SUMMARY.json"
        if [[ -f "${summary_path}" ]]; then
          python - "${run_name}" "${summary_path}" <<'PY'
import json
import sys

run_name = sys.argv[1]
summary_path = sys.argv[2]
with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)
keys = [
    "model_utility",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "extraction_strength",
    "privleak",
]
vals = [str(data.get(k, "")) for k in keys]
print("run_name\t" + "\t".join(keys))
print(run_name + "\t" + "\t".join(vals))
PY
        else
          echo "[sweep] missing summary: ${summary_path}"
        fi
      fi
    done
  done
done

echo "[sweep] complete"