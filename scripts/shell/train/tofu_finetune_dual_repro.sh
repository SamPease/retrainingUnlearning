#!/bin/bash

set -euo pipefail

model="${MODEL:-Llama-3.2-1B-Instruct}"
epochs="${EPOCHS:-20}"
batch_size="${BATCH_SIZE:-8}"
grad_accum="${GRAD_ACCUM:-4}"
lr="${LR:-1e-5}"
warmup_epochs="${WARMUP_EPOCHS:-1.0}"
weight_decay="${WEIGHT_DECAY:-0.0}"
grad_ckpt="${GRAD_CKPT:-false}"
run_mode="${RUN_MODE:-both}"

run_trial () {
  local train_split="$1"
  local forget_split="$2"
  local holdout_split="$3"
  local retain_split="$4"

  local run_name="tofu_${model}_${train_split}_repro1gpu_min_eval_e${epochs}_lr1e5_bs${batch_size}_ga${grad_accum}"
  local retain_logs="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"

  if [[ ! -f "${retain_logs}" ]]; then
    echo "[dual-repro-1gpu] ERROR: missing retain reference logs at ${retain_logs}"
    echo "[dual-repro-1gpu] Please generate retain logs first (or copy them onto the results volume)."
    exit 1
  fi

  echo "[dual-repro-1gpu] starting ${train_split} -> run_name=${run_name}"
  echo "[dual-repro-1gpu] params: lr=${lr} batch_size=${batch_size} grad_accum=${grad_accum} epochs=${epochs} warmup_epochs=${warmup_epochs} weight_decay=${weight_decay}"
  echo "[dual-repro-1gpu] eval: forget_split=${forget_split} holdout_split=${holdout_split} retain_logs=${retain_logs}"

  CUDA_VISIBLE_DEVICES=0 python src/train.py \
    experiment=finetune/tofu/light_eval.yaml \
    eval=tofu_minimal \
    task_name="${run_name}" \
    model="${model}" \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name="${train_split}" \
    trainer.args.per_device_train_batch_size="${batch_size}" \
    trainer.args.gradient_accumulation_steps="${grad_accum}" \
    trainer.args.gradient_checkpointing="${grad_ckpt}" \
    trainer.args.learning_rate="${lr}" \
    trainer.args.warmup_epochs="${warmup_epochs}" \
    trainer.args.weight_decay="${weight_decay}" \
    trainer.args.num_train_epochs="${epochs}" \
    trainer.args.optim=paged_adamw_32bit \
    trainer.args.do_eval=true \
    trainer.args.eval_on_start=false \
    trainer.args.eval_strategy=epoch \
    trainer.args.save_strategy=no \
    forget_split="${forget_split}" \
    holdout_split="${holdout_split}" \
    retain_logs_path="${retain_logs}"

  echo "[dual-repro-1gpu] complete ${train_split} -> saves/finetune/${run_name}"
}

case "${run_mode}" in
  full)
    # Full TOFU run uses forget10/holdout10 with retain99 reference logs.
    run_trial "full" "forget10" "holdout10" "retain99"
    ;;
  retain95)
    # Retain95 run uses forget05/holdout05 with retain95 reference logs.
    run_trial "retain95" "forget05" "holdout05" "retain95"
    ;;
  both)
    run_trial "full" "forget10" "holdout10" "retain99"
    run_trial "retain95" "forget05" "holdout05" "retain95"
    ;;
  *)
    echo "[dual-repro-1gpu] ERROR: RUN_MODE must be one of: full, retain95, both"
    exit 1
    ;;
esac

echo "[dual-repro-1gpu] done: run_mode=${run_mode}"
