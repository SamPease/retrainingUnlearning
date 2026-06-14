#!/bin/bash

set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}"

model="${MODEL:-Llama-3.2-1B-Instruct}"
train_split="${TRAIN_SPLIT:-retain95}"
forget_split="${FORGET_SPLIT:-forget05}"
holdout_split="${HOLDOUT_SPLIT:-holdout05}"
retain_split="${RETAIN_SPLIT:-retain95}"

batch_size="${BATCH_SIZE:-8}"
grad_accum="${GRAD_ACCUM:-4}"
lr="${LR:-1e-5}"
warmup_epochs="${WARMUP_EPOCHS:-1.0}"
weight_decay="${WEIGHT_DECAY:-0.0}"
epochs="${EPOCHS:-10}"
attn_implementation="${ATTN_IMPLEMENTATION:-flash_attention_2}"
nccl_safe_mode="${NCCL_SAFE_MODE:-1}"

run_name="${RUN_NAME:-tofu_${model}_${train_split}_repro2gpu_e${epochs}_lr1e5_bs${batch_size}_ga${grad_accum}}"
finetune_dir="saves/finetune/${run_name}"

retain_eval_dir="saves/eval/${run_name}/retain_logs"
retain_eval_json="${retain_eval_dir}/TOFU_EVAL.json"

forget_eval_dir="saves/eval/${run_name}/evals_${forget_split}"
forget_summary_json="${forget_eval_dir}/TOFU_SUMMARY.json"

echo "[retain95-repro-2gpu] master_port=${MASTER_PORT}"
echo "[retain95-repro-2gpu] model=${model} train_split=${train_split} forget_split=${forget_split} holdout_split=${holdout_split} retain_split=${retain_split}"
echo "[retain95-repro-2gpu] batch_size=${batch_size} grad_accum=${grad_accum} lr=${lr} warmup_epochs=${warmup_epochs} weight_decay=${weight_decay} epochs=${epochs}"
echo "[retain95-repro-2gpu] attn_implementation=${attn_implementation}"
echo "[retain95-repro-2gpu] nccl_safe_mode=${nccl_safe_mode}"
echo "[retain95-repro-2gpu] run_name=${run_name}"

if [[ "${nccl_safe_mode}" == "1" ]]; then
  # Runtime-only transport fallbacks for environments that stall at first allreduce.
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  export NCCL_SHM_DISABLE=1
fi

# Train retain95 with the original repro distributed setup: Accelerate + DeepSpeed ZeRO-3 over two GPUs.
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port "${MASTER_PORT}" \
  src/train.py \
  experiment=finetune/tofu/default.yaml \
  task_name="${run_name}" \
  model="${model}" \
  model.model_args.attn_implementation="${attn_implementation}" \
  data/datasets@data.train=TOFU_QA_full \
  data.train.TOFU_QA_full.args.hf_args.name="${train_split}" \
  trainer.args.per_device_train_batch_size="${batch_size}" \
  trainer.args.gradient_accumulation_steps="${grad_accum}" \
  trainer.args.learning_rate="${lr}" \
  trainer.args.warmup_epochs="${warmup_epochs}" \
  trainer.args.weight_decay="${weight_decay}" \
  trainer.args.num_train_epochs="${epochs}" \
  trainer.args.optim=paged_adamw_32bit \
  trainer.args.do_eval=false \
  trainer.args.eval_on_start=false \
  trainer.args.eval_strategy=no

if [[ ! -f "${finetune_dir}/config.json" ]]; then
  echo "[retain95-repro-2gpu] ERROR: missing trained model config at ${finetune_dir}/config.json"
  exit 1
fi

# Generate retain reference logs for forget_quality computation.
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu_minimal.yaml \
  task_name="${run_name}_${retain_split}_reference" \
  model="${model}" \
  forget_split="${retain_split}" \
  holdout_split="${holdout_split}" \
  model.model_args.pretrained_model_name_or_path="${finetune_dir}" \
  retain_logs_path=null \
  paths.output_dir="${retain_eval_dir}"

if [[ ! -f "${retain_eval_json}" ]]; then
  echo "[retain95-repro-2gpu] ERROR: missing retain reference logs at ${retain_eval_json}"
  exit 1
fi

# Evaluate forget split with only the three minimal metrics.
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu_minimal.yaml \
  task_name="${run_name}_${forget_split}" \
  model="${model}" \
  forget_split="${forget_split}" \
  holdout_split="${holdout_split}" \
  model.model_args.pretrained_model_name_or_path="${finetune_dir}" \
  retain_logs_path="${retain_eval_json}" \
  paths.output_dir="${forget_eval_dir}"

if [[ ! -f "${forget_summary_json}" ]]; then
  echo "[retain95-repro-2gpu] ERROR: missing summary file at ${forget_summary_json}"
  exit 1
fi

python - "${forget_summary_json}" <<'PY'
import json
import sys

summary_path = sys.argv[1]

targets = {
    "forget_quality": 1.0,
    "model_utility": 0.63,
    "forget_truth_ratio": 0.67,
}

with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

print("[retain95-repro-2gpu] metric\tvalue\ttarget\tdelta(value-target)")
for metric in ["forget_quality", "model_utility", "forget_truth_ratio"]:
    value = summary.get(metric, None)
    target = targets[metric]
    if value is None:
        print(f"[retain95-repro-2gpu] {metric}\tNone\t{target}\tNone")
    else:
        delta = float(value) - float(target)
        print(f"[retain95-repro-2gpu] {metric}\t{value}\t{target}\t{delta:+.6f}")
PY

echo "[retain95-repro-2gpu] done"
echo "[retain95-repro-2gpu] model_dir=${finetune_dir}"
echo "[retain95-repro-2gpu] eval_summary=${forget_summary_json}"