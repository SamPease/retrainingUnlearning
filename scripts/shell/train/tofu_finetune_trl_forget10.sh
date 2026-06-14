#!/bin/bash

set -euo pipefail

model_name_or_path="${MODEL_NAME_OR_PATH:-open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90}"
train_split_name="${TRAIN_SPLIT_NAME:-forget10}"
output_dir="${OUTPUT_DIR:-saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget10_lora}"

epochs="${EPOCHS:-3}"
lr="${LR:-1e-5}"
warmup_ratio="${WARMUP_RATIO:-0.03}"
weight_decay="${WEIGHT_DECAY:-0.0}"
batch_size="${BATCH_SIZE:-4}"
grad_accum="${GRAD_ACCUM:-4}"
max_seq_length="${MAX_SEQ_LENGTH:-1024}"
lora_r="${LORA_R:-16}"
lora_alpha="${LORA_ALPHA:-32}"
seed="${SEED:-42}"

echo "[tofu-trl] model_name_or_path=${model_name_or_path}"
echo "[tofu-trl] train_split_name=${train_split_name} output_dir=${output_dir}"
echo "[tofu-trl] epochs=${epochs} lr=${lr} warmup_ratio=${warmup_ratio} weight_decay=${weight_decay}"
echo "[tofu-trl] batch_size=${batch_size} grad_accum=${grad_accum} max_seq_length=${max_seq_length}"
echo "[tofu-trl] lora_r=${lora_r} lora_alpha=${lora_alpha}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_tofu_forget10_trl_sft.py \
  --model_name_or_path "${model_name_or_path}" \
  --train_split_name "${train_split_name}" \
  --output_dir "${output_dir}" \
  --num_train_epochs "${epochs}" \
  --learning_rate "${lr}" \
  --warmup_ratio "${warmup_ratio}" \
  --weight_decay "${weight_decay}" \
  --per_device_train_batch_size "${batch_size}" \
  --gradient_accumulation_steps "${grad_accum}" \
  --max_seq_length "${max_seq_length}" \
  --lora_r "${lora_r}" \
  --lora_alpha "${lora_alpha}" \
  --seed "${seed}" \
  --gradient_checkpointing

echo "[tofu-trl] complete: ${output_dir}"
