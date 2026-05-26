#!/bin/bash

set -euo pipefail

model="${MODEL:-Llama-3.2-1B-Instruct}"
run_name="${RUN_NAME:-tofu_${model}_full_light_eval}"

CUDA_VISIBLE_DEVICES=0 python src/train.py \
  experiment=finetune/tofu/light_eval.yaml \
  task_name=${run_name} \
  model=${model} \
  data/datasets@data.train=TOFU_QA_full \
  data.train.TOFU_QA_full.args.hf_args.name=full

echo "[light-eval] complete: ${run_name}"
