#!/bin/bash

set -euo pipefail

model="Llama-3.2-1B-Instruct"

########################################################################################################################
########################################### FULL Finetuned TOFU models #################################################
########################################################################################################################

# Stability sanity run: single GPU, no distributed launcher.
CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=finetune/tofu/default.yaml \
task_name=tofu_${model}_full \
model=${model} \
data/datasets@data.train=TOFU_QA_full \
data.train.TOFU_QA_full.args.hf_args.name=full \
trainer.args.per_device_train_batch_size=4 \
trainer.args.gradient_accumulation_steps=8 \
trainer.args.gradient_checkpointing=true

# Single eval split for initial sanity check after training.
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
forget_split=forget10 \
holdout_split=holdout10 \
task_name=tofu_${model}_full_forget10 \
model=${model} \
model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
retain_logs_path=saves/eval/tofu_${model}_retain99/TOFU_EVAL.json \
paths.output_dir=saves/eval/tofu_${model}_full/evals_forget10