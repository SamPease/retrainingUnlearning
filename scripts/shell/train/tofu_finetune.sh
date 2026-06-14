#!/bin/bash

set -euo pipefail

model="${MODEL:-Llama-3.2-1B-Instruct}"
batch_size="${BATCH_SIZE:-8}"
grad_accum="${GRAD_ACCUM:-4}"
lr="${LR:-2e-5}"
warmup_epochs="${WARMUP_EPOCHS:-0.2}"
weight_decay="${WEIGHT_DECAY:-0.0}"
epochs="${EPOCHS:-10}"
check_nested_splits="${CHECK_NESTED_SPLITS:-1}"
skip_existing="${SKIP_EXISTING:-1}"
run_final_eval="${RUN_FINAL_EVAL:-1}"
skip_eval_if_exists="${SKIP_EVAL_IF_EXISTS:-1}"

echo "[tofu-finetune] model=${model}"
echo "[tofu-finetune] batch_size=${batch_size} grad_accum=${grad_accum} lr=${lr} warmup_epochs=${warmup_epochs} weight_decay=${weight_decay} epochs=${epochs}"

if [[ "${check_nested_splits}" == "1" ]]; then
	echo "[tofu-finetune] verifying nested TOFU forget/holdout splits (1% subset-of 5% subset-of 10%)"
	python scripts/verify_tofu_nested_splits.py
fi

########################################################################################################################
############################# FULL + RETAIN(99/95/90) Finetuned TOFU models ##########################################
########################################################################################################################

# Format: <train_split_name> <forget_split> <holdout_split>
split_specs=(
	"full forget10 holdout10"
	"retain99 forget01 holdout01"
	"retain95 forget05 holdout05"
	"retain90 forget10 holdout10"
)

for spec in "${split_specs[@]}"; do
	read -r train_split forget_split holdout_split <<< "${spec}"
	task_name="tofu_${model}_${train_split}"
	out_dir="saves/finetune/${task_name}"
	eval_dir="saves/eval/${task_name}/evals_${forget_split}"
	model_cfg="${out_dir}/config.json"
	eval_summary="${eval_dir}/TOFU_SUMMARY.json"
	train_ran=0

	if [[ "${skip_existing}" == "1" && -d "${out_dir}" && -f "${model_cfg}" ]]; then
		echo "[tofu-finetune] skipping existing run: ${task_name}"
	else
		if [[ -d "${out_dir}" && ! -f "${model_cfg}" ]]; then
			echo "[tofu-finetune] found incomplete model dir for ${task_name}; retraining"
		fi
		echo "[tofu-finetune] training ${task_name} (train split: ${train_split})"
		CUDA_VISIBLE_DEVICES=0 python src/train.py experiment=finetune/tofu/default.yaml \
			task_name=${task_name} \
			model=${model} \
			data/datasets@data.train=TOFU_QA_full \
			data.train.TOFU_QA_full.args.hf_args.name=${train_split} \
			trainer.args.per_device_train_batch_size=${batch_size} \
			trainer.args.gradient_accumulation_steps=${grad_accum} \
			trainer.args.gradient_checkpointing=true \
			trainer.args.learning_rate=${lr} \
			trainer.args.warmup_epochs=${warmup_epochs} \
			trainer.args.weight_decay=${weight_decay} \
			trainer.args.num_train_epochs=${epochs} \
			trainer.args.do_eval=false \
			trainer.args.eval_on_start=false \
			trainer.args.eval_strategy=no
		train_ran=1
	fi

	if [[ "${run_final_eval}" != "1" ]]; then
		echo "[tofu-finetune] skipping final eval for ${task_name} (RUN_FINAL_EVAL=${run_final_eval})"
		continue
	fi

	if [[ ! -f "${model_cfg}" ]]; then
		echo "[tofu-finetune] ERROR: missing ${model_cfg}; cannot run final eval for ${task_name}"
		exit 1
	fi

	if [[ "${skip_eval_if_exists}" == "1" && "${train_ran}" == "0" && -f "${eval_summary}" ]]; then
		echo "[tofu-finetune] skipping existing final eval: ${task_name}"
		continue
	fi

	# Keep per-split eval logs for downstream unlearning runs.
	echo "[tofu-finetune] final eval ${task_name} on ${forget_split}/${holdout_split}"
	CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
		forget_split=${forget_split} \
		holdout_split=${holdout_split} \
		task_name=${task_name}_${forget_split} \
		model=${model} \
		model.model_args.pretrained_model_name_or_path=${out_dir} \
		retain_logs_path=null \
		paths.output_dir=${eval_dir}
done

echo "[tofu-finetune] all runs complete"