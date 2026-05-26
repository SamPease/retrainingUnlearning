.PHONY: quality style train-tofu-light-eval report-tofu-runs

check_dirs := scripts src #setup.py

quality:
	ruff check $(check_dirs) setup.py setup_data.py
	ruff format --check $(check_dirs) setup.py setup_data.py

style:
	ruff check $(check_dirs) setup.py setup_data.py --fix
	ruff format $(check_dirs) setup.py setup_data.py

test:
	CUDA_VISIBLE_DEVICES= pytest tests/

train-tofu-light-eval:
	bash scripts/tofu_finetune_light_eval.sh

report-tofu-runs:
	python scripts/generate_tofu_epoch_report.py --output experiments/reports/training-runs.md
