from __future__ import annotations

import statistics
import sys

import modal
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/scripts")
sys.path.insert(0, "/workspace/src")

from modal_project_setup import (  # noqa: E402
    WORKDIR,
    build_project_image,
    hf_cache,
    results,
    runtime_env,
)
from omegaconf import OmegaConf  # noqa: E402

APP_NAME = "open-unlearning-tofu-data-sanity-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_data_sanity(sample_count: int = 200, max_length: int = 512) -> None:
    sys.path.insert(0, f"{WORKDIR}/src")
    from data.utils import preprocess_chat_instance

    cfg = OmegaConf.load(f"{WORKDIR}/configs/model/Llama-3.2-1B-Instruct.yaml")
    model_name = cfg.model_args.pretrained_model_name_or_path

    tok = AutoTokenizer.from_pretrained(model_name, token=runtime_env().get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("locuslab/TOFU", name="full", split="train")
    n = len(ds)
    print(f"dataset_len={n}")
    print(f"columns={ds.column_names}")

    if sample_count > n:
        sample_count = n

    step = max(1, n // sample_count)
    indices = list(range(0, n, step))[:sample_count]

    input_lens = []
    supervised_counts = []

    for i in indices:
        row = ds[int(i)]
        out = preprocess_chat_instance(
            tokenizer=tok,
            template_config=cfg.template_args,
            prompt_msgs=[row["question"]],
            response_msgs=[row["answer"]],
            max_length=max_length,
            predict_with_generate=False,
        )
        labels = out["labels"]
        input_ids = out["input_ids"]

        input_lens.append(int(input_ids.numel()))
        supervised_counts.append(int((labels != -100).sum().item()))

    zero_supervised = sum(1 for x in supervised_counts if x == 0)
    under8 = sum(1 for x in supervised_counts if x < 8)
    under16 = sum(1 for x in supervised_counts if x < 16)

    print(f"sample_count={len(indices)}")
    print(f"avg_input_len={statistics.mean(input_lens):.4f}")
    print(f"avg_supervised_tokens={statistics.mean(supervised_counts):.4f}")
    print(f"min_supervised_tokens={min(supervised_counts)}")
    print(f"pct_zero_supervised={zero_supervised / len(indices):.6f}")
    print(f"pct_supervised_under_8={under8 / len(indices):.6f}")
    print(f"pct_supervised_under_16={under16 / len(indices):.6f}")


@app.local_entrypoint()
def main(sample_count: int = 200, max_length: int = 512) -> None:
    run_data_sanity.remote(sample_count=sample_count, max_length=max_length)
