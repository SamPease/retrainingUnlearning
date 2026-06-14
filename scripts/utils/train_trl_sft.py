#!/usr/bin/env python3
"""Train a TOFU forget split model with TRL SFT + LoRA.

This script is intentionally standalone and does not depend on the repo's Hydra
training stack, so it can be used as an isolated TRL workflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL SFT finetune on TOFU forget split")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
        help="Base model to finetune.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="locuslab/TOFU",
        help="HF dataset path.",
    )
    parser.add_argument(
        "--train_split_name",
        type=str,
        default="forget10",
        help="TOFU subset name to load as train data (for example forget10).",
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="saves/finetune/tofu_Llama-3.2-1B-Instruct_retain90_trl_forget10_lora",
    )
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA.",
    )

    return parser.parse_args()


def _resolve_dtype() -> tuple[torch.dtype, bool, bool]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


def _build_text(example: dict[str, Any], tokenizer: AutoTokenizer) -> dict[str, str]:
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch_dtype, use_bf16, use_fp16 = _resolve_dtype()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.gradient_checkpointing:
        model.config.use_cache = False

    raw_train = load_dataset(
        args.dataset_path,
        name=args.train_split_name,
        split=args.dataset_split,
    )

    train_dataset = raw_train.map(
        lambda sample: _build_text(sample, tokenizer),
        remove_columns=raw_train.column_names,
        desc="Formatting TOFU samples as chat transcripts",
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
