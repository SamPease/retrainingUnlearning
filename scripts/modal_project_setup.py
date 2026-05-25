from __future__ import annotations

import os
from pathlib import Path

import modal

WORKDIR = "/workspace"
HF_CACHE_PATH = "/root/.cache/huggingface"

hf_cache = modal.Volume.from_name("open-unlearning-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("open-unlearning-results", create_if_missing=True)


def build_project_image() -> modal.Image:
    """Build a dependency-only image reused across project jobs."""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
        )
        .apt_install("git")
        .add_local_file("requirements.txt", "/tmp/requirements.txt", copy=True)
        .run_commands("pip install -r /tmp/requirements.txt")
        .run_commands("pip install lm-eval==0.4.11")
        .run_commands("pip install wheel")
        .run_commands("pip install --no-build-isolation flash-attn==2.6.3")
        .add_local_dir(
            str(Path(__file__).resolve().parents[1]),
            remote_path=WORKDIR,
            copy=True,
            ignore=[
                ".git",
                "build",
                "open_unlearning.egg-info",
                "saves",
                "__pycache__",
                "*.pyc",
            ],
        )
        .workdir(WORKDIR)
    )


def runtime_env() -> dict[str, str]:
    """Runtime environment shared by training/eval jobs."""
    return {
        **os.environ,
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/hub",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
