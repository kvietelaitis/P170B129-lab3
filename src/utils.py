"""Shared helpers: config loading, seeding, device selection, environment logging."""
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def load_config(path: str | os.PathLike = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fix all random states (project requirement)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    from transformers import set_seed as hf_set_seed
    hf_set_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available, running on CPU (slow).")
    return torch.device("cpu")


def mixed_precision() -> dict:
    """Trainer flags: bf16 on Ampere+ GPUs, fp16 on older ones, none on CPU."""
    if not torch.cuda.is_available():
        return {}
    return {"bf16": True} if torch.cuda.is_bf16_supported() else {"fp16": True}


def env_info() -> dict:
    """Library versions to store with every run (project requirement)."""
    import datasets, peft, transformers
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "device": str(get_device()),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_run_metadata(run_dir: Path, cfg: dict, extra: dict | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(run_dir / "env.json", "w") as f:
        json.dump({**env_info(), **(extra or {})}, f, indent=2)
