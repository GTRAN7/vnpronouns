"""Run configuration. Model name, device, precision and output paths live here, never in code.

Load with `load_config("configs/local.yaml")`. Any field can be overridden by an
environment variable named VNP_<FIELD> (e.g. VNP_OUTPUT_DIR=/content/drive/...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

_DTYPES = ("float32", "float16", "bfloat16")


@dataclass
class Config:
    model_name: str
    device: str = "auto"
    dtype: str = "float32"
    load_in_8bit: bool = False
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {_DTYPES}, got {self.dtype!r}")

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)


def load_config(path: str | os.PathLike) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    for fld in fields(Config):
        env = os.environ.get(f"VNP_{fld.name.upper()}")
        if env is not None:
            raw[fld.name] = env.lower() in ("1", "true", "yes") if fld.type == "bool" else env
    return Config(**raw)
