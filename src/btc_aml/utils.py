"""Small shared helpers: config, paths, seeding, logging, device, timing.

Nothing in here knows about Bitcoin or graphs. It exists so every other module
can stay focused on one job.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

# The repository root is two levels above this file: <root>/src/btc_aml/utils.py
ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Read a YAML config file into a plain dictionary.

    Every experiment setting lives in configs/*.yaml, so a run is fully
    described by (code version, config file). Nothing is hard-coded in the
    training scripts.
    """
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    """Every CLI entry point takes the same --config flag."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default="configs/quick.yaml",
        help="Path to a YAML config (configs/quick.yaml or configs/full.yaml).",
    )
    return parser


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def resolve(path: str | os.PathLike) -> Path:
    """Turn a config path (relative to the repo root) into an absolute Path."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def data_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"]["data_dir"])


def results_dir(cfg: dict) -> Path:
    path = resolve(cfg["paths"]["results_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(cfg: dict) -> Path:
    path = results_dir(cfg) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(cfg: dict) -> Path:
    path = resolve(cfg["paths"]["models_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every random number generator we use.

    A "seed" fixes the random choices (weight initialisation, dropout masks,
    which rows a tree samples). Same seed + same code + same machine gives the
    same numbers. We repeat each experiment over several seeds and report
    mean +/- std so one lucky initialisation cannot decide a conclusion.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict | None = None) -> torch.device:
    """Pick the device: CUDA if present, otherwise CPU.

    `device: auto` in the config means "use a CUDA GPU if there is one".
    Apple's MPS backend is only used when asked for explicitly
    (`device: mps`), because some sparse graph operations are slower or
    unsupported there and the graphs here are small enough for a CPU.
    """
    wanted = (cfg or {}).get("device", "auto")
    if wanted == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(wanted)


# --------------------------------------------------------------------------- #
# Logging and timing
# --------------------------------------------------------------------------- #
def get_logger(name: str = "btc_aml") -> logging.Logger:
    """One logger for the whole project, printing time + message."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextmanager
def timed(label: str, store: dict | None = None) -> Iterator[None]:
    """Time a block of code: `with timed("baselines", timings): ...`.

    If `store` is given, the elapsed seconds are written into it under `label`,
    which is how run_all.py builds results/timings.json.
    """
    log = get_logger()
    start = time.perf_counter()
    log.info("START %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if store is not None:
            store[label] = round(elapsed, 1)
        log.info("DONE  %s in %.1fs", label, elapsed)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
