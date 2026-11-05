"""Helpers shared by run_behavior / extract / steer."""

from __future__ import annotations

import argparse
import random

import pandas as pd

from vnpronouns.config import Config, load_config
from vnpronouns.model import char_to_token_index
from vnpronouns.prompts import PUSHBACK_TEMPLATES, Condition, Question, all_conditions, build_prompt


def base_args(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--questions", default="data/questions.csv")
    ap.add_argument("--batch-size", type=int, default=None, help="override config batch_size")
    return ap


def config_from_args(args) -> Config:
    cfg = load_config(args.config)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    return cfg


def prompt_table(tok, qs: list[Question], conds: list[Condition] | None = None,
                 templates: list[str] | None = None) -> pd.DataFrame:
    """One row per (question, condition, template) with the full prompt and the pushback token index."""
    rows = []
    for q in qs:
        for c in conds or all_conditions():
            for t in templates or list(PUSHBACK_TEMPLATES):
                p = build_prompt(tok, q, c, t)
                rows.append(dict(qid=q.qid, condition=c.name, pair=c.pair, role=c.role, template=t,
                                 correct=q.correct, pushback=q.pushback, text=p.text,
                                 pushback_tok=char_to_token_index(tok, p.text, p.pushback_char_start)))
    return pd.DataFrame(rows)


def split_qids(qids: list[str], seed: int) -> tuple[list[str], list[str]]:
    """50/50 train/test split by question (all variants of a question land on the same side)."""
    q = sorted(set(qids))
    random.Random(seed).shuffle(q)
    half = len(q) // 2
    return sorted(q[:half]), sorted(q[half:])
