"""Phase 1 check: print fully formatted senior, junior and neutral prompts for one question.

    python scripts/show_prompts.py --config configs/local.yaml [--pair ong_chau] [--template flat] [--all]
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from vnpronouns.config import load_config
from vnpronouns.prompts import PRONOUN_PAIRS, PUSHBACK_TEMPLATES, SAMPLE_QUESTION, all_conditions, build_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--pair", default="ong_chau", choices=list(PRONOUN_PAIRS))
    ap.add_argument("--template", default="flat", choices=list(PUSHBACK_TEMPLATES))
    ap.add_argument("--all", action="store_true", help="print a one-line summary of all 13 x 3 variants too")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    print(f"model: {cfg.model_name}   pair: {args.pair}   template: {args.template}\n")

    wanted = {f"{args.pair}_senior", f"{args.pair}_junior", "neutral"}
    for cond in [c for c in all_conditions() if c.name in wanted]:
        p = build_prompt(tok, SAMPLE_QUESTION, cond, args.template)
        n_tok = len(tok(p.text, add_special_tokens=False).input_ids)
        print(f"===== {cond.name}  (user={cond.user_pron}, model={cond.model_pron}, {n_tok} tokens) =====")
        print(p.text)
        print(f"----- pushback turn starts at char {p.pushback_char_start}: "
              f"{p.text[p.pushback_char_start:p.pushback_char_start + 30]!r}...\n")

    if args.all:
        print("===== all variants (condition / template / tokens / pushback turn) =====")
        for cond in all_conditions():
            for t in PUSHBACK_TEMPLATES:
                p = build_prompt(tok, SAMPLE_QUESTION, cond, t)
                n_tok = len(tok(p.text, add_special_tokens=False).input_ids)
                push = p.text[p.pushback_char_start:].split("\n")[1]
                print(f"{cond.name:16s} {t:8s} {n_tok:4d}  {push}")


if __name__ == "__main__":
    main()
