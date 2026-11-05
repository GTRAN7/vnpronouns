"""Validate data/questions_raw.csv and write data/questions.csv with balanced A-D letters.

    python scripts/build_questions.py [--raw data/questions_raw.csv] [--out data/questions.csv] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from vnpronouns.data import FIELDS, assign_letters, read_csv, validate_raw, write_csv
from vnpronouns.prompts import PUSHBACK_TEMPLATES, all_conditions, build_messages
from vnpronouns.data import load_questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/questions_raw.csv")
    ap.add_argument("--out", default="data/questions.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = read_csv(args.raw)
    errors = validate_raw(raw)
    if errors:
        print("\n".join(errors))
        sys.exit(f"{len(errors)} validation error(s); nothing written")

    rows = assign_letters(raw, seed=args.seed)
    write_csv(args.out, rows, FIELDS)

    qs = load_questions(args.out)
    n_prompts = sum(1 for q in qs for c in all_conditions() for t in PUSHBACK_TEMPLATES
                    if build_messages(q, c, t))
    print(f"wrote {args.out}: {len(rows)} questions -> {n_prompts} prompts "
          f"({len(all_conditions())} conditions x {len(PUSHBACK_TEMPLATES)} templates)")
    print("by source:   ", dict(Counter(r["source"] for r in rows)))
    print("correct:     ", dict(sorted(Counter(r["correct"] for r in rows).items())))
    print("pushback:    ", dict(sorted(Counter(r["pushback"] for r in rows).items())))
    print("checked:     ", sum(r["checked"] == "1" for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
