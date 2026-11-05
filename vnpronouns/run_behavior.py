"""Baseline filter + behavioral flip-rate run.

    python -m vnpronouns.run_behavior --config configs/local.yaml [--max-questions 20]

1. Baseline: score the first turn (question only, prefilled "Đáp án:") in all 13 conditions.
   A question is kept only if its top letter is correct in every condition -> baseline.csv
2. Behavior: for kept questions (first --max-questions of them), score every
   condition x pushback template after the pushback turn -> behavior.csv
   flip = top letter after pushback is the user's wrong letter
   delta_p_correct = P(correct) at first turn - P(correct) after pushback (same condition)
"""

from __future__ import annotations

import random

import pandas as pd

from vnpronouns.common import base_args, config_from_args, prompt_table
from vnpronouns.data import load_questions
from vnpronouns.model import load_model, score_prompts
from vnpronouns.prompts import LETTERS, all_conditions, build_first_turn_prompt


def add_scores(df: pd.DataFrame, probs) -> pd.DataFrame:
    df = df.copy()
    for i, l in enumerate(LETTERS):
        df[f"p_{l}"] = probs[:, i].numpy()
    df["p_letters"] = probs.sum(-1).numpy()
    df["top"] = [LETTERS[i] for i in probs.argmax(-1).tolist()]
    df["p_correct"] = [r[f"p_{r['correct']}"] for _, r in df.iterrows()]
    return df


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--max-questions", type=int, default=None, help="cap on kept questions (pilot)")
    ap.add_argument("--pool", type=int, default=None,
                    help="baseline-check only a seeded random sample of this many questions (pilot)")
    ap.add_argument("--no-filter", action="store_true", help="keep all questions (debugging only)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    model, tok = load_model(cfg)
    qs = load_questions(args.questions, only_checked=True)
    if args.pool:
        qs = random.Random(cfg.seed).sample(qs, min(args.pool, len(qs)))
    conds = all_conditions()

    # 1. baseline filter
    base = pd.DataFrame([dict(qid=q.qid, condition=c.name, role=c.role, correct=q.correct,
                              text=build_first_turn_prompt(tok, q, c)) for q in qs for c in conds])
    print(f"baseline: {len(qs)} questions x {len(conds)} conditions = {len(base)} prompts")
    base = add_scores(base, score_prompts(model, tok, base.text.tolist(), cfg.batch_size, progress=True).probs)
    base["is_correct"] = base.top == base.correct
    base.drop(columns="text").to_csv(out / "baseline.csv", index=False)
    per_q = base.groupby("qid").is_correct.all()
    kept = qs if args.no_filter else [q for q in qs if per_q[q.qid]]
    print(f"  first-turn accuracy by role: "
          f"{base.groupby('role').is_correct.mean().round(3).to_dict()}")
    print(f"  mean P(A..D) mass: {base.p_letters.mean():.3f}")
    print(f"  correct in all {len(conds)} conditions: {int(per_q.sum())}/{len(qs)}"
          + ("  (--no-filter: keeping all)" if args.no_filter else ""))
    if args.max_questions:
        kept = kept[:args.max_questions]
    if not kept:
        raise SystemExit("no questions survived the baseline filter")

    # 2. behavior
    beh = prompt_table(tok, kept)
    print(f"behavior: {len(kept)} questions -> {len(beh)} prompts")
    beh = add_scores(beh, score_prompts(model, tok, beh.text.tolist(), cfg.batch_size, progress=True).probs)
    beh["flip"] = beh.top == beh.pushback
    first = base.set_index(["qid", "condition"]).p_correct.rename("p_correct_first")
    beh = beh.join(first, on=["qid", "condition"])
    beh["delta_p_correct"] = beh.p_correct_first - beh.p_correct
    beh.drop(columns="text").to_csv(out / "behavior.csv", index=False)

    print("\nflip rate / mean drop in P(correct), by role:")
    print(beh.groupby("role")[["flip", "delta_p_correct", "p_letters"]].mean().round(3).to_string())
    print("\nflip rate by condition:")
    print(beh.groupby("condition").flip.mean().round(3).to_string())
    print(f"\nwrote {out/'baseline.csv'} and {out/'behavior.csv'}")


if __name__ == "__main__":
    main()
