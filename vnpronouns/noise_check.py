"""How much do scores move with batch composition / precision? (diagnostic, before Phase 5)

    python -m vnpronouns.noise_check --config configs/colab.yaml                      # bf16
    VNP_DTYPE=float32 python -m vnpronouns.noise_check --config configs/colab.yaml    # fp32 reference

The Phase 4 run (behavior_v1_3templates.csv) and the Phase 4b run (behavior.csv) scored the
same 13,962 prompts in different batches. This prints the distribution of their |dP(correct)|,
then re-scores the --n-top most-changed prompts plus --n-random random ones
  bs1     one prompt at a time (no padding at all)
  sorted  batched, with prompts sorted by length (little padding)
and compares v1 / new / sorted against bs1. Run once in bf16 and once with VNP_DTYPE=float32:
the fp32 run also compares against the saved bf16 numbers, so we can tell batching noise
(bf16 batched != bf16 bs1) from precision noise (bf16 bs1 != fp32 bs1).

Writes noise_check_<dtype>.csv.
"""

from __future__ import annotations

import pandas as pd

from vnpronouns.common import base_args, config_from_args, prompt_table
from vnpronouns.data import load_questions
from vnpronouns.model import load_model, score_prompts
from vnpronouns.run_behavior import add_scores

KEY = ["qid", "condition", "template"]


def describe(name: str, a: pd.Series, b: pd.Series, fa: pd.Series, fb: pd.Series) -> None:
    d = (a - b).abs()
    print(f"  {name:28s} |dP| median {d.median():.4f}  p90 {d.quantile(.9):.4f}  p99 {d.quantile(.99):.4f}  "
          f"max {d.max():.4f}  | >0.1: {(d > .1).mean():.1%}  | same flip: {(fa == fb).mean():.1%}")


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--n-top", type=int, default=150)
    ap.add_argument("--n-random", type=int, default=150)
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    v1 = pd.read_csv(out / "behavior_v1_3templates.csv")
    new = pd.read_csv(out / "behavior.csv")
    m = v1[KEY + ["p_correct", "flip"]].merge(new[KEY + ["p_correct", "flip"]], on=KEY, suffixes=("_v1", "_new"))
    m["d"] = (m.p_correct_v1 - m.p_correct_new).abs()
    print(f"v1 vs new on all {len(m)} shared prompts (different batch composition, same dtype):")
    describe("v1 vs new", m.p_correct_v1, m.p_correct_new, m.flip_v1, m.flip_new)
    print("  largest |dP|:")
    print(m.nlargest(8, "d")[KEY + ["p_correct_v1", "p_correct_new"]].round(4).to_string(index=False))

    top = m.nlargest(args.n_top, "d")
    rest = m.drop(top.index)
    rnd = rest.sample(min(args.n_random, len(rest)), random_state=cfg.seed)
    sel = pd.concat([top.assign(subset="top"), rnd.assign(subset="random")])

    model, tok = load_model(cfg)
    qs = [q for q in load_questions(args.questions) if q.qid in set(sel.qid)]
    table = prompt_table(tok, qs).merge(sel[KEY + ["subset"]], on=KEY)
    texts = table.text.tolist()
    print(f"\nre-scoring {len(table)} prompts in {cfg.dtype}: one at a time, and batched by length "
          f"(batch {cfg.batch_size})")
    bs1 = add_scores(table.drop(columns="text"), score_prompts(model, tok, texts, 1, progress=True).probs)
    srt = add_scores(table.drop(columns="text"),
                     score_prompts(model, tok, texts, cfg.batch_size, progress=True, sort_by_length=True).probs)
    res = table[KEY + ["subset", "correct", "pushback"]].copy()
    res[f"p_bs1"] = bs1.p_correct.to_numpy()
    res[f"flip_bs1"] = (bs1.top == bs1.pushback).to_numpy()
    res[f"p_sorted"] = srt.p_correct.to_numpy()
    res[f"flip_sorted"] = (srt.top == srt.pushback).to_numpy()
    res = res.merge(sel[KEY + ["p_correct_v1", "p_correct_new", "flip_v1", "flip_new"]], on=KEY)
    res["dtype"] = cfg.dtype
    res.to_csv(out / f"noise_check_{cfg.dtype}.csv", index=False)

    for subset in ("top", "random"):
        r = res[res.subset == subset]
        print(f"\n[{subset} {len(r)}] each compared with {cfg.dtype} one-at-a-time (bs1):")
        describe("v1 (Phase 4 batches)", r.p_correct_v1, r.p_bs1, r.flip_v1, r.flip_bs1)
        describe("new (Phase 4b batches)", r.p_correct_new, r.p_bs1, r.flip_new, r.flip_bs1)
        describe("length-sorted batches", r.p_sorted, r.p_bs1, r.flip_sorted, r.flip_bs1)

    other = out / "noise_check_bfloat16.csv"
    if cfg.dtype == "float32" and other.exists():
        bf = pd.read_csv(other)[KEY + ["p_bs1", "flip_bs1", "p_sorted", "flip_sorted"]]
        j = res.merge(bf, on=KEY, suffixes=("_fp32", "_bf16"))
        for subset in ("top", "random"):
            r = j[j.subset == subset]
            print(f"\n[{subset} {len(r)}] bf16 vs the fp32 one-at-a-time reference:")
            describe("bf16 bs1", r.p_bs1_bf16, r.p_bs1_fp32, r.flip_bs1_bf16, r.flip_bs1_fp32)
            describe("bf16 length-sorted", r.p_sorted_bf16, r.p_bs1_fp32, r.flip_sorted_bf16, r.flip_bs1_fp32)
            describe("bf16 v1 batches", r.p_correct_v1, r.p_bs1_fp32, r.flip_v1, r.flip_bs1_fp32)
            describe("bf16 new batches", r.p_correct_new, r.p_bs1_fp32, r.flip_new, r.flip_bs1_fp32)
    print(f"\nwrote {out / f'noise_check_{cfg.dtype}.csv'}")


if __name__ == "__main__":
    main()
