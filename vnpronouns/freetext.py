"""Reality check: does the letter metric match what the model actually writes?

    python -m vnpronouns.freetext --config configs/colab_t4.yaml [--n-questions 50] [--template flat]

For a seeded sample of kept questions (from behavior.csv), in all 13 conditions, the model
writes a free reply to the pushback (no "Đáp án:" prefill, greedy decoding). We pull the
first standalone letter A-D out of the reply (or, failing that, the single option text it
mentions) and compare it with the letter metric's top letter for the same prompt.

Writes freetext.csv (one row per reply, with the reply text, for reading by hand).
"""

from __future__ import annotations

import random
import re

import pandas as pd
import torch

from vnpronouns.common import base_args, config_from_args
from vnpronouns.data import load_questions
from vnpronouns.model import load_model
from vnpronouns.prompts import LETTERS, all_conditions, build_prompt

_LETTER = re.compile(r"(?<![0-9A-Za-zÀ-ỹ])([ABCD])(?![0-9A-Za-zÀ-ỹ])")


def parse_letter(reply: str, options: dict[str, str]) -> str | None:
    m = _LETTER.search(reply)
    if m:
        return m.group(1)
    low = reply.lower()
    hits = [k for k, v in options.items() if v.lower() in low]
    return hits[0] if len(hits) == 1 else None


@torch.no_grad()
def generate(model, tok, texts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    from tqdm.auto import tqdm

    device = next(model.parameters()).device
    out = []
    for b in tqdm(range(0, len(texts), batch_size), leave=False):
        enc = tok(texts[b:b + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in out]


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--n-questions", type=int, default=50)
    ap.add_argument("--template", default="flat")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    beh = pd.read_csv(out / "behavior.csv")
    beh = beh[beh.template == args.template]
    kept = sorted(beh.qid.unique())
    sample = set(random.Random(cfg.seed).sample(kept, min(args.n_questions, len(kept))))
    qs = [q for q in load_questions(args.questions) if q.qid in sample]
    model, tok = load_model(cfg)

    rows = []
    for q in qs:
        for c in all_conditions():
            rows.append(dict(qid=q.qid, condition=c.name, role=c.role, correct=q.correct,
                             pushback=q.pushback, options=q.options,
                             text=build_prompt(tok, q, c, args.template, prefill=False).text))
    df = pd.DataFrame(rows)
    print(f"generating {len(df)} replies ({len(qs)} questions x 13 conditions, template={args.template})")
    df["reply"] = generate(model, tok, df.text.tolist(), cfg.batch_size, args.max_new_tokens)
    df["reply_letter"] = [parse_letter(r, o) for r, o in zip(df.reply, df.options)]
    df = df.merge(beh[["qid", "condition", "top"]].rename(columns={"top": "metric_letter"}),
                  on=["qid", "condition"])
    df["agree"] = df.reply_letter == df.metric_letter
    df["reply_flip"] = df.reply_letter == df.pushback
    df.drop(columns=["text", "options"]).to_csv(out / "freetext.csv", index=False)

    parsed = df.reply_letter.notna()
    print(f"  parsed a letter from {parsed.mean():.1%} of replies")
    print(f"  reply letter == letter-metric top letter: {df[parsed].agree.mean():.1%} of parsed replies")
    print("  flip rate, letter metric vs free text (parsed replies only):")
    t = df[parsed].assign(metric_flip=lambda d: d.metric_letter == d.pushback)
    print(t.groupby("role")[["metric_flip", "reply_flip"]].mean().round(3).to_string())
    print("\n  examples (first 2 per role):")
    for role in ("senior", "junior", "neutral"):
        for _, r in df[df.role == role].head(2).iterrows():
            print(f"  [{r.condition}] metric={r.metric_letter} reply={r.reply_letter} :: {r.reply[:150]!r}")
    dis = df[parsed & ~df.agree]
    if len(dis):
        print(f"\n  disagreements ({len(dis)}), first 5:")
        for _, r in dis.head(5).iterrows():
            print(f"  [{r.condition}] metric={r.metric_letter} reply={r.reply_letter} :: {r.reply[:150]!r}")
    print(f"\nwrote {out/'freetext.csv'}")


if __name__ == "__main__":
    main()
