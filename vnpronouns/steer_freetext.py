"""Phase 7: free-text replies under steering / ablation (does it still write normal Vietnamese?).

    python -m vnpronouns.steer_freetext --config configs/colab.yaml [--n-questions 15]

For a seeded sample of FINAL-split questions x 13 conditions x the 3 pre-registered templates, the
model writes a greedy reply to the pushback (no "Đáp án:" prefill) under each setting:
  none | ablate v at all layers | ablate v in layers 10-19 | ablate a random direction at all layers
  | add: L18 lam +2 | add: L18 lam -2
Edits apply from the pushback turn on, including every generated token.

Per setting: reply stance (hold / flip / other / unclear, as in freetext.py) by role, the senior-junior
gap in reply flips, and simple health checks (share of replies with Vietnamese diacritics, empty
replies, distinct-word ratio to catch loops). Prints the same few prompts' replies under every
setting for reading by hand. Writes steer_freetext.csv.
"""

from __future__ import annotations

import random
import re

import pandas as pd
import torch

from vnpronouns.common import base_args, config_from_args
from vnpronouns.data import load_questions
from vnpronouns.freetext import generate, has_word, parse_answer, stance
from vnpronouns.model import Steering, char_to_token_index, decoder_layers, load_model
from vnpronouns.prompts import LETTERS, PRIMARY_TEMPLATES, all_conditions, build_prompt
from vnpronouns.steer import pick_vector

_VI = re.compile(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.IGNORECASE)


def settings(vectors: dict, n_layers: int, seed: int) -> dict:
    def abl(kind, layers):
        return [Steering(l, pick_vector(vectors, kind, l + 1, seed), 1.0, "ablate") for l in layers]
    return {
        "none": None,
        "ablate v, all layers": abl("deference", range(n_layers)),
        "ablate v, layers 10-19": abl("deference", range(10, 20)),
        "ablate random, all layers": abl("random", range(n_layers)),
        "add v, L18 lam +2": Steering(18, pick_vector(vectors, "deference", 19, seed), 2.0),
        "add v, L18 lam -2": Steering(18, pick_vector(vectors, "deference", 19, seed), -2.0),
    }


def health(reply: str) -> dict:
    words = reply.split()
    return dict(vietnamese=bool(_VI.search(reply)), empty=not reply.strip(),
                distinct=len(set(words)) / len(words) if words else 0.0, n_words=len(words))


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--n-questions", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--n-show", type=int, default=4, help="prompts whose replies are printed under every setting")
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    vectors = torch.load(out / "vectors.pt", weights_only=False)
    final = sorted(vectors["final_qids"])
    sample = set(random.Random(cfg.seed).sample(final, min(args.n_questions, len(final))))
    qs = [q for q in load_questions(args.questions) if q.qid in sample]
    model, tok = load_model(cfg)
    n_layers = len(decoder_layers(model))
    rows = []
    for q in qs:
        for c in all_conditions():
            for t in PRIMARY_TEMPLATES:
                p = build_prompt(tok, q, c, t, prefill=False)
                rows.append(dict(qid=q.qid, condition=c.name, role=c.role, template=t, user_pron=c.user_pron,
                                 correct=q.correct, pushback=q.pushback, question=q.question,
                                 **{f"opt_{l}": q.options[l] for l in LETTERS}, text=p.text,
                                 start=char_to_token_index(tok, p.text, p.pushback_char_start)))
    table = pd.DataFrame(rows)
    cache = out / ".cache"
    cache.mkdir(exist_ok=True)
    sets = settings(vectors, n_layers, cfg.seed)
    print(f"{len(table)} prompts x {len(sets)} settings ({len(qs)} final questions x 13 conditions x "
          f"{len(PRIMARY_TEMPLATES)} templates)")
    res = []
    for i, (name, st) in enumerate(sets.items()):
        print(f"[{i + 1}/{len(sets)}] {name}", flush=True)
        replies = generate(model, tok, table.text.tolist(), cfg.batch_size, args.max_new_tokens,
                           cache_path=cache / f"steer_ft_{i}.pt", steering=st,
                           steer_starts=table.start.tolist() if st is not None else None)
        d = table.drop(columns=["text", "start"]).assign(setting=name, reply=replies)
        res.append(d)
    df = pd.concat(res, ignore_index=True)
    opts = df[[f"opt_{l}" for l in LETTERS]].astype(str).to_numpy()
    df["reply_letter"] = [parse_answer(r, dict(zip(LETTERS, o))) for r, o in zip(df.reply, opts)]
    df["reply_stance"] = [stance(l, c, p) for l, c, p in zip(df.reply_letter, df.correct, df.pushback)]
    df["reply_flip"] = df.reply_stance == "flip"
    df["addr_user_pron"] = [has_word(r, u) for r, u in zip(df.reply, df.user_pron)]
    df = pd.concat([df, pd.DataFrame([health(r) for r in df.reply])], axis=1)
    df.to_csv(out / "steer_freetext.csv", index=False)

    order = list(sets)
    print("\nREPLY STANCE share by setting:")
    st = df.groupby("setting").reply_stance.value_counts(normalize=True).unstack().reindex(order)
    print(st.round(3).to_string())
    print("\nREPLY FLIP RATE by role (and senior - junior gap):")
    fr = df.pivot_table(index="setting", columns="role", values="reply_flip").reindex(order)
    fr["S-J"] = fr["senior"] - fr["junior"]
    print(fr[["senior", "neutral", "junior", "S-J"]].round(3).to_string())
    print("\nHEALTH: share with Vietnamese diacritics, empty, mean distinct-word ratio, mean words, "
          "addresses user with their pronoun:")
    print(df.groupby("setting")[["vietnamese", "empty", "distinct", "n_words", "addr_user_pron"]].mean()
          .reindex(order).round(3).to_string())
    print("\nSAME PROMPTS UNDER EVERY SETTING (for reading):")
    keys = (table[table.role.isin(["senior", "junior"])].sample(args.n_show, random_state=cfg.seed)
            [["qid", "condition", "template"]])
    for _, k in keys.iterrows():
        sub = df[(df.qid == k.qid) & (df.condition == k.condition) & (df.template == k.template)]
        r0 = sub.iloc[0]
        print(f"\n--- {k.qid} {k.condition} {k.template}: correct {r0.correct}) {r0[f'opt_{r0.correct}']} | "
              f"user claims {r0.pushback}) {r0[f'opt_{r0.pushback}']}")
        for _, r in sub.set_index("setting").reindex(order).iterrows():
            print(f"  [{r.name}] ({r.reply_stance}) {' '.join(str(r.reply).split())[:260]}")
    print(f"\nwrote {out / 'steer_freetext.csv'}")


if __name__ == "__main__":
    main()
