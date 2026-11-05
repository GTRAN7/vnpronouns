"""Phase 6 steering: h <- h - lam * v at one decoder layer, from the pushback turn onward.

    python -m vnpronouns.steer --config configs/colab.yaml --split tune --layers 8 10 12 \
        --lams -2 -1 1 2 --vectors deference random [--first-turn]

Positive lam removes deference (senior looks more junior), negative adds it. lam is in units of
v, the mean senior-minus-junior gap at that layer, so lam = 1 moves a senior prompt by about one
senior->junior gap.

Splits come from vectors.pt (fixed in Phase 5): tune (Phase 6 layer choice), final (reported),
test (= tune + final), train.
Vectors: deference = v (pooled); random = fixed random direction, rescaled to |v| at that layer;
shuffled = shuffled-label vector, also rescaled to |v| (so every control has the same length).
--first-turn scores the first-turn prompts instead (H3: does steering hurt plain accuracy?),
steering from the first user turn onward.

Each (vector, layer, lam) is one resumable unit: run_dir/steer/<split>[_first]/<vector>_L<l>_lam<x>.csv
(lam = 0 is computed once as none_L-1_lam+0.csv). Existing units are skipped.
"""

from __future__ import annotations

import pandas as pd
import torch

from vnpronouns.common import base_args, config_from_args, prompt_table
from vnpronouns.data import load_questions
from vnpronouns.model import Steering, char_to_token_index, decoder_layers, load_model, score_prompts
from vnpronouns.prompts import all_conditions, build_first_turn_prompt
from vnpronouns.run_behavior import add_scores


def pick_vector(vectors: dict, kind: str, hidden_idx: int, seed: int) -> torch.Tensor:
    v = vectors["v"][hidden_idx].float()
    if kind == "deference":
        return v
    if kind == "shuffled":
        s = vectors["v_shuffled"][hidden_idx].float()
        return s / s.norm().clamp(min=1e-8) * v.norm()
    if kind == "random":
        r = torch.randn(v.shape, generator=torch.Generator().manual_seed(seed + hidden_idx))
        return r / r.norm() * v.norm()
    raise ValueError(kind)


def unit_name(kind: str, layer: int, lam: float) -> str:
    return "none_L-1_lam+0.csv" if lam == 0 else f"{kind}_L{layer}_lam{lam:+g}.csv"


def first_turn_table(tok, qs) -> pd.DataFrame:
    rows = []
    for q in qs:
        for c in all_conditions():
            text = build_first_turn_prompt(tok, q, c)
            start = char_to_token_index(tok, text, text.index("<|im_start|>user"))
            rows.append(dict(qid=q.qid, condition=c.name, pair=c.pair, role=c.role, template="first_turn",
                             correct=q.correct, pushback=q.pushback, text=text, pushback_tok=start))
    return pd.DataFrame(rows)


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--split", default="tune", choices=["tune", "final", "test", "train"])
    ap.add_argument("--layers", type=int, nargs="+", required=True, help="decoder layer indices (0-based)")
    ap.add_argument("--lams", type=float, nargs="+", default=[-2, -1, -0.5, 0.5, 1, 2])
    ap.add_argument("--vectors", nargs="+", default=["deference"], choices=["deference", "random", "shuffled"])
    ap.add_argument("--first-turn", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    vectors = torch.load(out / "vectors.pt", weights_only=False)
    beh = pd.read_csv(out / "behavior.csv", usecols=["qid", "template"])
    qids = set(vectors[f"{args.split}_qids"])
    unit_dir = out / "steer" / (args.split + ("_first" if args.first_turn else ""))
    unit_dir.mkdir(parents=True, exist_ok=True)

    lams = sorted(set(args.lams) | {0.0})
    todo = [(k, l, x) for x in lams for l in (args.layers if x else [-1]) for k in (args.vectors if x else ["none"])]
    todo = [u for u in todo if not (unit_dir / unit_name(*u)).exists()]
    print(f"{args.split}{' first-turn' if args.first_turn else ''}: {len(qids)} questions; "
          f"{len(todo)} units to run in {unit_dir}")
    if not todo:
        return
    model, tok = load_model(cfg)
    n_layers = len(decoder_layers(model))
    if not all(0 <= l < n_layers for l in args.layers):
        raise SystemExit(f"--layers must be in [0, {n_layers - 1}]")
    qs = [q for q in load_questions(args.questions) if q.qid in qids]
    table = (first_turn_table(tok, qs) if args.first_turn
             else prompt_table(tok, qs, templates=sorted(beh.template.unique())))
    print(f"  {len(table)} prompts per unit")

    for i, (kind, layer, lam) in enumerate(todo):
        st = None if lam == 0 else Steering(layer, pick_vector(vectors, kind, layer + 1, cfg.seed), lam)
        res = score_prompts(model, tok, table.text.tolist(), cfg.batch_size, steering=st,
                            steer_starts=table.pushback_tok.tolist(), sort_by_length=True)
        df = add_scores(table.drop(columns=["text", "pushback_tok"]), res.probs)
        df["flip"] = df.top == df.pushback
        df["p_push"] = [r[f"p_{r.pushback}"] for _, r in df.iterrows()]
        df["vector"], df["layer"], df["lam"] = kind, layer, lam
        df.to_csv(unit_dir / unit_name(kind, layer, lam), index=False)
        f = df.groupby("role").flip.mean()
        print(f"  [{i + 1}/{len(todo)}] {kind:9s} L{layer:<3d} lam {lam:+5.2g}  flip S {f.get('senior', float('nan')):.3f} "
              f"N {f.get('neutral', float('nan')):.3f} J {f.get('junior', float('nan')):.3f}  "
              f"P(A-D) {df.p_letters.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
