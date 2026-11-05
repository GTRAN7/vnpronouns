"""Steering run: h <- h - lam * v at one decoder layer, from the pushback turn onward.

    python -m vnpronouns.steer --config configs/local.yaml --layer 12 --lams -2 -1 0 1 2 \
        [--vector deference|random|shuffled] [--roles senior junior neutral]

Runs on the test-half questions from vectors.pt. Positive lam removes deference, negative adds it.
`random` is a fixed random direction rescaled to the deference vector's norm at that layer.
Writes steer_<vector>_L<layer>.csv with one row per prompt per lam.
"""

from __future__ import annotations

import pandas as pd
import torch

from vnpronouns.common import base_args, config_from_args, prompt_table
from vnpronouns.data import load_questions
from vnpronouns.model import Steering, decoder_layers, load_model, score_prompts
from vnpronouns.prompts import all_conditions
from vnpronouns.run_behavior import add_scores


def pick_vector(vectors: dict, kind: str, hidden_idx: int, seed: int) -> torch.Tensor:
    v = vectors["v"][hidden_idx].float()
    if kind == "deference":
        return v
    if kind == "shuffled":
        return vectors["v_shuffled"][hidden_idx].float()
    if kind == "random":
        r = torch.randn(v.shape, generator=torch.Generator().manual_seed(seed))
        return r / r.norm() * v.norm()
    raise ValueError(kind)


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--layer", type=int, required=True, help="decoder layer index (0-based)")
    ap.add_argument("--lams", type=float, nargs="+", default=[-2, -1, 0, 1, 2])
    ap.add_argument("--vector", default="deference", choices=["deference", "random", "shuffled"])
    ap.add_argument("--roles", nargs="+", default=["senior", "junior", "neutral"])
    ap.add_argument("--split", default="test", choices=["test", "train"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir

    vectors = torch.load(out / "vectors.pt", weights_only=False)
    model, tok = load_model(cfg)
    n_layers = len(decoder_layers(model))
    if not 0 <= args.layer < n_layers:
        raise SystemExit(f"--layer must be in [0, {n_layers - 1}]")
    vec = pick_vector(vectors, args.vector, args.layer + 1, cfg.seed)

    qids = set(vectors[f"{args.split}_qids"])
    qs = [q for q in load_questions(args.questions) if q.qid in qids]
    table = prompt_table(tok, qs, [c for c in all_conditions() if c.role in args.roles])
    print(f"steering layer {args.layer} with {args.vector} vector (‖v‖={vec.norm():.2f}) "
          f"on {len(qs)} {args.split} questions, {len(table)} prompts per lam")

    results = []
    for lam in args.lams:
        res = score_prompts(model, tok, table.text.tolist(), cfg.batch_size,
                            steering=Steering(args.layer, vec, lam),
                            steer_starts=table.pushback_tok.tolist(), progress=True)
        df = add_scores(table.drop(columns="text"), res.probs)
        df["flip"] = df.top == df.pushback
        df["lam"], df["layer"], df["vector"] = lam, args.layer, args.vector
        results.append(df)
    allres = pd.concat(results)
    path = out / f"steer_{args.vector}_L{args.layer}.csv"
    allres.to_csv(path, index=False)

    print("\nflip rate by lam x role:")
    print(allres.pivot_table(index="lam", columns="role", values="flip", aggfunc="mean").round(3).to_string())
    print("\nmean P(correct) / P(A..D) mass by lam:")
    print(allres.groupby("lam")[["p_correct", "p_letters"]].mean().round(3).to_string())
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
