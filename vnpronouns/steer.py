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

Phase 6b:
  --n-random K   adds random1..randomK (independent random directions, same length as v): a null
                 distribution for the steering effect instead of one random draw.
  --mode ablate  directional ablation h <- h - (h . u) u (u = unit vector) instead of adding lam * v;
                 --layers all ablates at every decoder layer; --positions all edits every token, not
                 just from the pushback turn. Units go to steer/<split>_ablate[_allpos]/ with lam = 1
                 as a placeholder; layer -2 means "all layers".
"""

from __future__ import annotations

import shutil

import pandas as pd
import torch

from vnpronouns.common import base_args, config_from_args, prompt_table
from vnpronouns.data import load_questions
from vnpronouns.model import Steering, char_to_token_index, decoder_layers, load_model, score_prompts
from vnpronouns.prompts import all_conditions, build_first_turn_prompt
from vnpronouns.run_behavior import add_scores


def pick_vector(vectors: dict, kind: str, hidden_idx: int, seed: int) -> torch.Tensor:
    v = vectors["v"][hidden_idx].float()
    if kind.endswith("_orth"):   # the control with its v component removed (ablation ignores sign, so overlap matters)
        c = pick_vector(vectors, kind[:-len("_orth")], hidden_idx, seed)
        u = v / v.norm().clamp(min=1e-8)
        c = c - (c @ u) * u
        return c / c.norm().clamp(min=1e-8) * v.norm()
    if kind == "deference":
        return v
    if kind == "shuffled":
        s = vectors["v_shuffled"][hidden_idx].float()
        return s / s.norm().clamp(min=1e-8) * v.norm()
    if kind.startswith("pairshuf"):   # shuffled-label vector built within one pronoun pair (Phase 5)
        s = list(vectors["pair_shuffled"].values())[int(kind[len("pairshuf"):])][hidden_idx].float()
        return s / s.norm().clamp(min=1e-8) * v.norm()
    if kind.startswith("random"):
        k = int(kind[len("random"):] or 0)   # random == random0 (the Phase 6 draw)
        r = torch.randn(v.shape, generator=torch.Generator().manual_seed(seed + hidden_idx + 7919 * k))
        return r / r.norm() * v.norm()
    raise ValueError(kind)


ALL_LAYERS = -2


def unit_name(kind: str, layer, lam: float) -> str:
    """layer: an int, ALL_LAYERS, or a range label "a-b" (ablation only)."""
    if lam == 0:
        return "none_L-1_lam+0.csv"
    return f"{kind}_L{'all' if layer == ALL_LAYERS else layer}_lam{lam:+g}.csv"


def layer_spec(spec: str):
    """'18' -> 18, 'all' -> ALL_LAYERS, '10-19' -> '10-19' (decoder layers 10..19 inclusive)."""
    if spec == "all":
        return ALL_LAYERS
    if "-" in spec:
        a, b = map(int, spec.split("-"))
        if not 0 <= a <= b:
            raise SystemExit(f"bad layer range {spec}")
        return f"{a}-{b}"
    return int(spec)


def layers_of(label, n_layers: int) -> list[int]:
    if label == ALL_LAYERS:
        return list(range(n_layers))
    if isinstance(label, str):
        a, b = map(int, label.split("-"))
        return list(range(a, b + 1))
    return [label]


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
    ap.add_argument("--layers", nargs="+", required=True, help="decoder layer indices (0-based), or 'all' (ablate)")
    ap.add_argument("--lams", type=float, nargs="+", default=[-2, -1, -0.5, 0.5, 1, 2])
    ap.add_argument("--vectors", nargs="+", default=["deference"], choices=["deference", "random", "shuffled"])
    ap.add_argument("--n-random", type=int, default=0, help="also run random1..randomK")
    ap.add_argument("--random-from", type=int, default=1, help="first K for --n-random (to extend a set)")
    ap.add_argument("--pair-shuffled", action="store_true", help="also run pairshuf0..5 (per-pair shuffled vectors)")
    ap.add_argument("--mode", default="add", choices=["add", "ablate"])
    ap.add_argument("--positions", default="pushback", choices=["pushback", "all"])
    ap.add_argument("--orth", action="store_true", help="use shuffled / pair-shuffled controls orthogonalized to v")
    ap.add_argument("--first-turn", action="store_true")
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    vectors = torch.load(out / "vectors.pt", weights_only=False)
    beh = pd.read_csv(out / "behavior.csv", usecols=["qid", "template"])
    qids = set(vectors[f"{args.split}_qids"])
    args.layers = [layer_spec(l) for l in args.layers]
    if args.mode == "add" and (args.positions != "pushback" or any(not isinstance(l, int) or l < 0 for l in args.layers)):
        raise SystemExit("--positions all / --layers all / layer ranges are only for --mode ablate")
    args.vectors = list(args.vectors) + [f"random{k}" for k in range(args.random_from, args.random_from + args.n_random)]
    if args.pair_shuffled:
        args.vectors += [f"pairshuf{k}" for k in range(len(vectors["pair_shuffled"]))]
    if args.orth:
        args.vectors = [k + "_orth" if k.startswith(("shuffled", "pairshuf")) else k for k in args.vectors]
    sub = args.split + ("_first" if args.first_turn else "")
    if args.mode == "ablate":
        sub += "_ablate" + ("_allpos" if args.positions == "all" else "")
        args.lams = [1.0]
    unit_dir = out / "steer" / sub
    unit_dir.mkdir(parents=True, exist_ok=True)
    base_src = out / "steer" / args.split / unit_name("none", -1, 0)
    if sub != args.split and not args.first_turn and base_src.exists() and not (unit_dir / base_src.name).exists():
        shutil.copy(base_src, unit_dir / base_src.name)   # the unsteered unit is the same prompts

    lams = sorted(set(args.lams) | {0.0})
    todo = [(k, l, x) for x in lams for l in (args.layers if x else [-1]) for k in (args.vectors if x else ["none"])]
    todo = [u for u in todo if not (unit_dir / unit_name(*u)).exists()]
    print(f"{args.split}{' first-turn' if args.first_turn else ''}: {len(qids)} questions; "
          f"{len(todo)} units to run in {unit_dir}")
    if not todo:
        return
    model, tok = load_model(cfg)
    n_layers = len(decoder_layers(model))
    if not all(0 <= l < n_layers for lab in args.layers for l in layers_of(lab, n_layers)):
        raise SystemExit(f"--layers must be in [0, {n_layers - 1}]")
    qs = [q for q in load_questions(args.questions) if q.qid in qids]
    table = (first_turn_table(tok, qs) if args.first_turn
             else prompt_table(tok, qs, templates=sorted(beh.template.unique())))
    print(f"  {len(table)} prompts per unit")

    for i, (kind, layer, lam) in enumerate(todo):
        if lam == 0:
            st = None
        else:
            st = [Steering(l, pick_vector(vectors, kind, l + 1, cfg.seed), lam, args.mode)
                  for l in layers_of(layer, n_layers)]
        starts = [0] * len(table) if args.positions == "all" else table.pushback_tok.tolist()
        res = score_prompts(model, tok, table.text.tolist(), cfg.batch_size, steering=st,
                            steer_starts=starts, sort_by_length=True)
        df = add_scores(table.drop(columns=["text", "pushback_tok"]), res.probs)
        df["flip"] = df.top == df.pushback
        df["p_push"] = [r[f"p_{r.pushback}"] for _, r in df.iterrows()]
        df["vector"], df["layer"], df["lam"] = kind, layer, lam
        df.to_csv(unit_dir / unit_name(kind, layer, lam), index=False)
        f = df.groupby("role").flip.mean()
        print(f"  [{i + 1}/{len(todo)}] {kind:9s} L{str(layer):<5s} {args.mode} lam {lam:+5.2g}  flip S {f.get('senior', float('nan')):.3f} "
              f"N {f.get('neutral', float('nan')):.3f} J {f.get('junior', float('nan')):.3f}  "
              f"P(A-D) {df.p_letters.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
