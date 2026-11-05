"""Cache last-token activations for the behavior prompts and build the deference vectors.

    python -m vnpronouns.extract --config configs/local.yaml

Reads the (qid, condition, template) rows from behavior.csv, re-runs those prompts with
output_hidden_states, and keeps the residual stream at the final token (the ":" of the
prefilled "Đáp án:") at every hidden-state index.

Writes
  activations.pt   {"acts": fp16 [N, n_layers+1, hidden], "meta": DataFrame records}
  vectors.pt       v[i] = mean over matched train pairs of (senior - junior) at hidden index i,
                   plus per-pair vectors, a shuffled-label control, and the train/test qid split.
"""

from __future__ import annotations

import pandas as pd
import torch

from vnpronouns.common import base_args, config_from_args, prompt_table, split_qids
from vnpronouns.data import load_questions
from vnpronouns.model import load_model, score_prompts
from vnpronouns.prompts import PRONOUN_PAIRS

KEY = ["qid", "pair", "template"]


def matched_diffs(acts: torch.Tensor, meta: pd.DataFrame, qids: list[str],
                  pair: str | None = None, shuffle_seed: int | None = None) -> torch.Tensor:
    """[n_matched, n_layers+1, hidden] senior-minus-junior differences on matched twins."""
    m = meta[meta.qid.isin(qids) & meta.role.isin(["senior", "junior"])]
    if pair:
        m = m[m.pair == pair]
    sen = m[m.role == "senior"].set_index(KEY).row
    jun = m[m.role == "junior"].set_index(KEY).row
    idx = sen.index.intersection(jun.index)
    d = acts[torch.as_tensor(sen[idx].to_numpy().copy())].float() - \
        acts[torch.as_tensor(jun[idx].to_numpy().copy())].float()
    if shuffle_seed is not None:  # randomly swap senior/junior labels within each twin
        g = torch.Generator().manual_seed(shuffle_seed)
        sign = torch.randint(0, 2, (len(d),), generator=g).float() * 2 - 1
        d = d * sign[:, None, None]
    return d


def main() -> None:
    args = base_args(__doc__).parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    beh = pd.read_csv(out / "behavior.csv")
    model, tok = load_model(cfg)
    qs = [q for q in load_questions(args.questions) if q.qid in set(beh.qid)]
    table = prompt_table(tok, qs)
    table = table.merge(beh[["qid", "condition", "template"]], on=["qid", "condition", "template"])
    table = table.reset_index(drop=True)
    table["row"] = table.index
    print(f"caching activations for {len(table)} prompts")
    acts = score_prompts(model, tok, table.text.tolist(), cfg.batch_size, return_hidden=True,
                         progress=True).hidden
    meta = table.drop(columns="text")
    torch.save({"acts": acts, "meta": meta.to_dict("records")}, out / "activations.pt")
    print(f"  acts {tuple(acts.shape)} fp16 -> {out/'activations.pt'}")

    train, test = split_qids(meta.qid.tolist(), cfg.seed)
    d_train = matched_diffs(acts, meta, train)
    v = d_train.mean(0)
    vectors = {
        "v": v,
        "v_shuffled": matched_diffs(acts, meta, train, shuffle_seed=cfg.seed).mean(0),
        "per_pair": {p: matched_diffs(acts, meta, train, pair=p).mean(0) for p in PRONOUN_PAIRS},
        "train_qids": train, "test_qids": test, "n_train_pairs": len(d_train),
    }
    torch.save(vectors, out / "vectors.pt")
    print(f"  built vectors from {len(d_train)} matched train pairs ({len(train)} train / {len(test)} test questions)"
          f" -> {out/'vectors.pt'}")

    # quick look (the full sanity checks are Phase 5): held-out separation per layer
    d_test = matched_diffs(acts, meta, test)
    unit = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sep = ((d_test * unit).sum(-1) > 0).float().mean(0)  # fraction of test twins with senior > junior
    print("\n  hidden idx | ‖v‖     | held-out: senior projects above junior twin")
    for i in range(v.shape[0]):
        print(f"  {i:10d} | {v[i].norm():7.2f} | {sep[i]:.2f}")


if __name__ == "__main__":
    main()
