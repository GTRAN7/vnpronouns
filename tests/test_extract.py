import numpy as np
import pandas as pd
import torch

from vnpronouns.extract import between, cos, flip_corr, mean_diff, project, separation, twin_index
from vnpronouns.prompts import PRONOUN_PAIRS


def _synthetic(seed=0, n_q=20, layers=3, d=16):
    """senior = base + u, junior = base - u, neutral = base, with noise; u is the planted rank direction."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(layers, d, generator=g)
    rows, acts = [], []
    for q in range(n_q):
        for t in ("flat", "soft"):
            base = torch.randn(layers, d, generator=g) * 3
            rows.append(dict(qid=f"q{q}", pair="neutral", role="neutral", template=t, p_push=0.5))
            acts.append(base + 0.1 * torch.randn(layers, d, generator=g))
            for p in PRONOUN_PAIRS:
                for role, sgn in (("senior", 1), ("junior", -1)):
                    rows.append(dict(qid=f"q{q}", pair=p, role=role, template=t, p_push=0.5 + 0.2 * sgn))
                    acts.append(base + sgn * u + 0.1 * torch.randn(layers, d, generator=g))
    meta = pd.DataFrame(rows)
    meta["row"] = meta.index
    return torch.stack(acts).half(), meta, u


def test_planted_direction_is_recovered():
    acts, meta, u = _synthetic()
    qids = sorted(meta.qid.unique())
    train, test = qids[:10], qids[10:]
    s, j = twin_index(meta, train)
    assert len(s) == 10 * 2 * len(PRONOUN_PAIRS)
    v = mean_diff(acts, s, j)
    assert (cos(v, u) > 0.99).all()                      # v ≈ 2u
    st, jt = twin_index(meta, test)
    proj = project(acts, v)
    sep, dp = separation(proj, st, jt)
    assert (sep == 1).all() and (dp > 3).all()
    assert (between(proj, meta, st, jt) > 0.95).all()
    test_rows = meta[meta.qid.isin(test)].row.to_numpy()
    assert (flip_corr(proj, meta, test_rows) > 0.9).all()
    # shuffled labels: the planted direction mostly cancels
    vs = mean_diff(acts, s, j, shuffle_seed=0)
    assert (vs.norm(dim=-1) < 0.5 * v.norm(dim=-1)).all()
    # mask: one pair only
    sp, jp = twin_index(meta, train, meta.pair == "anh_em")
    assert len(sp) == 10 * 2 and set(meta.loc[sp].pair) == {"anh_em"}
