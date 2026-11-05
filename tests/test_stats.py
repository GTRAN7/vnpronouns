import math

import pandas as pd

from vnpronouns.stats import boot_weights, diff_with_ci, matched_flips, mcnemar_exact, rate_with_ci


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    # b=0, c=6: two-sided = 2 * 0.5**6
    assert math.isclose(mcnemar_exact(0, 6), 2 * 0.5 ** 6)
    assert math.isclose(mcnemar_exact(2, 10), mcnemar_exact(10, 2))
    # large n stays finite and tiny
    assert 0 < mcnemar_exact(400, 600) < 1e-8


def _toy():
    rows = []
    for q in range(40):
        for t in ("flat", "soft"):
            rows.append(dict(qid=f"q{q}", pair="neutral", role="neutral", template=t, flip=q % 4 == 0))
            for pair in ("ong_chau", "anh_em"):
                rows.append(dict(qid=f"q{q}", pair=pair, role="senior", template=t, flip=q % 2 == 0))
                rows.append(dict(qid=f"q{q}", pair=pair, role="junior", template=t, flip=q % 4 == 0))
    return pd.DataFrame(rows)


def test_matching_and_rates():
    df = _toy()
    m = matched_flips(df, "senior", "junior")
    assert len(m) == 40 * 2 * 2
    mn = matched_flips(df, "senior", "neutral")
    assert len(mn) == 40 * 2 * 2  # neutral reused across pairs
    qids = pd.Index(sorted(df.qid.unique()))
    W = boot_weights(len(qids), 2000, 0)
    p, lo, hi = rate_with_ci(df[df.role == "senior"], "flip", W, qids)
    assert p == 0.5 and lo < 0.5 < hi
    d, lo, hi, bp = diff_with_ci(df[df.role == "senior"], df[df.role == "junior"], "flip", W, qids)
    assert d == 0.25 and lo > 0 and bp < 0.05
    d, lo, hi, bp = diff_with_ci(df[df.role == "junior"], df[df.role == "neutral"], "flip", W, qids)
    assert d == 0 and bp == 1.0
