"""Phase 4 statistics on behavior.csv (no GPU needed).

    python -m vnpronouns.stats --config configs/colab_t4.yaml [--n-boot 10000]

Flip rates (and mean drop in P(correct)) per role, with 95% bootstrap CIs that resample
*questions* (all of a question's prompts move together), broken out by pronoun pair,
gender of the senior pronoun, and pushback template.

Paired tests (SPEC: senior vs junior, senior vs neutral, junior vs neutral):
  McNemar exact test on matched prompts, Bonferroni-corrected for the 3 comparisons.
    senior vs junior   twins matched on (qid, pair, template), roles swapped
    X vs neutral       each senior/junior prompt matched to the neutral prompt with the
                       same (qid, template); a neutral prompt is reused across the 6 pairs
  Because prompts from one question are not independent (and the neutral reuse above),
  each comparison also gets a question-resampling bootstrap CI and p-value for the
  difference in flip rate. Treat the bootstrap as the check that McNemar is not overstating.

Template analysis: the same tests within each pushback template, the template x role
interaction (does e.g. the senior-junior gap differ between soft and flat?), and the
senior-junior gap per template x pair.

Writes stats_rates.csv, stats_tests.csv, stats_template_tests.csv,
stats_template_interaction.csv and stats_template_by_pair.csv next to behavior.csv.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from vnpronouns.common import base_args, config_from_args

ROLES = ("senior", "junior", "neutral")
COMPARISONS = (("senior", "junior"), ("senior", "neutral"), ("junior", "neutral"))
GENDER = {"ong_chau": "male", "chu_chau": "male", "anh_em": "male",
          "ba_chau": "female", "co_chau": "female", "chi_em": "female"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                 for i in range(k + 1)]
    m = max(log_terms)
    tail = math.exp(m) * sum(math.exp(t - m) for t in log_terms)
    return min(1.0, 2 * tail)


def boot_weights(n_q: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n_q] multinomial counts: how many times each question appears in each resample."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_q, np.full(n_q, 1 / n_q), size=n_boot).astype(np.float64)


def _per_question(df: pd.DataFrame, qids: pd.Index, col: str) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby("qid")[col].agg(["sum", "count"]).reindex(qids, fill_value=0)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def rate_with_ci(df: pd.DataFrame, col: str, W: np.ndarray, qids: pd.Index) -> tuple[float, float, float]:
    s, n = _per_question(df, qids, col)
    point = s.sum() / n.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (W @ s) / (W @ n)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, lo, hi


def diff_boots(a: pd.DataFrame, b: pd.DataFrame, col: str, W: np.ndarray, qids: pd.Index):
    """Point estimate and bootstrap draws of rate(a) - rate(b). Shared W keeps draws comparable."""
    sa, na = _per_question(a, qids, col)
    sb, nb = _per_question(b, qids, col)
    point = sa.sum() / na.sum() - sb.sum() / nb.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (W @ sa) / (W @ na) - (W @ sb) / (W @ nb)
    return point, boots


def summarize_boots(point: float, boots: np.ndarray):
    """(point, 95% CI lo, hi, two-sided bootstrap p) from bootstrap draws."""
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return point, lo, hi, p


def diff_with_ci(a: pd.DataFrame, b: pd.DataFrame, col: str, W: np.ndarray, qids: pd.Index):
    """Difference of rates a - b, bootstrap CI and two-sided bootstrap p (resampling questions)."""
    return summarize_boots(*diff_boots(a, b, col, W, qids))


def matched_flips(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Rows of matched (flip_x, flip_y) prompt pairs for roles x vs y."""
    if y == "neutral":
        xs = df[df.role == x][["qid", "pair", "template", "flip"]]
        ny = df[df.role == "neutral"][["qid", "template", "flip"]]
        return xs.merge(ny, on=["qid", "template"], suffixes=("_x", "_y"))
    key = ["qid", "pair", "template"]
    return df[df.role == x][key + ["flip"]].merge(df[df.role == y][key + ["flip"]], on=key,
                                                    suffixes=("_x", "_y"))


def paired_tests(df: pd.DataFrame, W: np.ndarray, qids: pd.Index, scope: str = "all",
                 n_comparisons: int = len(COMPARISONS)) -> list[dict]:
    rows = []
    for x, y in COMPARISONS:
        m = matched_flips(df, x, y)
        b = int((m.flip_x & ~m.flip_y).sum())   # x flipped, y held
        c = int((~m.flip_x & m.flip_y).sum())   # y flipped, x held
        p = mcnemar_exact(b, c)
        d, lo, hi, p_boot = diff_with_ci(df[df.role == x], df[df.role == y], "flip", W, qids)
        rows.append(dict(scope=scope, comparison=f"{x} vs {y}", n_matched=len(m),
                         n_questions=m.qid.nunique(), flip_x=m.flip_x.mean(), flip_y=m.flip_y.mean(),
                         x_only=b, y_only=c, mcnemar_p=p, mcnemar_p_bonf=min(1.0, p * n_comparisons),
                         flip_diff=d, diff_ci_lo=lo, diff_ci_hi=hi, boot_p=p_boot))
    return rows


def template_analysis(df: pd.DataFrame, W: np.ndarray, qids: pd.Index):
    """Role effects within each pushback template, and whether they differ between templates.

    tests:     McNemar + bootstrap for each comparison within each template (Bonferroni x9).
    contrasts: interaction = (x - y difference in template t1) - (same in t2), bootstrap over
               questions with shared resamples (Bonferroni x9: 3 comparisons x 3 template pairs).
    by_pair:   senior - junior difference per template x pair, to see if a pattern is pair-wide.
    """
    templates = sorted(df.template.unique())
    n_tests = len(COMPARISONS) * len(templates)
    tests, draws = [], {}
    for t in templates:
        sub = df[df.template == t]
        for row in paired_tests(sub, W, qids, scope=t, n_comparisons=n_tests):
            tests.append(row)
        for x, y in COMPARISONS:
            draws[(x, y, t)] = diff_boots(sub[sub.role == x], sub[sub.role == y], "flip", W, qids)
    pairs_t = [(a, b) for i, a in enumerate(templates) for b in templates[i + 1:]]
    n_con = len(COMPARISONS) * len(pairs_t)
    contrasts = []
    for x, y in COMPARISONS:
        for t1, t2 in pairs_t:
            (p1, b1), (p2, b2) = draws[(x, y, t1)], draws[(x, y, t2)]
            point, lo, hi, p = summarize_boots(p1 - p2, b1 - b2)
            contrasts.append(dict(comparison=f"{x} vs {y}", contrast=f"{t1} - {t2}", diff_t1=p1, diff_t2=p2,
                                  interaction=point, ci_lo=lo, ci_hi=hi, boot_p=p,
                                  boot_p_bonf=min(1.0, p * n_con)))
    by_pair = []
    for (t, pair), sub in df[df.role != "neutral"].groupby(["template", "pair"]):
        point, lo, hi, p = diff_with_ci(sub[sub.role == "senior"], sub[sub.role == "junior"], "flip", W, qids)
        by_pair.append(dict(template=t, pair=pair, senior_minus_junior=point, ci_lo=lo, ci_hi=hi, boot_p=p))
    return pd.DataFrame(tests), pd.DataFrame(contrasts), pd.DataFrame(by_pair)


def rate_rows(df: pd.DataFrame, W: np.ndarray, qids: pd.Index, by: str | None) -> list[dict]:
    rows = []
    groups = [("all", df)] if by is None else list(df.groupby(by))
    for g, sub in groups:
        for role in ROLES:
            r = sub[sub.role == role]
            if r.empty:
                continue
            for metric in ("flip", "delta_p_correct"):
                point, lo, hi = rate_with_ci(r, metric, W, qids)
                rows.append(dict(breakdown=by or "overall", group=g, role=role, metric=metric,
                                 n_prompts=len(r), n_questions=r.qid.nunique(),
                                 value=point, ci_lo=lo, ci_hi=hi))
    return rows


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    df = pd.read_csv(out / "behavior.csv")
    df["flip"] = df.flip.astype(bool)
    df["gender"] = df.pair.map(GENDER).fillna("neutral")
    qids = pd.Index(sorted(df.qid.unique()))
    W = boot_weights(len(qids), args.n_boot, cfg.seed)

    rates = []
    for by in (None, "pair", "gender", "template"):
        sub = df if by in (None, "template") else df[df.role != "neutral"]
        rates += rate_rows(sub, W, qids, by)
    rates = pd.DataFrame(rates)
    rates.to_csv(out / "stats_rates.csv", index=False)

    tests = paired_tests(df, W, qids)
    for pair, sub in df[df.role != "neutral"].groupby("pair"):  # exploratory: uncorrected per pair
        sub = pd.concat([sub, df[df.role == "neutral"]])
        tests += [dict(t, mcnemar_p_bonf=float("nan")) for t in paired_tests(sub, W, qids, scope=pair)]
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "stats_tests.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"{len(df)} prompts, {len(qids)} questions, {args.n_boot} bootstrap resamples over questions\n")
    ov = rates[(rates.breakdown == "overall")]
    for metric, title in (("flip", "FLIP RATE"), ("delta_p_correct", "MEAN DROP IN P(correct)")):
        t = ov[ov.metric == metric].set_index("role")
        print(f"{title} by role [95% CI]")
        for role, r in t.iterrows():
            print(f"  {role:8s} {r.value:.3f}  [{r.ci_lo:.3f}, {r.ci_hi:.3f}]  (n={r.n_prompts})")
        print()
    print("PAIRED TESTS (McNemar exact, Bonferroni x3; bootstrap over questions as a check)")
    for _, t in tests[tests.scope == "all"].iterrows():
        print(f"  {t.comparison:18s} {t.flip_x:.3f} vs {t.flip_y:.3f}  discordant {t.x_only}/{t.y_only}  "
              f"McNemar p={t.mcnemar_p:.2g} (bonf {t.mcnemar_p_bonf:.2g})  "
              f"diff {t.flip_diff:+.3f} [{t.diff_ci_lo:+.3f}, {t.diff_ci_hi:+.3f}] boot p={t.boot_p:.2g}")
    for by in ("pair", "gender", "template"):
        t = rates[(rates.breakdown == by) & (rates.metric == "flip")]
        tab = t.assign(cell=t.apply(lambda r: f"{r.value:.3f} [{r.ci_lo:.2f},{r.ci_hi:.2f}]", axis=1)) \
               .pivot(index="group", columns="role", values="cell")
        print(f"\nFLIP RATE by {by} [95% CI]")
        print(tab.to_string())
    sj = tests[(tests.scope != "all") & (tests.comparison == "senior vs junior")]
    print("\nSENIOR vs JUNIOR by pair (exploratory, uncorrected)")
    for _, t in sj.iterrows():
        print(f"  {t.scope:9s} diff {t.flip_diff:+.3f} [{t.diff_ci_lo:+.3f}, {t.diff_ci_hi:+.3f}]  "
              f"discordant {t.x_only}/{t.y_only}  McNemar p={t.mcnemar_p:.2g}")

    tt, con, bp = template_analysis(df, W, qids)
    tt.to_csv(out / "stats_template_tests.csv", index=False)
    con.to_csv(out / "stats_template_interaction.csv", index=False)
    bp.to_csv(out / "stats_template_by_pair.csv", index=False)
    print("\nWITHIN-TEMPLATE PAIRED TESTS (McNemar exact, Bonferroni x9; bootstrap over questions)")
    for _, t in tt.iterrows():
        print(f"  {t.scope:8s} {t.comparison:18s} {t.flip_x:.3f} vs {t.flip_y:.3f}  discordant {t.x_only}/{t.y_only}  "
              f"McNemar bonf p={t.mcnemar_p_bonf:.2g}  diff {t.flip_diff:+.3f} [{t.diff_ci_lo:+.3f}, {t.diff_ci_hi:+.3f}] "
              f"boot p={t.boot_p:.2g}")
    print("\nTEMPLATE x ROLE INTERACTION: does a role difference change between templates? (Bonferroni x9)")
    for _, c in con.iterrows():
        print(f"  {c.comparison:18s} {c.contrast:16s} {c.diff_t1:+.3f} vs {c.diff_t2:+.3f} -> "
              f"{c.interaction:+.3f} [{c.ci_lo:+.3f}, {c.ci_hi:+.3f}]  bonf p={c.boot_p_bonf:.2g}")
    print("\nSENIOR - JUNIOR flip difference by template x pair [95% CI]  (is a template's pattern pair-wide?)")
    tab = bp.assign(cell=bp.apply(lambda r: f"{r.senior_minus_junior:+.3f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}]", axis=1)) \
            .pivot(index="pair", columns="template", values="cell")
    print(tab.to_string())
    print(f"\nwrote stats_rates.csv, stats_tests.csv, stats_template_tests.csv, "
          f"stats_template_interaction.csv, stats_template_by_pair.csv in {out}")


if __name__ == "__main__":
    main()
