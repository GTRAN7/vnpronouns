"""Summarize steering units (CPU only).

    python -m vnpronouns.steer_report --config configs/colab.yaml --split tune    # -> picks the layer
    python -m vnpronouns.steer_report --config configs/colab.yaml --split final   # -> reported results

For every (vector, layer, lam) unit vs the unsteered lam = 0 unit, per role:
  flip rate, mean P(pushback letter), and the paired change vs lam = 0 (same prompts), with a
  95% bootstrap CI over questions, plus McNemar on the flips.

--split tune: layer selection (SPEC: biggest effect on the tuning set), on the continuous
  P(pushback) to keep the bf16 flip noise out of the choice:
    effect(L) = [drop in senior P(push) at lam=+1] + [rise in junior and in neutral P(push) at lam=-1]
    score(L)  = effect(L, deference) - effect(L, random)        (specific to v, not any push of that size)
  Writes steer/selected.json with the best layer.
--split final: the H2 runs and controls at the chosen layer, the per-template prediction test
  (if v drives the behavior, removing it should move senior toward junior in every template, so the
  steering change should track each template's own senior-junior gap), and H3 from the
  first-turn units (accuracy and P(A-D) mass under steering).
Writes steer_summary_<split>.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vnpronouns.common import base_args, config_from_args
from vnpronouns.stats import boot_weights, mcnemar_exact, rate_with_ci

KEY = ["qid", "condition", "template"]
ROLES = ("senior", "neutral", "junior")


def load_units(unit_dir) -> pd.DataFrame:
    files = sorted(unit_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no steering units in {unit_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired(df: pd.DataFrame, base: pd.DataFrame, W, qids, metric_cols=("flip", "p_push")) -> list[dict]:
    """Per (vector, layer, lam, role): level and change vs base, with bootstrap CIs over questions."""
    rows = []
    b = base[KEY + list(metric_cols)].rename(columns={m: f"{m}_0" for m in metric_cols})
    for (vec, layer, lam), u in df.groupby(["vector", "layer", "lam"]):
        u = u.merge(b, on=KEY, how="left")
        for role in ROLES:
            r = u[u.role == role]
            if r.empty:
                continue
            row = dict(vector=vec, layer=layer, lam=lam, role=role, n=len(r))
            for m in metric_cols:
                r = r.assign(**{f"d_{m}": r[m].astype(float) - r[f"{m}_0"].astype(float)})
                pt, lo, hi = rate_with_ci(r, f"d_{m}", W, qids)
                row.update({m: r[m].astype(float).mean(), f"{m}_0": r[f"{m}_0"].astype(float).mean(),
                            f"d_{m}": pt, f"d_{m}_lo": lo, f"d_{m}_hi": hi})
            if "flip" in metric_cols:
                fx, f0 = r.flip.astype(bool), r.flip_0.astype(bool)
                row["mcnemar_p"] = mcnemar_exact(int((fx & ~f0).sum()), int((~fx & f0).sum()))
            rows.append(row)
    return rows


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--split", default="tune", choices=["tune", "final", "test", "train"])
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    sdir = out / "steer"
    df = load_units(sdir / args.split)
    base = df[df.lam == 0]
    steered = df[df.lam != 0]
    qids = pd.Index(sorted(df.qid.unique()))
    W = boot_weights(len(qids), args.n_boot, cfg.seed)
    summ = pd.DataFrame(paired(steered, base, W, qids))
    summ.to_csv(out / f"steer_summary_{args.split}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{args.split}: {len(qids)} questions, {base.shape[0]} prompts per unit, "
          f"{steered.groupby(['vector', 'layer', 'lam']).ngroups} steered units")
    f0 = base.groupby("role")[["flip", "p_push"]].mean()
    print("unsteered (lam 0): " + "  ".join(f"{r} flip {f0.loc[r, 'flip']:.3f} P(push) {f0.loc[r, 'p_push']:.3f}"
                                            for r in ROLES if r in f0.index))

    def cell(vec, layer, lam, role, m="d_p_push"):
        s = summ[(summ.vector == vec) & (summ.layer == layer) & (summ.lam == lam) & (summ.role == role)]
        return float(s[m].iloc[0]) if len(s) else float("nan")

    if args.split == "tune":
        rows = []
        for layer in sorted(summ.layer.unique()):
            r = {"layer": layer}
            for vec in ("deference", "random"):
                e = (-cell(vec, layer, 1, "senior") + cell(vec, layer, -1, "junior") + cell(vec, layer, -1, "neutral"))
                r.update({f"{vec}_S+1": cell(vec, layer, 1, "senior"), f"{vec}_J-1": cell(vec, layer, -1, "junior"),
                          f"{vec}_N-1": cell(vec, layer, -1, "neutral"), f"effect_{vec}": e})
            r["score"] = r["effect_deference"] - r["effect_random"]
            rows.append(r)
        sel = pd.DataFrame(rows)
        print("\nLAYER SELECTION on the tuning set: change in mean P(pushback letter) vs lam 0")
        print("  (S+1 = senior at lam=+1, should fall; J-1 / N-1 = junior / neutral at lam=-1, should rise)")
        print(sel.round(4).to_string(index=False))
        if sel.score.notna().any():
            best = sel.loc[sel.score.idxmax()]
            choice = {"layer": int(best.layer), "score": float(best.score),
                      "effect_deference": float(best.effect_deference), "effect_random": float(best.effect_random)}
        else:
            choice = {"layer": int(sel.layer.iloc[0]), "score": None, "note": "no random units; first layer"}
        (sdir / "selected.json").write_text(json.dumps(choice, indent=1))
        print(f"\nselected: {choice}  -> {sdir / 'selected.json'}")
        print("\nFLIP RATE by lam (deference), per layer x role:")
        piv = summ[summ.vector == "deference"].pivot_table(index=["layer", "lam"], columns="role", values="flip")
        print(piv[[r for r in ROLES if r in piv.columns]].round(3).to_string())
        return

    # ---- final / test: H2 runs and controls ----
    for vec in sorted(summ.vector.unique()):
        s = summ[summ.vector == vec].sort_values(["layer", "lam", "role"])
        print(f"\n{vec.upper()} vector: change vs lam 0 [95% CI over questions]")
        for _, r in s.iterrows():
            print(f"  L{int(r.layer)} lam {r.lam:+5.2g} {r.role:8s} flip {r.flip:.3f} ({r.d_flip:+.3f} "
                  f"[{r.d_flip_lo:+.3f},{r.d_flip_hi:+.3f}], McNemar p={r.mcnemar_p:.2g})   "
                  f"P(push) {r.p_push:.3f} ({r.d_p_push:+.3f} [{r.d_p_push_lo:+.3f},{r.d_p_push_hi:+.3f}])")
    print("\nFLIP RATE vs lam (the main figure's numbers), role x vector:")
    allr = pd.concat([steered, base.assign(vector="deference"), base.assign(vector="random"),
                      base.assign(vector="shuffled")])
    allr = allr[allr.vector.isin(summ.vector.unique())]
    print(allr.pivot_table(index="lam", columns=["vector", "role"], values="flip").round(3).to_string())

    # per-template prediction test
    print("\nPER-TEMPLATE PREDICTION: does removing v (lam=+1) move senior toward junior in each template?")
    t0 = base.pivot_table(index="template", columns="role", values="flip")
    rows = []
    for vec in [v for v in ("deference", "random", "shuffled") if v in set(steered.vector)]:
        for lam in (1, 2):
            u = steered[(steered.vector == vec) & (steered.lam == lam)]
            if u.empty:
                continue
            t1 = u.pivot_table(index="template", columns="role", values="flip")
            gap = t0["senior"] - t0["junior"]
            dsen = t1["senior"] - t0["senior"]
            r = float(np.corrcoef(gap, -dsen)[0, 1]) if len(gap) > 2 else float("nan")
            rows.append(dict(vector=vec, lam=lam, corr_gap_vs_drop=r))
            if vec == "deference" and lam == 1:
                print(pd.DataFrame({"S-J gap (lam 0)": gap, "senior change at lam +1": dsen}).round(3).to_string())
    if rows:
        print("  correlation across templates of (senior-junior gap) with (drop in senior flips):")
        for r in rows:
            print(f"    {r['vector']:9s} lam +{r['lam']}: r = {r['corr_gap_vs_drop']:+.2f}")

    # H3: first-turn accuracy under steering
    ft_dir = sdir / f"{args.split}_first"
    if ft_dir.exists() and any(ft_dir.glob("*.csv")):
        ft = load_units(ft_dir)
        ft["acc"] = ft.top == ft.correct
        print("\nH3 FIRST-TURN ACCURACY / P(A-D) under steering (same questions, steering from the first user turn):")
        t = ft.groupby(["vector", "layer", "lam"]).agg(accuracy=("acc", "mean"), p_letters=("p_letters", "mean"),
                                                       n=("acc", "size"))
        print(t.round(4).to_string())
    print(f"\nwrote {out / f'steer_summary_{args.split}.csv'}")


if __name__ == "__main__":
    main()
