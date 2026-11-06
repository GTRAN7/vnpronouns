"""Summarize steering units (CPU only).

    python -m vnpronouns.steer_report --config configs/colab.yaml --split tune    # -> picks the layer
    python -m vnpronouns.steer_report --config configs/colab.yaml --split final   # -> reported results

For every (vector, layer, lam) unit vs the unsteered lam = 0 unit, per role:
  flip rate, mean P(pushback letter), and the paired change vs lam = 0 (same prompts), with a
  95% bootstrap CI over questions, plus McNemar on the flips.

--split tune: layer selection (SPEC: biggest effect on the tuning set), on the continuous
  P(pushback) to keep the bf16 flip noise out of the choice:
    effect(L) = [drop in senior P(push) at lam=+1] + [rise in junior and in neutral P(push) at lam=-1]
    score(L)  = effect(L, deference) - |effect(L, random)|      (specific to v, not any push of that size)
  Writes steer/selected.json with the best layer.
--split final: the H2 runs and controls at the chosen layer, the per-template prediction test
  (if v drives the behavior, removing it should move senior toward junior in every template, so the
  steering change should track each template's own senior-junior gap), and H3 from the
  first-turn units (accuracy and P(A-D) mass under steering).
Writes steer_summary_<split>.csv.

Phase 6b sections (--section; the default "main" is everything above):
  null    deference vs K independent random directions (random, random1..K) and shuffled, at each
          layer / lam they were run: is v's effect bigger than a same-length random push?
          Reports the empirical p = (1 + #{random with |effect| >= |v's|}) / (K + 1) for the shift
          in mean P(push) over roles and for the change in the senior-junior gap.
  ablate  directional-ablation units (steer/<split>_ablate*): per role change, and the change in
          the senior-junior / neutral-junior / senior-neutral flip gaps vs unsteered, with CIs.
          If v carries the rank information the model uses, ablating it should shrink S-J;
          ablating random directions should not. Writes steer_ablate_<split>.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from vnpronouns.common import base_args, config_from_args
from vnpronouns.stats import boot_weights, diff_boots, mcnemar_exact, rate_with_ci, summarize_boots

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
    ap.add_argument("--section", default="main", choices=["main", "null", "ablate"])
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    sdir = out / "steer"
    if args.section == "null":
        return null_report(sdir / args.split, args, cfg)
    if args.section == "ablate":
        return ablate_report(sdir, args, cfg, out)
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
            # a random direction's sign is arbitrary: one that happens to push the other way must
            # not inflate the score, so compare against its magnitude
            r["score"] = r["effect_deference"] - abs(r["effect_random"])
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


def null_report(unit_dir, args, cfg) -> pd.DataFrame:
    df = load_units(unit_dir)
    base, steered = df[df.lam == 0], df[df.lam != 0]
    qids = pd.Index(sorted(df.qid.unique()))
    W = boot_weights(len(qids), 200, cfg.seed)   # CIs are not used here, only point estimates
    summ = pd.DataFrame(paired(steered, base, W, qids))
    summ = summ[summ.vector.str.match(r"^(deference|shuffled|random\d*)$")]
    w = summ.pivot_table(index=["vector", "layer", "lam"], columns="role", values="d_p_push")
    wf = summ.pivot_table(index=["vector", "layer", "lam"], columns="role", values="d_flip")
    eff = pd.DataFrame({"shift": w[list(ROLES)].mean(axis=1), "gap_change": w["senior"] - w["junior"],
                        "flip_shift": wf[list(ROLES)].mean(axis=1), "flip_gap_change": wf["senior"] - wf["junior"]})
    eff = eff.reset_index()
    print(f"NULL over random directions ({args.split}, {len(qids)} questions). shift = mean change in P(push) over "
          f"roles; gap_change = change in senior - junior P(push); flip_* = same on flip rate.")
    rows = []
    for (layer, lam), g in eff.groupby(["layer", "lam"]):
        rnd = g[g.vector.str.startswith("random")]
        d = g[g.vector == "deference"]
        if d.empty or len(rnd) < 2:
            continue
        r = dict(layer=int(layer), lam=lam, n_random=len(rnd))
        for m in ("shift", "gap_change", "flip_shift", "flip_gap_change"):
            dv = float(d[m].iloc[0])
            rv = rnd[m].to_numpy()
            r.update({f"{m}_v": dv, f"{m}_rand_absmean": np.abs(rv).mean(), f"{m}_rand_absmax": np.abs(rv).max(),
                      f"{m}_p": (1 + (np.abs(rv) >= abs(dv)).sum()) / (len(rv) + 1)})
            sh = g[g.vector == "shuffled"]
            if len(sh):
                r[f"{m}_shuffled"] = float(sh[m].iloc[0])
        rows.append(r)
    res = pd.DataFrame(rows)
    for _, r in res.iterrows():
        print(f"\n  L{r.layer} lam {r.lam:+g}  ({r.n_random} random directions)")
        for m in ("shift", "gap_change", "flip_shift", "flip_gap_change"):
            sh = f"  shuffled {r[f'{m}_shuffled']:+.4f}" if f"{m}_shuffled" in r and pd.notna(r.get(f"{m}_shuffled")) else ""
            print(f"    {m:16s} v {r[f'{m}_v']:+.4f}   random |mean| {r[f'{m}_rand_absmean']:.4f} "
                  f"|max| {r[f'{m}_rand_absmax']:.4f}   p = {r[f'{m}_p']:.3f}{sh}")
    print("\n  per random direction (shift, gap_change):")
    print(eff.round(4).to_string(index=False))
    res.to_csv(cfg.run_dir / f"steer_null_{args.split}.csv", index=False)
    return res


def ablate_report(sdir, args, cfg, out) -> pd.DataFrame:
    dirs = sorted(d for d in sdir.glob(f"{args.split}_ablate*") if any(d.glob("*.csv")))
    if not dirs:
        raise SystemExit(f"no ablation units under {sdir}")
    W = qids = None
    rows = []
    gaps = (("senior", "junior"), ("neutral", "junior"), ("senior", "neutral"))
    for d in dirs:
        df = load_units(d)
        df["layer"] = df.layer.astype(str).str.replace(r"\.0$", "", regex=True).replace({"-2": "all"})
        base, steered = df[df.lam == 0], df[df.lam != 0]
        if W is None:
            qids = pd.Index(sorted(df.qid.unique()))
            W = boot_weights(len(qids), args.n_boot, cfg.seed)
        pos = "all tokens" if d.name.endswith("_allpos") else "from pushback"
        base_gap = {g: diff_boots(base[base.role == g[0]], base[base.role == g[1]], "flip", W, qids) for g in gaps}
        print(f"\nABLATION ({pos}; {d.name}): unsteered flip "
              + "  ".join(f"{r} {base[base.role == r].flip.mean():.3f}" for r in ROLES)
              + "  gaps " + "  ".join(f"{a[0].upper()}-{b[0].upper()} {base_gap[(a, b)][0]:+.3f}" for a, b in gaps))
        per_role = pd.DataFrame(paired(steered, base, W, qids))
        for (vec, layer), u in steered.groupby(["vector", "layer"]):
            r = dict(positions=pos, vector=vec, layer=layer)
            pr = per_role[(per_role.vector == vec) & (per_role.layer == layer)].set_index("role")
            for role in ROLES:
                r[f"flip_{role}"] = pr.loc[role, "flip"]
                r[f"d_flip_{role}"] = pr.loc[role, "d_flip"]
            r["p_letters"] = u.p_letters.mean()
            for a, b in gaps:
                pt, bt = diff_boots(u[u.role == a], u[u.role == b], "flip", W, qids)
                p0, b0 = base_gap[(a, b)]
                ch, lo, hi, p = summarize_boots(pt - p0, bt - b0)
                k = f"{a[0].upper()}{b[0].upper()}"
                r.update({f"gap_{k}": pt, f"d_gap_{k}": ch, f"d_gap_{k}_lo": lo, f"d_gap_{k}_hi": hi, f"d_gap_{k}_p": p})
            rows.append(r)
            print(f"  {vec:9s} L{r['layer']!s:4s} flip S {r['flip_senior']:.3f} ({r['d_flip_senior']:+.3f})  "
                  f"N {r['flip_neutral']:.3f} ({r['d_flip_neutral']:+.3f})  J {r['flip_junior']:.3f} ({r['d_flip_junior']:+.3f})"
                  f"  P(A-D) {r['p_letters']:.3f}")
            print("            " + "   ".join(
                f"{k} gap {r[f'gap_{k}']:+.3f} (change {r[f'd_gap_{k}']:+.3f} [{r[f'd_gap_{k}_lo']:+.3f},"
                f"{r[f'd_gap_{k}_hi']:+.3f}] p={r[f'd_gap_{k}_p']:.2g})" for k in ("SJ", "NJ", "SN")))
    res = pd.DataFrame(rows)
    print("\n  gap changes: deference vs controls (random = isotropic directions; shuffled = v_shuffled + the 6"
          " per-pair shuffled vectors, i.e. data-shaped directions with no consistent senior/junior sign)."
          "  p = (1 + #controls with |change| >= |v's|) / (n + 1)")
    for (pos, layer), g in res.groupby(["positions", "layer"], sort=False):
        d = g[g.vector == "deference"]
        if d.empty:
            continue
        for k in ("SJ", "NJ", "SN"):
            dv = float(d[f"d_gap_{k}"].iloc[0])
            msg = f"    {pos:13s} L{layer!s:6s} {k} deference {dv:+.3f}"
            for name, sel in (("random", g.vector.str.startswith("random")),
                              ("shuffled", g.vector.str.match(r"^(shuffled|pairshuf\d+)$"))):
                cv = g[sel][f"d_gap_{k}"].to_numpy()
                if len(cv):
                    msg += (f"   {name} ({len(cv)}): range [{cv.min():+.3f},{cv.max():+.3f}]"
                            f" p = {(1 + (np.abs(cv) >= abs(dv)).sum()) / (len(cv) + 1):.3f}")
            print(msg)
    res.to_csv(out / f"steer_ablate_{args.split}.csv", index=False)
    print(f"\nwrote {out / f'steer_ablate_{args.split}.csv'}")
    return res


if __name__ == "__main__":
    main()
