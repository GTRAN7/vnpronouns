"""Phase 5: cache last-token activations, build deference vectors, run the sanity checks.

    python -m vnpronouns.extract --config configs/colab.yaml [--acts-path /content/acts/activations.pt]

Re-runs every (qid, condition, template) prompt in behavior.csv with output_hidden_states and
keeps the residual stream at the final token (the ":" of the prefilled "Đáp án:") at every
hidden-state index (0 = embeddings, i = output of decoder layer i-1).

Split (fixed here so Phase 6 reuses it): 50/50 train/test by question; ~20% of the test
questions are carved out as the Phase 6 tuning set (`tune_qids`), the rest is `final_qids`.

Vectors (all = mean over matched TRAIN twins of senior - junior, per hidden index, float32):
  v            pooled over all templates and pairs  (the main deference vector)
  v_primary    pre-registered templates only
  v_shuffled   senior/junior labels randomly swapped within each twin (control)
  family/<f>, template/<t>, pair/<p>, and pair_shuffled/<p>

Sanity checks on TEST twins, per hidden index (SPEC):
  1. sep        held-out separation: share of twins whose senior projects above its junior twin
     dprime     mean(senior - junior projection) / std   (effect size)
  2. between    share of (qid, pair, template) where neutral lands strictly between the twins
                (chance = 1/3 for an unrelated direction)
  3. pair_cos   mean cosine between the 6 per-pair vectors (1 = same direction);
     cos_oc_ae  cos(ông/cháu, anh/em): pairs with no words in common
     loo_sep    separation on each pair's test twins using a vector built from the OTHER 5 pairs
  4. the same numbers for v_shuffled (should be ~0.5 / ~1/3 / ~0) and a random direction
  extra: flip_corr  correlation between projection and P(pushback letter) on test prompts,
                    after removing each (question, template)'s mean -> does the direction track caving?

Writes (run_dir): vectors.pt, projections.pt (every prompt's projection on unit v at every
layer), sanity_layers.csv, sanity_pair_cosine.csv; activations to --acts-path.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vnpronouns.common import base_args, config_from_args, prompt_table, split_qids
from vnpronouns.data import load_questions
from vnpronouns.model import load_model, score_prompts
from vnpronouns.prompts import PRIMARY_TEMPLATES, PRONOUN_PAIRS, TEMPLATE_FAMILY

KEY = ["qid", "pair", "template"]


def twin_index(meta: pd.DataFrame, qids, mask: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of matched (senior, junior) twins among `qids` (optionally within `mask`)."""
    m = meta[meta.qid.isin(set(qids)) & meta.role.isin(["senior", "junior"])]
    if mask is not None:
        m = m[mask.loc[m.index]]
    sen = m[m.role == "senior"].set_index(KEY).row
    jun = m[m.role == "junior"].set_index(KEY).row
    idx = sen.index.intersection(jun.index)
    return sen[idx].to_numpy().copy(), jun[idx].to_numpy().copy()


def mean_diff(acts: torch.Tensor, s: np.ndarray, j: np.ndarray, shuffle_seed: int | None = None,
              chunk: int = 2048) -> torch.Tensor:
    """[n_layers+1, hidden] float32 mean of acts[s] - acts[j] (chunked to keep memory low)."""
    total = torch.zeros(acts.shape[1:], dtype=torch.float64)
    sign = None
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        sign = torch.randint(0, 2, (len(s),), generator=g).double() * 2 - 1
    for a in range(0, len(s), chunk):
        d = acts[torch.as_tensor(s[a:a + chunk])].double() - acts[torch.as_tensor(j[a:a + chunk])].double()
        if sign is not None:
            d = d * sign[a:a + chunk, None, None]
        total += d.sum(0)
    return (total / max(len(s), 1)).float()


def project(acts: torch.Tensor, v: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """[N, n_layers+1] projection of every prompt onto unit v at each hidden index."""
    unit = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    out = []
    for a in range(0, len(acts), chunk):
        out.append((acts[a:a + chunk].float() * unit).sum(-1))
    return torch.cat(out)


def separation(proj: torch.Tensor, s: np.ndarray, j: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    d = proj[torch.as_tensor(s)] - proj[torch.as_tensor(j)]
    return (d > 0).float().mean(0), d.mean(0) / d.std(0).clamp(min=1e-8)


def between(proj: torch.Tensor, meta: pd.DataFrame, s: np.ndarray, j: np.ndarray) -> torch.Tensor:
    neutral = meta[meta.role == "neutral"].set_index(["qid", "template"]).row
    keys = pd.MultiIndex.from_frame(meta.loc[s, ["qid", "template"]])
    n = neutral.reindex(keys).to_numpy()
    ok = ~np.isnan(n)
    ps, pj, pn = proj[torch.as_tensor(s[ok])], proj[torch.as_tensor(j[ok])], proj[torch.as_tensor(n[ok].astype(int))]
    return ((pn > pj) & (pn < ps)).float().mean(0)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-8)


def flip_corr(proj: torch.Tensor, meta: pd.DataFrame, rows: np.ndarray) -> torch.Tensor:
    """Correlation of projection with P(pushback letter), both demeaned within (qid, template)."""
    m = meta.loc[rows]
    groups = torch.as_tensor(pd.factorize(m.qid + "|" + m.template)[0])
    p = torch.as_tensor(m.p_push.to_numpy(), dtype=torch.float32)
    x = proj[torch.as_tensor(rows)]
    n_g = int(groups.max()) + 1
    cnt = torch.zeros(n_g).index_add_(0, groups, torch.ones(len(groups)))
    p = p - (torch.zeros(n_g).index_add_(0, groups, p) / cnt)[groups]
    x = x - (torch.zeros(n_g, x.shape[1]).index_add_(0, groups, x) / cnt[:, None])[groups]
    return (x * p[:, None]).sum(0) / (x.norm(dim=0) * p.norm()).clamp(min=1e-8)


def main() -> None:
    ap = base_args(__doc__)
    ap.add_argument("--acts-path", default=None, help="where to save activations.pt (default: run_dir)")
    ap.add_argument("--tune-frac", type=float, default=0.2, help="share of test questions held for Phase 6 tuning")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()
    cfg = config_from_args(args)
    out = cfg.run_dir
    acts_path = Path(args.acts_path) if args.acts_path else out / "activations.pt"
    acts_path.parent.mkdir(parents=True, exist_ok=True)

    beh = pd.read_csv(out / "behavior.csv")
    model, tok = load_model(cfg)
    qs = [q for q in load_questions(args.questions) if q.qid in set(beh.qid)]
    table = prompt_table(tok, qs, templates=sorted(beh.template.unique()))
    cols = ["qid", "condition", "template", "flip", "p_A", "p_B", "p_C", "p_D"]
    table = table.merge(beh[cols], on=["qid", "condition", "template"]).reset_index(drop=True)
    table["row"] = table.index
    table["p_push"] = [r[f"p_{r.pushback}"] for _, r in table.iterrows()]
    table["family"] = table.template.map(TEMPLATE_FAMILY)
    n_hidden = model.config.num_hidden_layers + 1
    acts = torch.empty(len(table), n_hidden, model.config.hidden_size, dtype=torch.float16)
    print(f"caching activations for {len(table)} prompts -> {tuple(acts.shape)} fp16 "
          f"({acts.numel() * 2 / 2**30:.1f} GB)")
    texts = table.text.tolist()
    for a in range(0, len(texts), args.chunk):
        print(f"  chunk {a // args.chunk + 1}/{(len(texts) + args.chunk - 1) // args.chunk}", flush=True)
        acts[a:a + args.chunk] = score_prompts(model, tok, texts[a:a + args.chunk], cfg.batch_size,
                                               return_hidden=True, progress=True, sort_by_length=True).hidden
    del model
    meta = table.drop(columns="text")
    torch.save({"acts": acts, "meta": meta.to_dict("records")}, acts_path)
    print(f"  saved activations -> {acts_path}")

    train, test = split_qids(meta.qid.tolist(), cfg.seed)
    shuffled_test = test[:]
    random.Random(cfg.seed + 1).shuffle(shuffled_test)
    n_tune = round(len(test) * args.tune_frac)
    tune, final = sorted(shuffled_test[:n_tune]), sorted(shuffled_test[n_tune:])

    s_tr, j_tr = twin_index(meta, train)
    v = mean_diff(acts, s_tr, j_tr)
    vectors = {"v": v, "v_shuffled": mean_diff(acts, s_tr, j_tr, shuffle_seed=cfg.seed),
               "v_primary": mean_diff(acts, *twin_index(meta, train, meta.template.isin(PRIMARY_TEMPLATES))),
               "family": {}, "template": {}, "pair": {}, "pair_shuffled": {},
               "train_qids": train, "test_qids": test, "tune_qids": tune, "final_qids": final,
               "n_train_twins": len(s_tr), "hidden_index_note": "i = output of decoder layer i-1"}
    for f in sorted(set(TEMPLATE_FAMILY.values())):
        vectors["family"][f] = mean_diff(acts, *twin_index(meta, train, meta.family == f))
    for t in sorted(meta.template.unique()):
        vectors["template"][t] = mean_diff(acts, *twin_index(meta, train, meta.template == t))
    pair_sums = {}
    for i, p in enumerate(PRONOUN_PAIRS):
        sp, jp = twin_index(meta, train, meta.pair == p)
        vectors["pair"][p] = mean_diff(acts, sp, jp)
        vectors["pair_shuffled"][p] = mean_diff(acts, sp, jp, shuffle_seed=cfg.seed + 10 + i)
        pair_sums[p] = (vectors["pair"][p] * len(sp), len(sp))
    torch.save(vectors, out / "vectors.pt")
    print(f"built vectors from {len(s_tr)} train twins ({len(train)} train questions); "
          f"test {len(test)} = tune {len(tune)} + final {len(final)} -> {out / 'vectors.pt'}")

    # ---- sanity checks on the test half ----
    s_te, j_te = twin_index(meta, test)
    test_rows = meta[meta.qid.isin(set(test))].row.to_numpy()
    g = torch.Generator().manual_seed(cfg.seed)
    v_rand = torch.randn(v.shape, generator=g)
    proj_v = project(acts, v)
    torch.save({"proj": proj_v.half(), "meta": meta.to_dict("records")}, out / "projections.pt")
    rows = []
    kinds = {"deference": v, "shuffled": vectors["v_shuffled"], "random": v_rand}
    projs = {"deference": proj_v, "shuffled": project(acts, vectors["v_shuffled"]), "random": project(acts, v_rand)}
    for kind, vec in kinds.items():
        sep, dp = separation(projs[kind], s_te, j_te)
        btw = between(projs[kind], meta, s_te, j_te)
        fc = flip_corr(projs[kind], meta, test_rows)
        for i in range(n_hidden):
            rows.append(dict(vector=kind, hidden_idx=i, decoder_layer=i - 1, norm=float(vec[i].norm()),
                             sep=float(sep[i]), dprime=float(dp[i]), between=float(btw[i]), flip_corr=float(fc[i])))
    lay = pd.DataFrame(rows)

    # pairs: cosine matrix and leave-one-pair-out separation
    P = list(PRONOUN_PAIRS)
    cos_rows, loo = [], {}
    for a in P:
        for b in P:
            if a < b:
                c = cos(vectors["pair"][a], vectors["pair"][b])
                cs = cos(vectors["pair_shuffled"][a], vectors["pair_shuffled"][b])
                for i in range(n_hidden):
                    cos_rows.append(dict(pair_a=a, pair_b=b, hidden_idx=i, cos=float(c[i]), cos_shuffled=float(cs[i])))
    for p in P:
        tot = sum(pair_sums[q][0] for q in P if q != p)
        n = sum(pair_sums[q][1] for q in P if q != p)
        sp, jp = twin_index(meta, test, meta.pair == p)
        loo[p] = separation(project(acts, tot / n), sp, jp)[0]
    pc = pd.DataFrame(cos_rows)
    pc.to_csv(out / "sanity_pair_cosine.csv", index=False)
    mean_cos = pc.groupby("hidden_idx")[["cos", "cos_shuffled"]].mean()
    oc_ae = pc[(pc.pair_a == "anh_em") & (pc.pair_b == "ong_chau")].set_index("hidden_idx").cos
    loo_mean = torch.stack(list(loo.values())).mean(0)
    loo_min = torch.stack(list(loo.values())).min(0).values
    d = lay[lay.vector == "deference"].set_index("hidden_idx")
    d = d.assign(pair_cos=mean_cos.cos, cos_oc_ae=oc_ae, loo_sep=loo_mean.numpy(), loo_sep_min=loo_min.numpy())
    lay = lay.merge(d[["pair_cos", "cos_oc_ae", "loo_sep", "loo_sep_min"]].reset_index().assign(vector="deference"),
                    on=["vector", "hidden_idx"], how="left")
    lay = lay.merge(mean_cos.cos_shuffled.rename("pair_cos").reset_index().assign(vector="shuffled"),
                    on=["vector", "hidden_idx"], how="left", suffixes=("", "_s"))
    lay["pair_cos"] = lay.pair_cos.fillna(lay.pop("pair_cos_s"))
    lay.to_csv(out / "sanity_layers.csv", index=False)

    fam = cos(vectors["family"]["hedged"], vectors["family"]["blunt"])
    prim = cos(vectors["v_primary"], v)
    tmpl = list(vectors["template"])
    tcos = torch.stack([cos(vectors["template"][a], vectors["template"][b])
                        for i, a in enumerate(tmpl) for b in tmpl[i + 1:]]).mean(0)

    pd.set_option("display.width", 220)
    print(f"\nSANITY CHECKS on {len(s_te)} test twins ({len(test)} questions), per hidden index "
          f"(i = output of decoder layer i-1)")
    print("  sep/between/flip_corr for deference (D), shuffled (S), random (R); chance: sep .5, between .333, corr 0")
    hdr = (f"  {'idx':>3} {'‖v‖':>6} | {'sep D':>5} {'S':>5} {'R':>5} | {'dprime':>6} | {'btw D':>5} {'S':>5} {'R':>5} | "
           f"{'pairCos':>7} {'shufCos':>7} {'ông·anh':>7} | {'LOO':>5} {'LOOmin':>6} | {'fam cos':>7} {'tmpl cos':>8} | "
           f"{'flipCorr D':>10} {'S':>6} {'R':>6}")
    print(hdr)
    L = {k: lay[lay.vector == k].set_index("hidden_idx") for k in kinds}
    for i in range(n_hidden):
        D, S, R = L["deference"].loc[i], L["shuffled"].loc[i], L["random"].loc[i]
        print(f"  {i:3d} {D.norm:6.2f} | {D.sep:5.2f} {S.sep:5.2f} {R.sep:5.2f} | {D.dprime:6.2f} | "
              f"{D.between:5.2f} {S.between:5.2f} {R.between:5.2f} | {D.pair_cos:7.2f} {S.pair_cos:7.2f} "
              f"{D.cos_oc_ae:7.2f} | {D.loo_sep:5.2f} {D.loo_sep_min:6.2f} | {float(fam[i]):7.2f} {float(tcos[i]):8.2f} | "
              f"{D.flip_corr:10.3f} {S.flip_corr:6.3f} {R.flip_corr:6.3f}")
    print(f"\n  cos(v_primary, v) at idx 8/16/24: {float(prim[8]):.2f} / {float(prim[16]):.2f} / {float(prim[24]):.2f}")
    print("\n  mean projection on v by role (test prompts, demeaned within question x template), selected layers:")
    m = meta.loc[test_rows]
    for i in (4, 8, 12, 16, 20, 24, n_hidden - 1):
        x = proj_v[torch.as_tensor(test_rows), i].numpy()
        x = x - pd.Series(x).groupby([m.qid.to_numpy(), m.template.to_numpy()]).transform("mean").to_numpy()
        r = pd.Series(x).groupby(m.role.to_numpy()).mean()
        print(f"    idx {i:2d}: senior {r['senior']:+.3f}  neutral {r['neutral']:+.3f}  junior {r['junior']:+.3f}")
    print(f"\nwrote vectors.pt, projections.pt, sanity_layers.csv, sanity_pair_cosine.csv in {out}")


if __name__ == "__main__":
    main()
