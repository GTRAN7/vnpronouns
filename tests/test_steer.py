import json

import numpy as np
import pandas as pd
import torch
import yaml

from vnpronouns.prompts import PUSHBACK_TEMPLATES, all_conditions
from vnpronouns.steer import first_turn_table, pick_vector, unit_name


def test_pick_vector_controls_have_same_norm():
    g = torch.Generator().manual_seed(0)
    vectors = {"v": torch.randn(5, 32, generator=g) * 3, "v_shuffled": torch.randn(5, 32, generator=g) * 0.1}
    v = pick_vector(vectors, "deference", 2, 0)
    for kind in ("random", "shuffled"):
        c = pick_vector(vectors, kind, 2, 0)
        assert torch.isclose(c.norm(), v.norm())
        assert abs(float(torch.nn.functional.cosine_similarity(c, v, dim=0))) < 0.9
    assert unit_name("random", 12, 1.0) == "random_L12_lam+1.csv" and unit_name("x", 3, 0) == "none_L-1_lam+0.csv"
    assert unit_name("deference", -2, 1.0) == "deference_Lall_lam+1.csv"
    # random == random0 (Phase 6 draw); randomK are independent directions of the same length
    assert torch.equal(pick_vector(vectors, "random0", 2, 0), pick_vector(vectors, "random", 2, 0))
    r1, r2 = pick_vector(vectors, "random1", 2, 0), pick_vector(vectors, "random2", 2, 0)
    assert torch.isclose(r1.norm(), v.norm()) and not torch.allclose(r1, r2)


def test_first_turn_table_steers_from_user_turn():
    from transformers import AutoTokenizer

    from vnpronouns.prompts import SAMPLE_QUESTION

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    t = first_turn_table(tok, [SAMPLE_QUESTION])
    assert len(t) == 13
    r = t.iloc[0]
    ids = tok(r.text, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids[r.pushback_tok:r.pushback_tok + 2]).startswith("<|im_start|>user")


def _fake_units(root, split, effect):
    """lam-0 base + deference/random units; deference lowers senior P(push) by `effect` per unit lam."""
    rng = np.random.default_rng(0)
    d = root / "Qwen2.5-0.5B-Instruct" / "steer" / split
    d.mkdir(parents=True)
    rows = []
    for q in range(12):
        for c in all_conditions():
            for t in list(PUSHBACK_TEMPLATES)[:3]:
                rows.append(dict(qid=f"q{q}", condition=c.name, pair=c.pair, role=c.role, template=t,
                                 correct="A", pushback="B", p_push=0.5 + 0.05 * rng.standard_normal()))
    base = pd.DataFrame(rows)

    def unit(vec, layer, lam, eff=0.0):
        u = base.copy()
        if vec == "deference":
            u.loc[u.role == "senior", "p_push"] -= eff * lam
            u.loc[u.role != "senior", "p_push"] -= 0.5 * eff * lam
        u["flip"] = u.p_push > 0.5
        u["top"] = np.where(u.flip, "B", "A")
        u["p_letters"] = 0.99
        u["vector"], u["layer"], u["lam"] = (vec, layer, lam) if lam else ("none", -1, 0.0)
        u.to_csv(d / unit_name(vec, layer, lam), index=False)

    unit("none", -1, 0)
    for layer, eff in ((8, 0.2 * effect), (12, effect)):
        for lam in (-1, 1):
            unit("deference", layer, lam, eff)
            unit("random", layer, lam)
    return d


def test_steer_report_selects_the_effective_layer(tmp_path, monkeypatch, capsys):
    import sys

    from vnpronouns import steer_report

    _fake_units(tmp_path, "tune", effect=0.1)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "output_dir": str(tmp_path)}))
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--split", "tune", "--n-boot", "200"])
    steer_report.main()
    sel = json.loads((tmp_path / "Qwen2.5-0.5B-Instruct" / "steer" / "selected.json").read_text())
    assert sel["layer"] == 12 and sel["score"] > 0
    # final-style report on the same units also runs
    (tmp_path / "Qwen2.5-0.5B-Instruct" / "steer" / "tune").rename(tmp_path / "Qwen2.5-0.5B-Instruct" / "steer" / "final")
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--split", "final", "--n-boot", "200"])
    steer_report.main()
    out = capsys.readouterr().out
    assert "PER-TEMPLATE PREDICTION" in out and "DEFERENCE vector" in out


def test_null_and_ablate_sections(tmp_path, monkeypatch, capsys):
    import sys

    from vnpronouns import steer_report

    d = _fake_units(tmp_path, "tune", effect=0.1)
    base = pd.read_csv(d / "none_L-1_lam+0.csv")
    for k in range(1, 6):   # extra random directions with small effects
        u = base.copy()
        u["p_push"] -= 0.002 * k * (u.role == "senior")
        u["flip"] = u.p_push > 0.5
        u["top"] = np.where(u.flip, "B", "A")
        u["p_letters"], u["vector"], u["layer"], u["lam"] = 0.99, f"random{k}", 12, 1.0
        u.to_csv(d / unit_name(f"random{k}", 12, 1.0), index=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "output_dir": str(tmp_path)}))
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--split", "tune", "--section", "null"])
    res = steer_report.main()
    r = res[(res.layer == 12) & (res.lam == 1)].iloc[0]
    assert r.n_random == 6 and r.gap_change_p < 0.2   # deference beats every random direction

    # ablation: deference closes the S-J gap, random does nothing
    a = d.parent / "tune_ablate_allpos"
    a.mkdir()
    base.to_csv(a / "none_L-1_lam+0.csv", index=False)
    for vec in ("deference", "random", "random1"):
        u = base.copy()
        if vec == "deference":
            u.loc[u.role == "senior", "p_push"] -= 0.2
        u["flip"] = u.p_push > 0.5
        u["top"] = np.where(u.flip, "B", "A")
        u["p_letters"], u["vector"], u["layer"], u["lam"] = 0.99, vec, -2, 1.0
        u.to_csv(a / unit_name(vec, -2, 1.0), index=False)
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg), "--split", "tune", "--section", "ablate", "--n-boot", "200"])
    res = steer_report.main()
    dd = res.set_index("vector")
    assert dd.loc["deference", "d_gap_SJ"] < -0.1 and dd.loc["random", "d_gap_SJ"] == 0
    assert "S-J gap change" in capsys.readouterr().out
