from collections import Counter
from pathlib import Path

import pytest

from vnpronouns.data import assign_letters, load_questions, pronoun_hits, read_csv, validate_raw
from vnpronouns.prompts import LETTERS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "questions_raw.csv"
BUILT = ROOT / "data" / "questions.csv"


@pytest.mark.parametrize("text,hits", [
    ("Tiếng Anh", ["anh"]), ("Đảo Cát Bà", ["bà"]), ("người đàn ông", ["ông"]),
    ("Quảng trường Ba Đình", []), ("Nhà Thanh", []), ("Thành Cổ Loa", []), ("Emmerdale", []),
])
def test_pronoun_hits_whole_words_with_diacritics(text, hits):
    assert pronoun_hits(text) == hits


def _row(i, **kw):
    return {"qid": f"q{i}", "source": "t", "source_id": "", "question_en": "", "question": f"Câu {i}?",
            "answer": "Đúng", "pushback_text": "Sai1", "distractor_1": "Sai2", "distractor_2": "Sai3",
            "checked": "0", "notes": "", **kw}


def test_validate_raw_catches_problems():
    errs = validate_raw([_row(1), _row(1), _row(2, answer="Sai1"), _row(3, distractor_2="Nước Anh")])
    assert any("duplicate qid" in e for e in errs)
    assert any("not distinct" in e for e in errs)
    assert any("pronoun" in e for e in errs)


def test_assign_letters_balanced_and_faithful():
    rows = [_row(i) for i in range(120)]
    out = assign_letters(rows, seed=1)
    assert Counter(r["correct"] for r in out) == {l: 30 for l in LETTERS}
    assert Counter(r["pushback"] for r in out) == {l: 30 for l in LETTERS}
    for r in out:
        assert r["correct"] != r["pushback"]
        assert r[r["correct"]] == "Đúng" and r[r["pushback"]] == "Sai1"
        assert sorted(r[l] for l in LETTERS) == ["Sai1", "Sai2", "Sai3", "Đúng"]
    assert assign_letters(rows, seed=1) == out  # deterministic


def test_real_dataset_is_valid_and_built():
    raw = read_csv(RAW)
    assert validate_raw(raw) == []
    assert len({r["question"].casefold() for r in raw}) == len(raw), "duplicate question text"
    qs = load_questions(BUILT)
    assert [q.qid for q in qs] == [r["qid"] for r in raw]
    by_id = {r["qid"]: r for r in raw}
    for q in qs:
        assert q.options[q.correct] == by_id[q.qid]["answer"]
        assert q.options[q.pushback] == by_id[q.qid]["pushback_text"]
