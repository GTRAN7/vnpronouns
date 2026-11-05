import re
from collections import Counter

import pytest
from transformers import AutoTokenizer

from vnpronouns.prompts import (
    PREFILL, PRONOUN_PAIRS, PUSHBACK_TEMPLATES, SAMPLE_QUESTION, all_conditions, build_messages, build_prompt,
)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def _words(msgs):
    return Counter(w.lower() for m in msgs for w in re.findall(r"\w+", m["content"]))


def test_thirteen_conditions():
    names = [c.name for c in all_conditions()]
    assert len(names) == 13 and len(set(names)) == 13


@pytest.mark.parametrize("pair", list(PRONOUN_PAIRS))
@pytest.mark.parametrize("template", list(PUSHBACK_TEMPLATES))
def test_senior_junior_twins_have_identical_word_multisets(pair, template):
    conds = {c.role: c for c in all_conditions() if c.pair == pair}
    s = build_messages(SAMPLE_QUESTION, conds["senior"], template)
    j = build_messages(SAMPLE_QUESTION, conds["junior"], template)
    assert _words(s) == _words(j)
    assert s != j


def test_first_assistant_turn_identical_across_conditions():
    turns = {build_messages(SAMPLE_QUESTION, c, "flat")[1]["content"] for c in all_conditions()}
    assert turns == {"Đáp án: B"}


def test_prompt_format(tok):
    for c in all_conditions():
        p = build_prompt(tok, SAMPLE_QUESTION, c, "flat")
        assert p.text.count("You are Qwen, created by Alibaba Cloud.") == 1  # default system prompt
        assert p.text.endswith("<|im_start|>assistant\n" + PREFILL)
        assert p.text[p.pushback_char_start:].startswith("<|im_start|>user\n")


def test_no_prefill_variant():
    from transformers import AutoTokenizer

    from vnpronouns.prompts import PREFILL, SAMPLE_QUESTION, all_conditions, build_prompt

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    c = all_conditions()[0]
    a = build_prompt(tok, SAMPLE_QUESTION, c, "flat").text
    b = build_prompt(tok, SAMPLE_QUESTION, c, "flat", prefill=False).text
    assert a == b + PREFILL and b.endswith("<|im_start|>assistant\n")


def test_parse_letter():
    from vnpronouns.freetext import parse_letter

    opts = {"A": "Pablo Picasso", "B": "Leonardo da Vinci", "C": "Van Gogh", "D": "Michelangelo"}
    assert parse_letter("Đáp án: B", opts) == "B"
    assert parse_letter("Xin lỗi, đáp án đúng là A) Pablo Picasso.", opts) == "A"
    assert parse_letter("Cháu nghĩ là Leonardo da Vinci đã vẽ.", opts) == "B"
    assert parse_letter("Cháu không biết.", opts) is None
