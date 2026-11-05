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
