import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from vnpronouns.common import prompt_table
from vnpronouns.model import Steering, letter_token_ids, score_prompts
from vnpronouns.prompts import SAMPLE_QUESTION, Question, all_conditions


@pytest.fixture(scope="module")
def tiny():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg).eval()
    other = Question("q2", "Thủ đô của Pháp là gì? Một câu hỏi dài hơn hẳn để có padding.",
                     {"A": "Paris", "B": "Lyon", "C": "Marseille", "D": "Nice"}, "A", "B")
    table = prompt_table(tok, [SAMPLE_QUESTION, other], all_conditions()[:2], ["flat"])
    return model, tok, table


def test_letter_ids(tiny):
    _, tok, _ = tiny
    ids = letter_token_ids(tok)
    assert list(ids) == ["A", "B", "C", "D"] and all(len(v) == 2 for v in ids.values())


def test_batched_equals_single(tiny):
    model, tok, table = tiny
    texts = table.text.tolist()
    batched = score_prompts(model, tok, texts, batch_size=len(texts), return_hidden=True)
    single = score_prompts(model, tok, texts, batch_size=1, return_hidden=True)
    torch.testing.assert_close(batched.probs, single.probs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(batched.hidden.float(), single.hidden.float(), atol=2e-2, rtol=1e-2)


def test_steering(tiny):
    model, tok, table = tiny
    texts, starts = table.text.tolist(), table.pushback_tok.tolist()
    v = torch.randn(64)
    none = score_prompts(model, tok, texts, batch_size=4)
    zero = score_prompts(model, tok, texts, 4, steering=Steering(1, v, 0.0), steer_starts=starts)
    torch.testing.assert_close(none.probs, zero.probs)
    on_b = score_prompts(model, tok, texts, len(texts), steering=Steering(1, v, 3.0), steer_starts=starts)
    on_1 = score_prompts(model, tok, texts, 1, steering=Steering(1, v, 3.0), steer_starts=starts)
    assert not torch.allclose(on_b.probs, none.probs)
    torch.testing.assert_close(on_b.probs, on_1.probs, atol=1e-5, rtol=1e-4)  # mask aligned under padding
