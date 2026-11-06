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


def test_resumable_matches_and_resumes(tiny, tmp_path):
    from vnpronouns.model import score_prompts_resumable

    model, tok, table = tiny
    texts, starts = table.text.tolist(), table.pushback_tok.tolist()
    ref = score_prompts(model, tok, texts, 2)
    path = tmp_path / "c.pt"
    r = score_prompts_resumable(model, tok, texts, 2, path, chunk_size=3, progress=False)
    torch.testing.assert_close(r.probs, ref.probs)
    # simulate a crash after the first chunk, then resume
    saved = torch.load(path, weights_only=False)
    saved["probs"] = saved["probs"][:1]
    torch.save(saved, path)
    r2 = score_prompts_resumable(model, tok, texts, 2, path, chunk_size=3, progress=False)
    torch.testing.assert_close(r2.probs, ref.probs)
    # different texts -> cache ignored
    r3 = score_prompts_resumable(model, tok, texts[:2], 2, path, chunk_size=3, progress=False)
    torch.testing.assert_close(r3.probs, ref.probs[:2])
    # steering slices steer_starts per chunk
    v = torch.randn(64)
    st = Steering(1, v, 3.0)
    ref_s = score_prompts(model, tok, texts, 2, steering=st, steer_starts=starts)
    r4 = score_prompts_resumable(model, tok, texts, 2, tmp_path / "s.pt", chunk_size=3, progress=False,
                                 steering=st, steer_starts=starts)
    torch.testing.assert_close(r4.probs, ref_s.probs)


def test_sort_by_length_preserves_order(tiny):
    model, tok, table = tiny
    texts, starts = table.text.tolist(), table.pushback_tok.tolist()
    st = Steering(1, torch.randn(64), 2.0)
    a = score_prompts(model, tok, texts, 3, return_hidden=True, steering=st, steer_starts=starts)
    b = score_prompts(model, tok, texts, 3, return_hidden=True, steering=st, steer_starts=starts, sort_by_length=True)
    torch.testing.assert_close(a.probs, b.probs, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(a.hidden.float(), b.hidden.float(), atol=2e-2, rtol=1e-2)


def test_ablation_removes_direction(tiny):
    from vnpronouns.model import decoder_layers

    model, tok, table = tiny
    texts = table.text.tolist()
    starts = [0] * len(texts)
    v = torch.randn(64) * 5
    u = v / v.norm()
    seen = []

    def grab(_m, args, kwargs):   # what decoder layer 1 actually receives (= layer 0 output after edits)
        h = args[0] if args else kwargs["hidden_states"]
        seen.append((h.float() @ u).abs().max())

    hd = decoder_layers(model)[1].register_forward_pre_hook(grab, with_kwargs=True)
    try:
        # one prompt, so no pad positions (those are left unedited on purpose)
        score_prompts(model, tok, texts[:1], 1)
        score_prompts(model, tok, texts[:1], 1, steer_starts=[0],
                      steering=[Steering(0, v, 1.0, "ablate"), Steering(2, v, 1.0, "ablate")])
    finally:
        hd.remove()
    before, after = seen
    assert after < 0.01 * before                                    # v component gone from the layer-0 output
    none = score_prompts(model, tok, texts, len(texts))
    abl = score_prompts(model, tok, texts, len(texts), steer_starts=starts,
                        steering=[Steering(0, v, 1.0, "ablate"), Steering(2, v, 1.0, "ablate")])
    # ablation does not depend on |v|, and batching / padding does not change it
    abl1 = score_prompts(model, tok, texts, 1, steer_starts=starts,
                         steering=[Steering(0, v * 0.01, 1.0, "ablate"), Steering(2, v, 1.0, "ablate")])
    torch.testing.assert_close(abl.probs, abl1.probs, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(abl.probs, none.probs)
