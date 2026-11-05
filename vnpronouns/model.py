"""Model loading, batched next-letter scoring, last-token activation capture, and steering.

Everything here works on lists of already-built prompt strings, so the same code serves
the baseline filter, the behavior run, activation caching and steering.

Conventions
- hidden-state index i (0..n_layers): 0 = embeddings, i = output of decoder layer i-1.
  Vectors are stored per hidden-state index; steering "at layer L" edits the output of
  decoder layer L, i.e. hidden-state index L+1.
- Prompts are left-padded and explicit position_ids are passed, so a padded batch gives the
  same numbers as running each prompt alone.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vnpronouns.config import Config
from vnpronouns.prompts import LETTERS


def load_model(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tok.padding_side = "left"
    kw = {}
    if cfg.load_in_8bit:
        from transformers import BitsAndBytesConfig

        kw.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
    else:
        kw.update(dtype=cfg.torch_dtype())
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kw)
    if not cfg.load_in_8bit:
        model.to(cfg.resolve_device())
    model.eval()
    return model, tok


def letter_token_ids(tok) -> dict[str, list[int]]:
    """Single-token spellings of each answer letter ('A' and ' A'); their probabilities are summed."""
    out = {}
    for letter in LETTERS:
        ids = {tok.encode(s, add_special_tokens=False)[0] for s in (letter, " " + letter)
               if len(tok.encode(s, add_special_tokens=False)) == 1}
        if not ids:
            raise ValueError(f"no single-token spelling for {letter!r}")
        out[letter] = sorted(ids)
    return out


def decoder_layers(model):
    return model.model.layers


def char_to_token_index(tok, text: str, char_pos: int) -> int:
    """Index of the first token that starts at or after `char_pos` (unpadded)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    for i, (start, _end) in enumerate(enc["offset_mapping"]):
        if start >= char_pos:
            return i
    raise ValueError("char_pos is past the end of the text")


@dataclass
class Steering:
    """Edit h <- h - lam * vector at the output of decoder layer `layer`, for token positions >= start."""
    layer: int
    vector: torch.Tensor  # [hidden]
    lam: float


@dataclass
class ScoreResult:
    probs: torch.Tensor           # [N, 4] probability of A..D (each summed over spellings)
    hidden: torch.Tensor | None   # [N, n_layers+1, hidden] last-token residual stream, if requested


@contextmanager
def _steering_hook(model, steering: Steering | None, state: dict):
    if steering is None or steering.lam == 0:
        yield
        return
    layer = decoder_layers(model)[steering.layer]

    def hook(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        v = steering.vector.to(device=h.device, dtype=h.dtype)
        h = h - steering.lam * state["mask"].to(h.dtype)[..., None] * v
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def score_prompts(model, tok, texts: list[str], batch_size: int = 16, return_hidden: bool = False,
                  steering: Steering | None = None, steer_starts: list[int] | None = None,
                  progress: bool = False) -> ScoreResult:
    """Next-token letter probabilities at the end of each prompt (and optionally last-token activations).

    steer_starts[i] = unpadded token index in texts[i] from which steering applies.
    """
    if steering is not None and steer_starts is None:
        raise ValueError("steer_starts is required with steering")
    device = next(model.parameters()).device
    letter_ids = letter_token_ids(tok)
    probs_out, hidden_out, state = [], [], {}
    batches = range(0, len(texts), batch_size)
    if progress:
        from tqdm.auto import tqdm
        batches = tqdm(batches, leave=False)
    with _steering_hook(model, steering, state):
        for b in batches:
            chunk = texts[b:b + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            attn = enc["attention_mask"]
            pos = (attn.cumsum(-1) - 1).clamp(min=0)
            if steering is not None:
                pad = attn.shape[1] - attn.sum(-1)  # left-pad lengths
                start = pad + torch.tensor(steer_starts[b:b + batch_size], device=device)
                state["mask"] = torch.arange(attn.shape[1], device=device)[None, :] >= start[:, None]
            out = model(input_ids=enc["input_ids"], attention_mask=attn, position_ids=pos,
                        output_hidden_states=return_hidden, use_cache=False)
            p = torch.softmax(out.logits[:, -1].float(), dim=-1)
            probs_out.append(torch.stack([p[:, ids].sum(-1) for ids in letter_ids.values()], -1).cpu())
            if return_hidden:
                hidden_out.append(torch.stack([h[:, -1] for h in out.hidden_states], 1).to(torch.float16).cpu())
    return ScoreResult(torch.cat(probs_out), torch.cat(hidden_out) if return_hidden else None)
