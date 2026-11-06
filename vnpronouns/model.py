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

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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

        # dtype = compute/precision of the non-quantized modules (float16 on a T4)
        kw.update(quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
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
    """Edit the output of decoder layer `layer` at token positions >= start.

    mode "add":    h <- h - lam * vector
    mode "ablate": h <- h - (h . u) u with u = vector / |vector| (directional ablation; lam only
                   switches it on, any non-zero value)
    """
    layer: int
    vector: torch.Tensor  # [hidden]
    lam: float
    mode: str = "add"


@dataclass
class ScoreResult:
    probs: torch.Tensor           # [N, 4] probability of A..D (each summed over spellings)
    hidden: torch.Tensor | None   # [N, n_layers+1, hidden] last-token residual stream, if requested


@contextmanager
def _steering_hook(model, steering, state: dict):
    """steering: one Steering, a list of them (e.g. ablation at every layer), or None."""
    edits = [s for s in (steering if isinstance(steering, (list, tuple)) else [steering])
             if s is not None and s.lam != 0]
    if not edits:
        yield
        return

    def make_hook(st: Steering):
        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            mask = state["mask"]
            if mask.shape[1] != h.shape[1]:   # incremental decoding (generate): new tokens are past the start
                mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            mask = mask.to(device=h.device, dtype=h.dtype)[..., None]
            if st.mode == "ablate":
                u = (st.vector.float() / st.vector.float().norm().clamp(min=1e-8)).to(h.device)
                proj = (h.float() @ u)[..., None]
                h = h - (mask * proj * u).to(h.dtype)
            else:
                h = h - st.lam * mask * st.vector.to(device=h.device, dtype=h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return hook

    layers = decoder_layers(model)
    handles = [layers[st.layer].register_forward_hook(make_hook(st)) for st in edits]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


@torch.no_grad()
def score_prompts(model, tok, texts: list[str], batch_size: int = 16, return_hidden: bool = False,
                  steering: "Steering | list[Steering] | None" = None, steer_starts: list[int] | None = None,
                  progress: bool = False, sort_by_length: bool = False) -> ScoreResult:
    """Next-token letter probabilities at the end of each prompt (and optionally last-token activations).

    steer_starts[i] = unpadded token index in texts[i] from which steering applies.
    sort_by_length batches prompts of similar length together (less padding); outputs are
    returned in the original order.
    """
    if steering is not None and steer_starts is None:
        raise ValueError("steer_starts is required with steering")
    if sort_by_length and len(texts) > 1:
        lens = [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
        order = sorted(range(len(texts)), key=lambda i: -lens[i])
        r = score_prompts(model, tok, [texts[i] for i in order], batch_size, return_hidden, steering,
                          None if steer_starts is None else [steer_starts[i] for i in order], progress)
        inv = torch.empty(len(order), dtype=torch.long)
        inv[torch.tensor(order)] = torch.arange(len(order))
        return ScoreResult(r.probs[inv], None if r.hidden is None else r.hidden[inv])
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
                        output_hidden_states=return_hidden, use_cache=False, logits_to_keep=1)
            p = torch.softmax(out.logits[:, -1].float(), dim=-1)
            probs_out.append(torch.stack([p[:, ids].sum(-1) for ids in letter_ids.values()], -1).cpu())
            if return_hidden:
                hidden_out.append(torch.stack([h[:, -1] for h in out.hidden_states], 1).to(torch.float16).cpu())
    return ScoreResult(torch.cat(probs_out), torch.cat(hidden_out) if return_hidden else None)


def score_prompts_resumable(model, tok, texts: list[str], batch_size: int, cache_path: str | Path,
                            chunk_size: int = 512, progress: bool = True,
                            steer_starts: list[int] | None = None, **kw) -> ScoreResult:
    """score_prompts in chunks, checkpointing each finished chunk to `cache_path`.

    If the run dies (e.g. a Colab disconnect), calling this again with the same texts resumes
    after the last saved chunk. The cache is keyed on the texts and options, so a changed prompt
    set starts fresh instead of silently reusing stale numbers.
    """
    cache_path = Path(cache_path)
    h = hashlib.sha1("\x00".join(texts).encode())
    h.update(repr((steer_starts, sorted((k, v) for k, v in kw.items() if k != "steering"))).encode())
    if kw.get("steering") is not None:
        sts = kw["steering"] if isinstance(kw["steering"], (list, tuple)) else [kw["steering"]]
        for st in sts:
            h.update(f"{st.layer}|{st.lam}|{st.mode}|".encode() + st.vector.float().numpy().tobytes())
    key = h.hexdigest()
    done = {"key": key, "probs": [], "hidden": []}
    if cache_path.exists():
        prev = torch.load(cache_path, weights_only=False)
        if prev.get("key") == key:
            done = prev
    n_chunks = (len(texts) + chunk_size - 1) // chunk_size
    if done["probs"]:
        print(f"  resuming from {cache_path.name}: {len(done['probs'])}/{n_chunks} chunks already done")
    for i in range(len(done["probs"]), n_chunks):
        if progress:
            print(f"  chunk {i + 1}/{n_chunks}", flush=True)
        sl = slice(i * chunk_size, (i + 1) * chunk_size)
        r = score_prompts(model, tok, texts[sl], batch_size, progress=progress,
                          steer_starts=None if steer_starts is None else steer_starts[sl], **kw)
        done["probs"].append(r.probs)
        if r.hidden is not None:
            done["hidden"].append(r.hidden)
        tmp = cache_path.with_suffix(".tmp")
        torch.save(done, tmp)
        os.replace(tmp, cache_path)
    return ScoreResult(torch.cat(done["probs"]) if done["probs"] else torch.empty(0, len(LETTERS)),
                       torch.cat(done["hidden"]) if done["hidden"] else None)
