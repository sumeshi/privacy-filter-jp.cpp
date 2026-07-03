#!/usr/bin/env python3
"""Convert an OpenAIPrivacyFilter HF checkpoint to the privacy-filter.cpp GGUF.

Self-contained: reads ``config.json`` + ``model.safetensors`` (single or
sharded) + ``tokenizer.json`` straight from the HF model dir and emits the
``openai-privacy-filter`` GGUF that this repo's loader (src/gguf_loader.cpp,
src/model.cpp) expects. It does NOT depend on llama.cpp or its convert script —
the architecture is small and fully specified by the loader, so the whole
mapping lives here.

The Japanese GGUF built with this script is published at:
  - https://huggingface.co/sumeshi/privacy-filter-jp-GGUF

Usage:
  python scripts/convert.py --model runs/pf-jp/model-ft \\
      --outfile runs/pf-jp/privacy-filter-jp-f16.gguf --name privacy-filter-jp
  python scripts/convert.py --model <dir> --outtype f32 --outfile pf-f32.gguf

The architecture (gpt-oss MoE body re-purposed as a bidirectional token
classifier): 8 layers, 14/2 heads, head_dim 64, d_model 640, 128 experts top-4,
o200k vocab, attention sinks, interleaved RoPE + YaRN, a TOKEN_CLS score head.
The two load-bearing tensor transforms (verified by the parity test):
  - experts.gate_up_proj is packed as CONCATENATED halves (gate = first
    intermediate_size output columns, up = the rest), transposed to
    [E, 2*inter, in] then split — NOT gpt-oss's interleaved even/odd;
  - experts.down_proj is transposed (-1,-2) like the gpt-oss dense path.
The expert matrices are square (inter == d_model == 640), so a wrong transpose
is silently shape-valid but numerically wrong — parity is the guard.
RoPE: the loader recomputes the YaRN freq factors from the rope KVs at load
time (rope.scaling.yarn_truncate=false), so no per-dim rope_freqs are baked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARCH = "openai-privacy-filter"

# Token types in tokenizer.ggml.token_type (gguf.TokenType), mirrored here so we
# don't depend on the enum's import path: NORMAL vocab, CONTROL specials, UNUSED
# placeholders for the reserved gap ids between the specials and vocab_size.
TT_NORMAL, TT_CONTROL, TT_UNUSED = 1, 3, 5

# GGUF tensor names that carry the weight matrices quantized to f16 in an f16
# build; everything else (norms, biases, attn sinks, the router weight) stays
# f32. Matches the published reference file's per-tensor precision exactly.
F16_SUFFIXES = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
)
F16_GLOBALS = ("token_embd.weight", "cls.output.weight")


def gpt2_byte_encoder() -> dict[int, str]:
    # GPT-2/o200k byte<->unicode: printable bytes map to themselves, the rest to
    # 0x100+n in order. tokenizer.json vocab keys are already in this encoding,
    # so we only need it to byte-encode any added-token content we synthesize.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(0x100 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def load_tokenizer(model_dir: Path):
    """Extract (tokens, token_types, merges, eos, pad) from tokenizer.json.

    Mirrors llama.cpp's gpt2 vocab handling: base BPE vocab as NORMAL, special
    added tokens as CONTROL, and reserved gap ids up to vocab_size filled with
    ``[PAD{i}]`` placeholders typed UNUSED.
    """
    tj = json.loads((model_dir / "tokenizer.json").read_text())
    cfg = json.loads((model_dir / "config.json").read_text())
    vocab_size = cfg["vocab_size"]

    enc = gpt2_byte_encoder()
    byte_encode = lambda s: "".join(enc[b] for b in s.encode("utf-8"))

    tokens: list[str | None] = [None] * vocab_size
    types = [TT_UNUSED] * vocab_size
    for tok, tid in tj["model"]["vocab"].items():
        tokens[tid] = tok
        types[tid] = TT_NORMAL
    for at in tj.get("added_tokens", []):
        tid = at["id"]
        if 0 <= tid < vocab_size:
            tokens[tid] = byte_encode(at["content"])
            types[tid] = TT_CONTROL if at.get("special") else TT_NORMAL
    for i in range(vocab_size):
        if tokens[i] is None:
            tokens[i] = f"[PAD{i}]"  # reserved gap id; never matches real text

    merges_raw = tj["model"]["merges"]
    merges = [m if isinstance(m, str) else f"{m[0]} {m[1]}" for m in merges_raw]

    eos = cfg.get("eos_token_id")
    pad = cfg.get("pad_token_id", eos)
    return tokens, types, merges, eos, pad


def load_state_dict(model_dir: Path):
    """Yield (name, torch.Tensor) for every weight, from single or sharded
    safetensors, one tensor resident at a time."""
    from safetensors import safe_open

    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        shard_of = {name: model_dir / shard for name, shard in weight_map.items()}
    else:
        single = model_dir / "model.safetensors"
        with safe_open(single, framework="pt") as f:
            shard_of = {name: single for name in f.keys()}

    handles: dict[Path, object] = {}
    for name, path in shard_of.items():
        if path not in handles:
            handles[path] = safe_open(path, framework="pt")
        yield name, handles[path].get_tensor(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="HF model directory")
    ap.add_argument("--outfile", required=True, type=Path, help="output .gguf path")
    ap.add_argument("--outtype", choices=("f16", "f32"), default="f16")
    ap.add_argument("--name", default=None, help="general.name (default: model dir name)")
    args = ap.parse_args()

    import torch
    import gguf

    cfg = json.loads((args.model / "config.json").read_text())
    n_layer = cfg["num_hidden_layers"]
    inter = cfg["intermediate_size"]
    rope = cfg.get("rope_parameters") or cfg.get("rope_scaling") or {}
    id2label = {int(k): v for k, v in cfg["id2label"].items()}
    labels = [id2label[i] for i in range(len(id2label))]

    writer = gguf.GGUFWriter(str(args.outfile), ARCH)  # writes general.architecture

    # --- metadata -----------------------------------------------------------
    writer.add_string("general.type", "model")
    writer.add_string("general.name", args.name or args.model.resolve().name)
    writer.add_uint32("general.file_type", 1 if args.outtype == "f16" else 0)
    writer.add_uint32("general.quantization_version", 2)

    writer.add_uint32(f"{ARCH}.block_count", n_layer)
    writer.add_uint32(f"{ARCH}.context_length", cfg["max_position_embeddings"])
    writer.add_uint32(f"{ARCH}.embedding_length", cfg["hidden_size"])
    writer.add_uint32(f"{ARCH}.feed_forward_length", inter)
    writer.add_uint32(f"{ARCH}.expert_feed_forward_length", inter)
    writer.add_uint32(f"{ARCH}.attention.head_count", cfg["num_attention_heads"])
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", cfg["num_key_value_heads"])
    writer.add_uint32(f"{ARCH}.attention.key_length", cfg["head_dim"])
    writer.add_uint32(f"{ARCH}.attention.value_length", cfg["head_dim"])
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    writer.add_uint32(f"{ARCH}.attention.sliding_window", cfg["sliding_window"])
    writer.add_uint32(f"{ARCH}.expert_count", cfg["num_local_experts"])
    writer.add_uint32(f"{ARCH}.expert_used_count", cfg["num_experts_per_tok"])

    writer.add_float32(f"{ARCH}.rope.freq_base", rope["rope_theta"])
    writer.add_string(f"{ARCH}.rope.scaling.type", "yarn")
    writer.add_float32(f"{ARCH}.rope.scaling.factor", rope["factor"])
    writer.add_uint32(f"{ARCH}.rope.scaling.original_context_length",
                      rope["original_max_position_embeddings"])
    writer.add_float32(f"{ARCH}.rope.scaling.yarn_beta_fast", rope["beta_fast"])
    writer.add_float32(f"{ARCH}.rope.scaling.yarn_beta_slow", rope["beta_slow"])
    writer.add_bool(f"{ARCH}.rope.scaling.yarn_truncate", bool(rope.get("truncate", False)))

    writer.add_uint32(f"{ARCH}.pooling_type", 5)  # TOKEN_CLS
    writer.add_uint32(f"{ARCH}.embedding_length_out", len(labels))
    writer.add_array(f"{ARCH}.classifier.output_labels", labels)

    # --- tokenizer ----------------------------------------------------------
    tokens, types, merges, eos, pad = load_tokenizer(args.model)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "gpt-4o")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.token_type", types)
    writer.add_array("tokenizer.ggml.merges", merges)
    if eos is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", eos)
    if pad is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", pad)

    # --- tensors ------------------------------------------------------------
    def emit(name: str, t: "torch.Tensor"):
        is_f16 = args.outtype == "f16" and (name in F16_GLOBALS or name.endswith(F16_SUFFIXES))
        t = t.to(torch.float16 if is_f16 else torch.float32).contiguous()
        writer.add_tensor(name, t.numpy())

    n_emitted = 0
    for name, t in load_state_dict(args.model):
        if name == "model.embed_tokens.weight":
            emit("token_embd.weight", t)
        elif name == "model.norm.weight":
            emit("output_norm.weight", t)
        elif name == "score.weight":
            emit("cls.output.weight", t)
        elif name == "score.bias":
            emit("cls.output.bias", t)
        elif name.startswith("model.layers."):
            bid = int(name.split(".")[2])
            sub = name.split(".", 3)[3]
            p = f"blk.{bid}."
            simple = {
                "input_layernorm.weight": "attn_norm.weight",
                "self_attn.q_proj.weight": "attn_q.weight",
                "self_attn.q_proj.bias": "attn_q.bias",
                "self_attn.k_proj.weight": "attn_k.weight",
                "self_attn.k_proj.bias": "attn_k.bias",
                "self_attn.v_proj.weight": "attn_v.weight",
                "self_attn.v_proj.bias": "attn_v.bias",
                "self_attn.o_proj.weight": "attn_output.weight",
                "self_attn.o_proj.bias": "attn_output.bias",
                "self_attn.sinks": "attn_sinks.weight",
                "post_attention_layernorm.weight": "post_attention_norm.weight",
                "mlp.router.weight": "ffn_gate_inp.weight",
                "mlp.router.bias": "ffn_gate_inp.bias",
                "mlp.experts.down_proj_bias": "ffn_down_exps.bias",
            }
            if sub in simple:
                emit(p + simple[sub], t)
            elif sub == "mlp.experts.down_proj":
                emit(p + "ffn_down_exps.weight", t.transpose(-1, -2))
            elif sub == "mlp.experts.gate_up_proj":
                w = t.transpose(-1, -2)              # [E, 2*inter, in]
                emit(p + "ffn_gate_exps.weight", w[:, :inter, :])
                emit(p + "ffn_up_exps.weight",   w[:, inter:, :])
                n_emitted += 1                       # two tensors from one source
            elif sub == "mlp.experts.gate_up_proj_bias":
                emit(p + "ffn_gate_exps.bias", t[..., :inter])
                emit(p + "ffn_up_exps.bias",   t[..., inter:])
                n_emitted += 1
            else:
                sys.exit(f"unmapped tensor: {name}")
        else:
            sys.exit(f"unmapped tensor: {name}")
        n_emitted += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"\nwrote {args.outfile}  ({n_emitted} tensors, {args.outtype}, {len(labels)} labels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
