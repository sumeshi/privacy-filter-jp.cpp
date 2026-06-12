# privacy-filter.cpp

Minimal [GGML](https://github.com/ggml-org/ggml) inference engine for the
`openai-privacy-filter` token-classification model family
([openai/privacy-filter](https://huggingface.co/openai/privacy-filter),
[OpenMed/privacy-filter-multilingual](https://huggingface.co/OpenMed/privacy-filter-multilingual)):
PII/NER entity spans with exact UTF-8 byte offsets. Stock upstream ggml — no
patches; the model's YaRN `truncate=false` frequencies are computed at load
time and fed to `ggml_rope_ext` as `freq_factors`.

Uses the same GGUF files as the llama.cpp-based path (arch
`openai-privacy-filter`, converted by the llama.cpp-fork converter).

## Build

```sh
git clone --recursive <repo>
cmake --preset release && cmake --build --preset release -j
```

Presets: `release`, `debug` (ASan+UBSan), `profile`, `fuzz` (clang libFuzzer).
Vulkan: `-DPF_VULKAN=ON` (needs Vulkan headers/loader + glslc).

## Run

```sh
build/release/pf-cli --info model.gguf
echo "Contact John Doe at jdoe@example.com" | \
  build/release/pf-cli --classify model.gguf 0.5            # [cpu|vulkan]
```

C API in `include/pf.h`: `pf_load` / `pf_classify` (text -> entities with
byte offsets) / `pf_tokenize` / `pf_logits`. Inputs longer than the forward
window run as overlapping halo windows (exact: the halo covers the model's
full receptive field).

## Verify

```sh
ctest --preset debug -LE model            # fast suite, sanitizers, no assets
# reference fixtures (one-time, pinned env: scripts/requirements.txt):
python scripts/hf_dump.py --model <hf-model-dir> --out tests/fixtures/hf
PF_GGUF_DIR=<dir-with-ggufs> ctest --preset release          # full parity
PF_DEVICE=vulkan PF_GGUF_DIR=... ctest --preset release -L model   # on GPU
```

Measured parity (all four fixture cases, incl. a 3k-token document):
- f32 GGUF vs HF reference taps: all 91 layer taps OK, expert routing exact,
  final logits cosine 1.000000 (`scripts/compare_taps.py`).
- f16 GGUF end-to-end: argmax 100% (reference-tie carve-out), cosine >= 0.999.
- Vulkan runs at ggml's fp16 matmul precision: cosine >= 0.9985, identical
  span sets; gates widen accordingly (`PF_DEVICE`).
- Tokenizer vs HF tokenizers: 4 fixture cases + 38-text torture corpus +
  100k random differential strings — zero id/offset mismatches
  (`scripts/hf_tok_diff.py`).

## Fuzz

```sh
cmake --preset fuzz && cmake --build --preset fuzz -j
PF_GGUF=model.gguf ./build/fuzz/fuzz_tokenizer corpus_tok/
./build/fuzz/fuzz_gguf corpus_gguf/
```

`fuzz_gguf` found two upstream ggml parser crashes (hard-assert on empty KV
key; FPE on zero tensor dims in gguf.cpp:681's overflow check) — the loader
structurally pre-validates files to cover them; artifacts are pinned under
`tests/fixtures/fuzz/` and replayed by `test_loader`.

## Bench

```sh
build/release/pf-bench model.gguf [cpu|vulkan] [iters]
```

Ryzen 9 7900 (12 threads) / RTX 5070 Ti, f16 GGUF, forward tok/s by length
(one untimed warm-up per length; GPU pipelines compile lazily):

| tokens | cpu | vulkan |
|-------:|----:|-------:|
|    189 | 161 | 51 583 |
|    756 | 178 | 99 756 |
|  2 898 | 129 | 45 416 |
| 11 403 |  68 | 20 085 |
| 45 234 |  60 | 17 390 |

Weights stay in one zero-copy buffer: ~2.8 GiB RSS over baseline (f16).
