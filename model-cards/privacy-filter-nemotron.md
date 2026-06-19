---
license: apache-2.0
base_model: OpenMed/privacy-filter-nemotron
base_model_relation: quantized
pipeline_tag: token-classification
library_name: gguf
datasets:
  - nvidia/Nemotron-PII
tags:
  - gguf
  - privacy-filter.cpp
  - llama-cpp
  - localai
  - token-classification
  - pii
  - ner
  - privacy
  - redaction
  - nemotron
  - openai-privacy-filter
language:
  - en
---

# privacy-filter-nemotron — GGUF (F16 + Q8_0)

GGUF conversion of [`OpenMed/privacy-filter-nemotron`](https://huggingface.co/OpenMed/privacy-filter-nemotron),
a **fine-grained** PII **token-classification** model — a fine-tune of
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) on the
[`nvidia/Nemotron-PII`](https://huggingface.co/datasets/nvidia/Nemotron-PII) dataset. It labels
every token with a BIOES tag over **55 PII categories (221 classes)** in a single forward pass,
then decodes coherent spans with a constrained Viterbi procedure — so it can be served locally
with **no Python** as the encoder/NER tier of a PII redactor.

Where the base [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) covers 8
coarse categories, this fine-tune trades multilingual breadth for **category depth**: 55
fine-grained English categories (first/last name, government IDs, financial, healthcare,
vehicle, digital, …).

For the full model description, label space, evaluation, limitations, and citations, see the
**[source model card](https://huggingface.co/OpenMed/privacy-filter-nemotron)** — this card only
covers the GGUF packaging and how to run it.

> For broader **language** coverage (54 categories across 16 languages) instead of this model's
> English-only depth, see the multilingual fine-tune
> [`privacy-filter-multilingual` GGUF](https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF).

## Runtimes

This GGUF uses a **custom architecture, `openai-privacy-filter`**, that is not (yet) part of
upstream llama.cpp. It runs on:

1. **[privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp)** *(recommended)* —
   a small standalone GGML engine for exactly this model family, on **stock upstream ggml with
   no patches** (CPU / CUDA / Vulkan). This is the reference runtime and what the parity numbers
   below are measured against.

   ```sh
   # build (see the repo README for CUDA/Vulkan)
   cmake --preset release && cmake --build --preset release -j
   # run
   echo "Contact John Doe at jdoe@example.com" | \
     build/release/pf-cli --classify privacy-filter-nemotron-f16.gguf 0.5
   ```

   It exposes a flat C API (`pf_load` / `pf_classify` → entity spans with UTF-8 byte offsets;
   `pf_tokenize` / `pf_logits`) shaped for FFI — see the repo README.

2. **[LocalAI](https://github.com/mudler/LocalAI)** — install from the model gallery; LocalAI
   serves it behind the gRPC `TokenClassify` RPC and runs the constrained BIOES Viterbi decode,
   returning entity spans. LocalAI drives it through the **`privacy-filter` backend** (which
   wraps privacy-filter.cpp). The model is **not** a chat/completion model — it is a PII detector
   that other models opt into via a `pii.detectors` list.

3. **llama.cpp — only with a patch.** Stock `llama.cpp`, `llama-cpp-python`, Ollama, and
   LM Studio will **fail to load** this file (`unknown model architecture:
   'openai-privacy-filter'`). The arch can be added with carry-patches (TOKEN_CLS pooling, the
   architecture + HF→GGUF converter, the bidirectional banded-attention graph, and an all-SWA
   no-cache mask fix; TOKEN_CLS pooling tracks the still-open
   [PR #19725](https://github.com/ggml-org/llama.cpp/pull/19725)). Until that support lands
   upstream, `privacy-filter.cpp` above is the patch-free alternative.

> **Pooling note (llama.cpp path only):** the model must be loaded with **TOKEN_CLS pooling**
> (the GGUF's default). If you drive `llama-embedding` directly for testing, do **not** pass
> `--pooling none` — that overrides the default and yields raw hidden states instead of label
> logits. privacy-filter.cpp handles this automatically.

## Files

| File | Precision | Size | Notes |
|---|---|---|---|
| `privacy-filter-nemotron-f16.gguf` | F16 | 2.82 GB | Reference artifact. 156 tensors; 221 `classifier.output_labels`; `pooling_type = TOKEN_CLS`. |
| `privacy-filter-nemotron-q8.gguf` | Q8_0 (experts) | 1.64 GB | MoE expert weights → Q8_0, the rest F16. For RAM-constrained / edge use. |

```
sha256 (f16): 70dfe91ff220ff04594168a83e296dcc2054449cde77f98d0e782edbb6a31f5a
sha256 (q8):  2ec11c154e572a2686f4d77e861b7f74e6917e09638fe9bd27156d48bd99e21a
```

**Q8_0 quantization — and why it isn't free.** `q8` stores the bulk of the weights (the MoE
expert matrices) as 8-bit integers instead of 16-bit floats — via
[`scripts/requant_q8.py`](https://github.com/localai-org/privacy-filter.cpp/blob/master/scripts/requant_q8.py),
with attention, embeddings and the classifier head left at F16. That cuts the download by ~42%
(2.82 GB → 1.64 GB) and is usually a bit faster on CPU.

The catch: **reducing precision throws information away, and it is almost never a free lunch.**
On a mixed-PII document (1,360 tokens) q8 matched f16 on **99.93%** of token labels (1,359/1,360)
and produced an **identical span set** at threshold 0.5, with an average prediction shift (KL
divergence) of just **2.6e-5** — but note it did **not** match on all tokens; one token flipped.
That is the point in miniature: a reassuring average still hides the specific cases that change,
and **accuracy benchmarks routinely look fine right up until the one that bites.** Those numbers
also come from a single English document; a tiny *average* shift can still hide a flip on the one
input that matters to you — a rare name, an unusual ID format, or one of the fuzzier categories
below. For PII detection a single missed span is a leak, so:

- **Prefer F16** if you can afford the 2.82 GB — it is the reference these numbers are measured
  against, and what we trust by default.
- **Use Q8_0** when memory or speed forces it (e.g. a 4 GB Raspberry Pi 5), treat it as a
  deliberate size/speed tradeoff, and **validate it on your own data** first.

## Architecture & conversion

gpt-oss-style sparse **MoE** (8 layers, `d_model=640`, 128 experts, top-4 routing; ~1.5B total /
~50M active per token), **bidirectional banded attention** (symmetric sliding window 128,
attention sinks retained), **interleaved (GPT-J) RoPE** with YaRN (θ=150000, factor 32), o200k
(`o200k_base`) tokenizer, and a 221-way token-classification head (`score` → `cls.output`). The
architecture is identical to the rest of the `privacy-filter` family — only the fine-tuned
weights and the larger (221-class) head differ.

The conversion reproduces the unmodified `transformers` reference at F16: across the parity
prompt set (short / PII-dense / multilingual / a 3k-token document) the F16 GGUF agrees with HF
on **99.94%** of per-token argmaxes — **100% up to ~300 tokens**, with the only two flips being
argmax *ties* at ~3k positions (the F16-rounding regime) — at **full-logit cosine ≥ 0.9995
(mean 0.999997)**. A wrong expert transpose would crater that cosine, so the two load-bearing
conversion choices — the expert `gate_up` `chunk(2)` split and the `n_swa = 2·sliding_window`
window mapping — are confirmed by it. privacy-filter.cpp re-derives the YaRN `truncate=false`
frequencies at load time (fed to `ggml_rope_ext` as `freq_factors`) so the same GGUF is
interchangeable across runtimes.

This GGUF was produced by [`scripts/convert.py`](https://github.com/localai-org/privacy-filter.cpp/blob/master/scripts/convert.py)
— a self-contained HF→GGUF converter (no llama.cpp dependency). The same converter is re-run by
CI and gated against the HF reference logits for the sibling models, so the published artifact
stays in parity.

## Label space

`O` plus `B-`/`I-`/`E-`/`S-` for each of 55 categories (1 + 55×4 = 221), spanning identity,
contact, address, dates/time, government IDs, financial, healthcare, enterprise IDs, vehicle, and
digital entities:

- **Identity** — `first_name`, `last_name`, `user_name`, `age`, `gender`, `race_ethnicity`,
  `sexuality`, `religious_belief`, `political_view`, `education_level`, `occupation`,
  `employment_status`, `language`, `blood_type`, `biometric_identifier`
- **Contact** — `email`, `phone_number`, `fax_number`, `url`
- **Address** — `street_address`, `city`, `county`, `state`, `country`, `postcode`, `coordinate`
- **Dates** — `date`, `date_of_birth`, `date_time`, `time`
- **Government IDs** — `ssn`, `national_id`, `tax_id`
- **Financial** — `account_number`, `bank_routing_number`, `swift_bic`, `credit_debit_card`,
  `cvv`, `pin`, `password`
- **Healthcare** — `medical_record_number`, `health_plan_beneficiary_number`
- **Enterprise** — `company_name`, `customer_id`, `employee_id`, `unique_id`,
  `certificate_license_number`
- **Vehicle** — `license_plate`, `vehicle_identifier`
- **Digital** — `ipv4`, `ipv6`, `mac_address`, `device_identifier`, `api_key`, `http_cookie`

The ordered `id2label` table is embedded in the GGUF (`classifier.output_labels`). See the
[source card](https://huggingface.co/OpenMed/privacy-filter-nemotron#label-space-55-categories)
for the canonical grouping.

## Evaluation

From the [source model card](https://huggingface.co/OpenMed/privacy-filter-nemotron#performance)
(on the Nemotron-PII test split): **macro B-F1 = 0.9533**, **token accuracy = 0.9910**, with
**46/55** labels at F1 ≥ 0.90, **7/55** in 0.70–0.89, and **none** below 0.70. The fuzzier,
more subjective categories (`occupation`, `language`, `gender`, `state`, `race_ethnicity`,
`political_view`, `education_level`) sit lowest (F1 ≈ 0.65–0.89) vs the strictly-formatted
identifiers (≥ 0.95). The GGUF reproduces the HF logits at F16, so these numbers carry over.

## Limitations & intended use

Identical to the [source model](https://huggingface.co/OpenMed/privacy-filter-nemotron#limitations--intended-use):

- **English-only.** Nemotron-PII is predominantly English (50/50 US/international locale split);
  non-English performance is not guaranteed. For multilingual text, prefer the
  [multilingual fine-tune](https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF).
- **Synthetic training data.** Nemotron-PII is synthesized; real clinical notes, legal documents,
  and web text may show different surface forms — collect a domain eval set and re-calibrate
  thresholds for high-stakes use.
- **Fuzzier categories are weaker.** Treat low-confidence predictions on the subjective
  categories above accordingly.
- **Not a substitute for legal/compliance review**, and **not** a clinical PHI model. Use it as
  one tier behind deterministic regex pre-filters and human review.

## License

**Apache-2.0**, inherited from `openai/privacy-filter` and `OpenMed/privacy-filter-nemotron`.

## Credits & citation

Conversion and runtime support by the **LocalAI** project (`privacy-filter.cpp`). The model
itself is by **OpenMed**, fine-tuned from **OpenAI**'s `privacy-filter` on **NVIDIA**'s
**Nemotron-PII** dataset — please cite all of them (BibTeX in the
[source card](https://huggingface.co/OpenMed/privacy-filter-nemotron#citation)).
