// o200k_base byte-level BPE tokenizer with exact UTF-8 byte offsets.
//
// Scope: exactly what this model needs — the single o200k pre-tokenizer
// regex (hand-compiled scanner, oniguruma leftmost-alternative semantics),
// ByteLevel with use_regex=false / add_prefix_space=false (BPE runs on the
// pre-token's raw bytes; offsets are exact by construction), BPE with
// ignore_merges=true. No normalizer. No special-token parsing: a literal
// "<|endoftext|>" in input is plain text (deliberate for a PII filter; the
// HF differential harness sets split_special_tokens accordingly).
// Vocab and merges come from the GGUF (tokenizer.ggml.{tokens,merges},
// GPT-2 byte-unicode form, decoded to raw bytes at load).
#pragma once

#include <gguf.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pf {

struct token_span {
    int32_t id;
    int32_t start, end;  // byte offsets into the input text
};

// Pre-tokenizer: byte ranges of each pre-token (exposed for unit tests).
std::vector<std::pair<int32_t, int32_t>> pretokenize(const char * text, size_t len);

struct tokenizer {
    std::unordered_map<std::string, int32_t> vocab;       // raw bytes -> id
    std::unordered_map<std::string, int32_t> merge_rank;  // pair_key(a,b) -> rank
    std::string error;

    bool load(const gguf_context * g);
    std::vector<token_span> encode(const char * text, size_t len) const;

    static std::string pair_key(const std::string & a, const std::string & b) {
        return std::to_string(a.size()) + ":" + a + b;
    }
};

} // namespace pf
