// BIOES NER decode: label table, constrained Viterbi, span assembly, and the
// halo-window geometry for inputs longer than one forward pass. Ported from
// LocalAI's llama.cpp backend (grpc-server.cpp pf_ner namespace), which is
// the behavior LocalAI ships today.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pf {

struct model;  // model.h

namespace ner {

enum bioes_tag { TAG_O = 0, TAG_B, TAG_I, TAG_E, TAG_S };

struct label_info {
    bioes_tag tag;
    int       cat;  // index into label_table::categories; -1 for O / unknown
};

struct label_table {
    std::vector<label_info>  labels;      // size n_cls, indexed by class id
    std::vector<std::string> categories;  // distinct entity-group names
    int                      o_label = 0;

    struct open_ids { int b = -1; int i = -1; };
    std::vector<open_ids> per_cat;  // per-category open-state ids (B/I)

    const std::string & category_name(int cat) const { return categories[cat]; }
};

label_table build_label_table(const std::vector<std::string> & labels);

// emit is row-major [n_tok * n_cls] LOG-probabilities. Returns the best valid
// BIOES path (one class id per token); falls back to per-token argmax if no
// valid path survives numerically.
std::vector<int> bioes_viterbi(const label_table & lt, const std::vector<float> & emit,
                               int n_tok, int n_cls);

// One assembled entity span over token indices [tok_begin, tok_end] inclusive;
// score is the mean per-token probability of the chosen labels.
struct tok_span {
    int   cat;
    int   tok_begin;
    int   tok_end;
    float score;
};

std::vector<tok_span> assemble_spans(const label_table & lt, const std::vector<int> & path,
                                     const std::vector<float> & emit, int n_cls);

// Halo-window geometry. A single window is exact when n_tok <= W; longer
// inputs slide with a halo of n_layer*radius so every interior token sees its
// full receptive field. Interior rows [lo, hi) tile [0, n_tok) exactly once.
struct window {
    int start, wlen, lo, hi;
};
std::vector<window> make_windows(int n_tok, int W, int halo);

// Windowed forward + per-row stable log_softmax into emit [n_tok * n_cls].
// Window-local positions are exact: RoPE is relative and the attention band
// symmetric, so a window starting at position 0 matches the absolute layout.
bool emit_logprobs(model & m, const int32_t * ids, int n_tok, int W,
                   std::vector<float> & emit, std::string & error);

// Full token-level pipeline: emit -> viterbi -> spans (>= threshold).
bool classify_tokens(model & m, const int32_t * ids, int n_tok, int W, float threshold,
                     std::vector<tok_span> & out, std::string & error);

} // namespace ner
} // namespace pf
