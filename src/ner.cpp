#include "ner.h"

#include "model.h"

#include <cmath>
#include <limits>
#include <map>

namespace pf {
namespace ner {

// Split a "B-CATEGORY" label into its BIOES tag and category name. The model's
// labels use a single '-' separator and category names contain none.
label_table build_label_table(const std::vector<std::string> & labels) {
    label_table t;
    const size_t n = labels.size();
    t.labels.resize(n, { TAG_O, -1 });
    std::map<std::string, int> cat_index;
    bool found_o = false;
    for (size_t i = 0; i < n; i++) {
        const std::string & s = labels[i];
        if (s.empty() || s == "O") {
            t.labels[i] = { TAG_O, -1 };
            if (!found_o) { t.o_label = (int) i; found_o = true; }
            continue;
        }
        bioes_tag tag;
        switch (s[0]) {
            case 'B': tag = TAG_B; break;
            case 'I': tag = TAG_I; break;
            case 'E': tag = TAG_E; break;
            case 'S': tag = TAG_S; break;
            default:  t.labels[i] = { TAG_O, -1 }; continue;  // unknown -> O
        }
        const size_t dash = s.find('-');
        const std::string cat = (dash == std::string::npos) ? s : s.substr(dash + 1);
        int ci;
        auto it = cat_index.find(cat);
        if (it == cat_index.end()) {
            ci = (int) t.categories.size();
            cat_index.emplace(cat, ci);
            t.categories.push_back(cat);
        } else {
            ci = it->second;
        }
        t.labels[i] = { tag, ci };
    }
    t.per_cat.assign(t.categories.size(), {});
    for (size_t i = 0; i < n; i++) {
        const auto & li = t.labels[i];
        if (li.cat < 0) continue;
        if (li.tag == TAG_B) t.per_cat[li.cat].b = (int) i;
        if (li.tag == TAG_I) t.per_cat[li.cat].i = (int) i;
    }
    return t;
}

static inline bool tag_is_closed(bioes_tag tg) { return tg == TAG_O || tg == TAG_E || tg == TAG_S; }

// Constrained linear-chain Viterbi over BIOES. Exploits the BIOES structure so
// each step is O(n_cls): a fresh label (O/B/S) may only follow a closed state
// (O/E/S) and takes the single best closed predecessor; a continuation (I/E of
// category c) may only follow B-c or I-c.
std::vector<int> bioes_viterbi(const label_table & lt, const std::vector<float> & emit,
                               int n_tok, int n_cls) {
    const float NEG = -std::numeric_limits<float>::infinity();
    std::vector<float> prev_dp(n_cls, NEG), dp(n_cls, NEG);
    std::vector<int>   bp((size_t) n_tok * n_cls, -1);

    // t == 0: a span may only start with O / B / S.
    for (int j = 0; j < n_cls; j++) {
        const bioes_tag tg = lt.labels[j].tag;
        if (tg == TAG_O || tg == TAG_B || tg == TAG_S) prev_dp[j] = emit[j];
    }

    for (int t = 1; t < n_tok; t++) {
        std::fill(dp.begin(), dp.end(), NEG);
        const float * e = &emit[(size_t) t * n_cls];

        float best_closed = NEG;
        int   best_closed_arg = -1;
        for (int i = 0; i < n_cls; i++) {
            if (prev_dp[i] == NEG) continue;
            if (tag_is_closed(lt.labels[i].tag) && prev_dp[i] > best_closed) {
                best_closed = prev_dp[i];
                best_closed_arg = i;
            }
        }

        for (int j = 0; j < n_cls; j++) {
            const auto & lj = lt.labels[j];
            float pred = NEG;
            int   arg = -1;
            if (lj.tag == TAG_O || lj.tag == TAG_B || lj.tag == TAG_S) {
                pred = best_closed; arg = best_closed_arg;  // fresh start
            } else {
                // I-c or E-c: predecessor must be B-c or I-c
                const auto & oc = lt.per_cat[lj.cat];
                if (oc.b >= 0 && prev_dp[oc.b] > pred) { pred = prev_dp[oc.b]; arg = oc.b; }
                if (oc.i >= 0 && prev_dp[oc.i] > pred) { pred = prev_dp[oc.i]; arg = oc.i; }
            }
            if (arg >= 0 && pred != NEG) {
                dp[j] = pred + e[j];
                bp[(size_t) t * n_cls + j] = arg;
            }
        }
        prev_dp.swap(dp);
    }

    // terminate only on a closed state (no dangling B/I span)
    float best = NEG;
    int   arg = -1;
    for (int j = 0; j < n_cls; j++) {
        if (prev_dp[j] == NEG) continue;
        if (tag_is_closed(lt.labels[j].tag) && prev_dp[j] > best) { best = prev_dp[j]; arg = j; }
    }

    std::vector<int> path(n_tok, lt.o_label);
    if (arg < 0) {
        // safety net: the all-O path always exists, but guard numerically
        for (int t = 0; t < n_tok; t++) {
            const float * e = &emit[(size_t) t * n_cls];
            int a = 0;
            float m = e[0];
            for (int j = 1; j < n_cls; j++) if (e[j] > m) { m = e[j]; a = j; }
            path[t] = a;
        }
        return path;
    }
    int cur = arg;
    for (int t = n_tok - 1; t >= 0; t--) {
        path[t] = cur;
        if (t > 0) cur = bp[(size_t) t * n_cls + cur];
    }
    return path;
}

// Walk a (valid) BIOES label path into spans. Viterbi guarantees validity, so
// B is always closed by a matching E and S stands alone.
std::vector<tok_span> assemble_spans(const label_table & lt, const std::vector<int> & path,
                                     const std::vector<float> & emit, int n_cls) {
    std::vector<tok_span> out;
    const int n_tok = (int) path.size();
    int begin = -1, cat = -1;
    double prob_sum = 0.0;
    auto prob_at = [&](int t) {
        return (double) std::exp(emit[(size_t) t * n_cls + path[t]]);
    };
    for (int t = 0; t < n_tok; t++) {
        const auto & li = lt.labels[path[t]];
        switch (li.tag) {
            case TAG_S:
                out.push_back({ li.cat, t, t, (float) prob_at(t) });
                begin = -1;
                break;
            case TAG_B:
                begin = t; cat = li.cat; prob_sum = prob_at(t);
                break;
            case TAG_I:
                if (begin >= 0 && li.cat == cat) prob_sum += prob_at(t);
                break;
            case TAG_E:
                if (begin >= 0 && li.cat == cat) {
                    prob_sum += prob_at(t);
                    const int len = t - begin + 1;
                    out.push_back({ cat, begin, t, (float) (prob_sum / len) });
                }
                begin = -1;
                break;
            case TAG_O:
            default:
                begin = -1;
                break;
        }
    }
    return out;
}

std::vector<window> make_windows(int n_tok, int W, int halo) {
    std::vector<window> out;
    if (n_tok <= W) {
        out.push_back({ 0, n_tok, 0, n_tok });
        return out;
    }
    const int stride = W - 2 * halo;
    for (int start = 0; start < n_tok; start += stride) {
        const int wlen = std::min(W, n_tok - start);
        const int lo = (start == 0) ? 0 : halo;
        const int hi = (start + wlen >= n_tok) ? wlen : (wlen - halo);
        out.push_back({ start, wlen, lo, hi });
        if (start + wlen >= n_tok) break;
    }
    return out;
}

bool emit_logprobs(model & m, const int32_t * ids, int n_tok, int W,
                   std::vector<float> & emit, std::string & error) {
    const int n_cls = m.hp().n_cls;
    const int halo = m.hp().n_layer * m.hp().swa_radius;
    if (n_tok > W && W <= 2 * halo) {
        error = "input (" + std::to_string(n_tok) + " tokens) exceeds the single-forward window (" +
                std::to_string(W) + ") and exact windowing needs a window > " + std::to_string(2 * halo);
        return false;
    }

    emit.assign((size_t) n_tok * n_cls, 0.0f);
    std::vector<float> logits;
    for (const window & w : make_windows(n_tok, W, halo)) {
        if (!m.forward(ids + w.start, w.wlen, logits)) {
            error = m.error;
            return false;
        }
        for (int li = w.lo; li < w.hi; li++) {
            const float * row = logits.data() + (size_t) li * n_cls;
            // log_softmax (fp32 in, f64 accumulate, max-subtraction stable)
            float maxv = row[0];
            for (int c = 1; c < n_cls; c++) maxv = std::max(maxv, row[c]);
            double sum = 0.0;
            for (int c = 0; c < n_cls; c++) sum += std::exp((double) (row[c] - maxv));
            const double logsum = std::log(sum);
            float * dst = &emit[(size_t) (w.start + li) * n_cls];
            for (int c = 0; c < n_cls; c++) {
                dst[c] = (float) ((double) (row[c] - maxv) - logsum);
            }
        }
    }
    return true;
}

bool classify_tokens(model & m, const int32_t * ids, int n_tok, int W, float threshold,
                     std::vector<tok_span> & out, std::string & error) {
    out.clear();
    if (n_tok == 0) return true;
    std::vector<float> emit;
    if (!emit_logprobs(m, ids, n_tok, W, emit, error)) {
        return false;
    }
    const label_table lt = build_label_table(m.file.labels);
    const std::vector<int> path = bioes_viterbi(lt, emit, n_tok, m.hp().n_cls);
    for (const tok_span & sp : assemble_spans(lt, path, emit, m.hp().n_cls)) {
        if (sp.score >= threshold) out.push_back(sp);
    }
    return true;
}

} // namespace ner
} // namespace pf
