// Window-stitch equivalence (label: model): the long-3k case forced through
// multiple halo windows must reproduce the single-pass result.
//
// The halo (n_layer * radius) covers the full receptive field, so stitching
// is STRUCTURALLY exact; but windows restart positions at 0 (same as the
// LocalAI backend), and while RoPE attention is position-shift invariant in
// exact arithmetic, its f32 phase noise is not — tail log-probs (~ -15 nats)
// swing visibly. The decision-relevant gates: identical per-token argmax,
// identical spans, argmax-label log-prob within 0.05 nats, span scores
// within 0.02.
#include "model.h"
#include "ner.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK_MSG(cond, ...) do { \
    if (!(cond)) { failures++; std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } \
} while (0)

static bool read_i32(const std::string & path, std::vector<int32_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz / sizeof(int32_t));
    size_t got = std::fread(out.data(), sizeof(int32_t), out.size(), f);
    std::fclose(f);
    return got == out.size();
}

int main() {
    const char * gguf_dir = std::getenv("PF_GGUF_DIR");
    const char * fixtures = std::getenv("PF_FIXTURES");
    if (!gguf_dir || !fixtures) {
        std::fprintf(stderr, "PF_GGUF_DIR or PF_FIXTURES not set, skipping\n");
        return 77;
    }
    std::vector<int32_t> ids;
    if (!read_i32(std::string(fixtures) + "/long-3k/tokens.i32", ids)) {
        std::fprintf(stderr, "long-3k fixture missing, skipping\n");
        return 77;
    }
    const int n = (int) ids.size();

    const char * f16_name = std::getenv("PF_GGUF_NAME");
    pf::model m;
    if (!m.load(std::string(gguf_dir) + "/" + (f16_name ? f16_name : "pf-rope2-f16.gguf"), "cpu", 0)) {
        std::fprintf(stderr, "load: %s\n", m.error.c_str());
        return 1;
    }
    const int n_cls = m.hp().n_cls;
    const int halo = m.hp().n_layer * m.hp().swa_radius;  // 1024

    std::string err;
    std::vector<float> single, windowed;
    if (!pf::ner::emit_logprobs(m, ids.data(), n, /*W =*/ 4096, single, err)) {
        std::fprintf(stderr, "single-pass: %s\n", err.c_str());
        return 1;
    }
    const int W = 2560;  // stride = 2560 - 2*1024 = 512 -> 2 windows over 3053
    {
        auto ws = pf::ner::make_windows(n, W, halo);
        CHECK_MSG(ws.size() >= 2, "expected multiple windows, got %zu", ws.size());
    }
    if (!pf::ner::emit_logprobs(m, ids.data(), n, W, windowed, err)) {
        std::fprintf(stderr, "windowed: %s\n", err.c_str());
        return 1;
    }

    double max_d_top = 0.0;
    int argmax_diff = 0;
    for (int t = 0; t < n; t++) {
        int a = 0, b = 0;
        for (int c = 0; c < n_cls; c++) {
            const size_t i = (size_t) t * n_cls + c;
            if (single[i] > single[(size_t) t * n_cls + a]) a = c;
            if (windowed[i] > windowed[(size_t) t * n_cls + b]) b = c;
        }
        argmax_diff += a != b;
        max_d_top = std::max(max_d_top, (double) std::fabs(
            single[(size_t) t * n_cls + a] - windowed[(size_t) t * n_cls + a]));
    }
    std::printf("argmax-label max |dlogprob| = %.2e, argmax diffs = %d/%d\n",
                max_d_top, argmax_diff, n);
    CHECK_MSG(argmax_diff == 0, "argmax differs across stitching");
    CHECK_MSG(max_d_top < 0.05, "argmax-label log-prob delta %.2e >= 0.05", max_d_top);

    // span-level equivalence through the full decode
    const pf::ner::label_table lt = pf::ner::build_label_table(m.file.labels);
    auto spans_a = pf::ner::assemble_spans(lt, pf::ner::bioes_viterbi(lt, single, n, n_cls), single, n_cls);
    auto spans_b = pf::ner::assemble_spans(lt, pf::ner::bioes_viterbi(lt, windowed, n, n_cls), windowed, n_cls);
    CHECK_MSG(spans_a.size() == spans_b.size(), "span count %zu vs %zu", spans_a.size(), spans_b.size());
    for (size_t i = 0; i < std::min(spans_a.size(), spans_b.size()); i++) {
        CHECK_MSG(spans_a[i].cat == spans_b[i].cat &&
                  spans_a[i].tok_begin == spans_b[i].tok_begin &&
                  spans_a[i].tok_end == spans_b[i].tok_end,
                  "span %zu differs: [%d,%d] cat %d vs [%d,%d] cat %d", i,
                  spans_a[i].tok_begin, spans_a[i].tok_end, spans_a[i].cat,
                  spans_b[i].tok_begin, spans_b[i].tok_end, spans_b[i].cat);
        CHECK_MSG(std::fabs(spans_a[i].score - spans_b[i].score) < 0.02,
                  "span %zu score %.4f vs %.4f", i, spans_a[i].score, spans_b[i].score);
    }
    std::printf("spans: %zu (identical)\n", spans_a.size());

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
