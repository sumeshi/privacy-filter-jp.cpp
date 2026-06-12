// Synthetic BIOES Viterbi / span-assembly / window-geometry tests (fast).
#include "ner.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace pf::ner;

static int failures = 0;

#define CHECK_MSG(cond, ...) do { \
    if (!(cond)) { failures++; std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } \
} while (0)

// label set: O, B-X, I-X, E-X, S-X, B-Y, I-Y, E-Y, S-Y
static const std::vector<std::string> LABELS =
    { "O", "B-X", "I-X", "E-X", "S-X", "B-Y", "I-Y", "E-Y", "S-Y" };
enum { O = 0, BX, IX, EX, SX, BY, IY, EY, SY, NC };

// emit log-probs: every row mildly prefers O, then overrides push the path
static std::vector<float> base_emit(int n_tok) {
    std::vector<float> e((size_t) n_tok * NC, -8.0f);
    for (int t = 0; t < n_tok; t++) e[(size_t) t * NC + O] = -0.5f;
    return e;
}

static void set(std::vector<float> & e, int t, int c, float v) { e[(size_t) t * NC + c] = v; }

static void test_label_table() {
    label_table lt = build_label_table(LABELS);
    CHECK_MSG(lt.o_label == O, "o_label %d", lt.o_label);
    CHECK_MSG(lt.categories.size() == 2, "categories %zu", lt.categories.size());
    CHECK_MSG(lt.labels[BX].tag == TAG_B && lt.labels[BX].cat == 0, "B-X parse");
    CHECK_MSG(lt.labels[IY].tag == TAG_I && lt.labels[IY].cat == 1, "I-Y parse");
    CHECK_MSG(lt.per_cat[0].b == BX && lt.per_cat[0].i == IX, "per_cat X");
}

static void test_invalid_transition_suppressed() {
    // I-X right after O scores higher than anything else, but is invalid;
    // the path must not contain it.
    label_table lt = build_label_table(LABELS);
    auto e = base_emit(3);
    set(e, 1, IX, 2.0f);  // invalid continuation: nothing opened X at t=0
    auto path = bioes_viterbi(lt, e, 3, NC);
    CHECK_MSG(path[1] != IX, "invalid I-X after O chosen");
    CHECK_MSG(path[0] == O && path[2] == O, "path corrupted: %d %d", path[0], path[2]);
}

static void test_dangling_b_rejected() {
    // B-X at the last token cannot close: terminate-on-closed must reject it.
    label_table lt = build_label_table(LABELS);
    auto e = base_emit(2);
    set(e, 1, BX, 3.0f);  // tempting open at the end
    auto path = bioes_viterbi(lt, e, 2, NC);
    CHECK_MSG(path[1] != BX && path[1] != IX, "dangling open state at end: %d", path[1]);
}

static void test_full_span_and_scores() {
    // B-X I-X E-X with high scores; then S-Y standalone
    label_table lt = build_label_table(LABELS);
    auto e = base_emit(6);
    set(e, 1, BX, 1.0f);
    set(e, 2, IX, 1.0f);
    set(e, 3, EX, 1.0f);
    set(e, 5, SY, 2.0f);
    auto path = bioes_viterbi(lt, e, 6, NC);
    CHECK_MSG(path[1] == BX && path[2] == IX && path[3] == EX, "span path %d %d %d",
              path[1], path[2], path[3]);
    CHECK_MSG(path[5] == SY, "S-Y not chosen: %d", path[5]);

    auto spans = assemble_spans(lt, path, e, NC);
    CHECK_MSG(spans.size() == 2, "spans %zu", spans.size());
    if (spans.size() == 2) {
        CHECK_MSG(spans[0].cat == 0 && spans[0].tok_begin == 1 && spans[0].tok_end == 3, "span0");
        const float want = (float) std::exp(1.0);  // mean of three equal probs
        CHECK_MSG(std::fabs(spans[0].score - want) < 1e-5, "score %.5f want %.5f",
                  spans[0].score, want);
        CHECK_MSG(spans[1].cat == 1 && spans[1].tok_begin == 5 && spans[1].tok_end == 5, "span1");
    }
}

static void test_single_token() {
    label_table lt = build_label_table(LABELS);
    auto e = base_emit(1);
    set(e, 0, SX, 1.0f);   // S allowed standalone
    set(e, 0, BX, 2.0f);   // B would score higher but cannot close
    auto path = bioes_viterbi(lt, e, 1, NC);
    CHECK_MSG(path[0] == SX, "single token path %d", path[0]);
}

static void test_window_geometry() {
    // every token index covered exactly once by [start+lo, start+hi)
    for (int halo : { 1, 7, 1024 }) {
        for (int W : { 2 * halo + 1, 2 * halo + 13, 4096 }) {
            for (int n : { 1, W - 1, W, W + 1, 2 * W, 3 * W + 17 }) {
                auto ws = make_windows(n, W, halo);
                std::vector<int> cover(n, 0);
                for (const auto & w : ws) {
                    CHECK_MSG(w.wlen <= W, "wlen %d > W %d", w.wlen, W);
                    CHECK_MSG(w.start + w.wlen <= n, "window past end");
                    for (int i = w.lo; i < w.hi; i++) cover[w.start + i]++;
                }
                for (int i = 0; i < n; i++) {
                    CHECK_MSG(cover[i] == 1, "n=%d W=%d halo=%d: token %d covered %d times",
                              n, W, halo, i, cover[i]);
                }
            }
        }
    }
}

int main() {
    test_label_table();
    test_invalid_transition_suppressed();
    test_dangling_b_rejected();
    test_full_span_and_scores();
    test_single_token();
    test_window_geometry();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
