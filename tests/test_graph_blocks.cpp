// Fast block tests (no model assets): pin the pieces of the forward pass
// that can silently drift — YaRN frequencies vs HF golden values, the SWA
// mask band edge, ggml swiglu_oai semantics, and the rope configuration
// (interleaved pairing, freq_factors division, attn_factor application).
#include "model.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;

#define CHECK_MSG(cond, ...) do { \
    if (!(cond)) { failures++; std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } \
} while (0)

static pf::hparams model_hparams() {
    pf::hparams h;
    h.n_layer = 8;   h.n_embd = 640; h.n_head = 14; h.n_head_kv = 2; h.n_rot = 64;
    h.n_ff_exp = 640; h.n_expert = 128; h.n_expert_used = 4;
    h.swa_radius = 128; h.n_ctx_train = 131072; h.n_ctx_orig = 4096; h.n_cls = 217;
    h.rms_eps = 1e-5f; h.rope_freq_base = 150000.0f;
    h.yarn_factor = 32.0f; h.yarn_beta_fast = 32.0f; h.yarn_beta_slow = 1.0f;
    h.yarn_truncate = false;
    return h;
}

// torch.float64 inv_freq from the HF rotary module (transformers 5.9.0,
// _compute_yarn_parameters with truncate=False), attention_scaling below.
static const double GOLDEN_INV_FREQ[32] = {
    1.0000000000e+00, 6.8904429674e-01, 4.7478204966e-01, 3.2714587450e-01,
    2.2541800141e-01, 1.5532298386e-01, 1.0702442378e-01, 7.3744565248e-02,
    5.0813272595e-02, 3.1705696136e-02, 1.9334999844e-02, 1.1592049152e-02,
    6.7949593067e-03, 3.8603590801e-03, 2.0937926602e-03, 1.0526021942e-03,
    4.5648391824e-04, 1.2931869423e-04, 3.8308811781e-05, 2.6396468456e-05,
    1.8188336981e-05, 1.2532569599e-05, 8.6354957602e-06, 5.9502394834e-06,
    4.0999784687e-06, 2.8250667583e-06, 1.9465962851e-06, 1.3412909539e-06,
    9.2420896181e-07, 6.3682091422e-07, 4.3879785494e-07, 3.0235113968e-07,
};
static const double GOLDEN_ATTN_SCALING = 1.3465735902799727;

static void test_yarn_freq_factors() {
    pf::hparams h = model_hparams();
    std::vector<float> ff;
    float attn_factor = 0.f;
    pf::yarn_freq_factors(h, ff, attn_factor);

    CHECK_MSG(ff.size() == 32, "ff size %zu", ff.size());
    CHECK_MSG(std::fabs(attn_factor - GOLDEN_ATTN_SCALING) < 1e-6,
              "attn_factor %.9f vs %.9f", attn_factor, GOLDEN_ATTN_SCALING);
    for (int j = 0; j < 32; j++) {
        const double extrap   = std::pow((double) h.rope_freq_base, -2.0 * j / h.n_rot);
        const double inv_freq = extrap / ff[j];
        const double rel = std::fabs(inv_freq - GOLDEN_INV_FREQ[j]) / GOLDEN_INV_FREQ[j];
        CHECK_MSG(rel < 1e-6, "inv_freq[%d] %.10e vs golden %.10e (rel %.2e)",
                  j, inv_freq, GOLDEN_INV_FREQ[j], rel);
    }
}

static void test_swa_mask() {
    const int64_t radius = 128;
    for (int64_t n : {1, 2, 255, 256, 257, 300}) {
        std::vector<float> m(n * n);
        pf::fill_swa_mask(m.data(), n, radius);
        for (int64_t p1 = 0; p1 < n; p1++) {
            for (int64_t p0 = 0; p0 < n; p0++) {
                const bool visible = std::llabs(p1 - p0) <= radius;  // HF predicate
                const float got = m[p1 * n + p0];
                CHECK_MSG(visible ? got == 0.0f : std::isinf(got),
                          "mask(n=%lld, p1=%lld, p0=%lld) = %f", (long long) n,
                          (long long) p1, (long long) p0, got);
            }
        }
    }
}

// tiny single-op CPU eval helper. Uses the backend API rather than ggml-cpu's
// ggml_graph_compute_with_ctx so it links in a GGML_BACKEND_DL build (where the
// CPU compute symbols live in the variant .so, loaded at runtime). Tensors are
// in a no_alloc=false ctx, so their ->data is CPU-resident and computed in place.
static ggml_backend_t cpu_backend() {
    static ggml_backend_t be = [] {
        ggml_backend_load_all();
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }();
    return be;
}

template <typename BUILD>
static ggml_tensor * eval(ggml_context * ctx, BUILD build) {
    ggml_tensor * out = build();
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(cpu_backend(), gf);
    return out;
}

static void test_swiglu_oai() {
    ggml_init_params p = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(p);

    const int n = 256;
    ggml_tensor * gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * up   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-10.f, 10.f);
    for (int i = 0; i < n; i++) {
        ((float *) gate->data)[i] = d(rng);
        ((float *) up->data)[i]   = d(rng);
    }

    ggml_tensor * out = eval(ctx, [&] { return ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f); });

    for (int i = 0; i < n; i++) {
        // HF OpenAIPrivacyFilterExperts._apply_gate:
        //   gate = min(gate, 7); up = clamp(up, -7, 7)
        //   out  = (up + 1) * gate * sigmoid(1.702 * gate)
        const double g = std::min((double) ((float *) gate->data)[i], 7.0);
        const double u = std::min(std::max((double) ((float *) up->data)[i], -7.0), 7.0);
        const double want = (u + 1.0) * g / (1.0 + std::exp(-1.702 * g));
        const double got = ((float *) out->data)[i];
        CHECK_MSG(std::fabs(got - want) < 1e-4, "swiglu_oai[%d] got %.6f want %.6f", i, got, want);
    }
    ggml_free(ctx);
}

static void test_rope() {
    pf::hparams h = model_hparams();
    std::vector<float> ffv;
    float attn_factor = 0.f;
    pf::yarn_freq_factors(h, ffv, attn_factor);

    ggml_init_params p = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(p);

    const std::vector<int32_t> positions = { 0, 1, 100, 2047, 4095 };
    const int n = (int) positions.size();

    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, n);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_tensor * ff  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-2.f, 2.f);
    for (int i = 0; i < 64 * n; i++) ((float *) x->data)[i] = d(rng);
    std::memcpy(pos->data, positions.data(), n * sizeof(int32_t));
    std::memcpy(ff->data, ffv.data(), 32 * sizeof(float));

    ggml_tensor * out = eval(ctx, [&] {
        return ggml_rope_ext(ctx, x, pos, ff, 64, /*mode =*/ 0, h.n_ctx_orig,
                             h.rope_freq_base, 1.0f, 0.0f, attn_factor,
                             h.yarn_beta_fast, h.yarn_beta_slow);
    });

    for (int t = 0; t < n; t++) {
        for (int j = 0; j < 32; j++) {
            // exact reference: interleaved pair (2j, 2j+1) rotated by
            // pos * inv_freq[j] (float64), scaled by attention_scaling
            const double theta = (double) positions[t] * GOLDEN_INV_FREQ[j];
            const double c = std::cos(theta) * GOLDEN_ATTN_SCALING;
            const double s = std::sin(theta) * GOLDEN_ATTN_SCALING;
            const double x0 = ((float *) x->data)[t * 64 + 2 * j];
            const double x1 = ((float *) x->data)[t * 64 + 2 * j + 1];
            const double w0 = x0 * c - x1 * s;
            const double w1 = x1 * c + x0 * s;
            const double g0 = ((float *) out->data)[t * 64 + 2 * j];
            const double g1 = ((float *) out->data)[t * 64 + 2 * j + 1];
            // f32 phase noise grows with |theta|: |dtheta| ~ theta * k * eps
            const double tol = 1e-5 + 4e-7 * std::fabs(theta) * 2.7;  // 2.7 > attn_scaling * |x|max
            CHECK_MSG(std::fabs(g0 - w0) < tol && std::fabs(g1 - w1) < tol,
                      "rope pos %d pair %d: got (%.6f, %.6f) want (%.6f, %.6f) tol %.1e",
                      positions[t], j, g0, g1, w0, w1, tol);
        }
    }
    ggml_free(ctx);
}

int main() {
    test_yarn_freq_factors();
    test_swa_mask();
    test_swiglu_oai();
    test_rope();
    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
