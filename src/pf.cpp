#include "pf.h"

#include "model.h"
#include "ner.h"

#include <cstdlib>
#include <cstring>
#include <string>

struct pf_ctx {
    pf::model   m;
    int32_t     max_forward_tokens = 4096;
    std::string error;
};

int pf_abi_version(void) { return PF_ABI_VERSION; }

pf_ctx * pf_load(const char * gguf_path, const char * device, int n_threads) {
    if (!gguf_path) return nullptr;
    auto * ctx = new pf_ctx();
    if (!ctx->m.load(gguf_path, device ? device : "cpu", n_threads)) {
        ctx->error = ctx->m.error;
    }
    return ctx;
}

void pf_free(pf_ctx * ctx) { delete ctx; }

const char * pf_last_error(const pf_ctx * ctx) {
    if (!ctx) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

void pf_set_window(pf_ctx * ctx, int32_t max_forward_tokens) {
    if (ctx) ctx->max_forward_tokens = max_forward_tokens;
}

static int not_implemented(pf_ctx * ctx) {
    if (ctx) ctx->error = "not implemented yet";
    return -1;
}

int pf_classify(pf_ctx * ctx, const char *, size_t, float, pf_entity ** out, size_t * n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    return not_implemented(ctx);
}

void pf_entities_free(pf_entity * ents, size_t) { free(ents); }

int pf_tokenize(pf_ctx * ctx, const char *, size_t, int32_t ** ids, int32_t ** offsets, size_t * n) {
    if (ids) *ids = nullptr;
    if (offsets) *offsets = nullptr;
    if (n) *n = 0;
    return not_implemented(ctx);
}

int pf_logits(pf_ctx * ctx, const int32_t * ids, size_t n, float ** logits) {
    if (logits) *logits = nullptr;
    if (!ctx || !ids || n == 0) return -1;
    if (!ctx->error.empty()) return -1;

    const int n_cls = ctx->m.hp().n_cls;
    const int halo = ctx->m.hp().n_layer * ctx->m.hp().swa_radius;
    const int W = ctx->max_forward_tokens;
    if ((int) n > W && W <= 2 * halo) {
        ctx->error = "input exceeds the forward window and exact windowing needs window > " +
                     std::to_string(2 * halo);
        return -1;
    }
    std::vector<float> out((size_t) n * n_cls);
    std::vector<float> wlog;
    for (const pf::ner::window & w : pf::ner::make_windows((int) n, W, halo)) {
        if (!ctx->m.forward(ids + w.start, w.wlen, wlog)) {
            ctx->error = ctx->m.error;
            return -1;
        }
        std::memcpy(out.data() + (size_t) (w.start + w.lo) * n_cls,
                    wlog.data() + (size_t) w.lo * n_cls,
                    (size_t) (w.hi - w.lo) * n_cls * sizeof(float));
    }
    *logits = (float *) malloc(out.size() * sizeof(float));
    if (!*logits) return -1;
    std::memcpy(*logits, out.data(), out.size() * sizeof(float));
    return 0;
}

void pf_buf_free(void * buf) { free(buf); }
