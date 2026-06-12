#include "pf.h"

#include "model.h"
#include "ner.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

struct pf_ctx {
    pf::model            m;
    pf::tokenizer        tok;
    pf::ner::label_table lt;   // owns the category-name strings pf_entity points at
    int32_t              max_forward_tokens = 4096;
    std::string          error;
};

int pf_abi_version(void) { return PF_ABI_VERSION; }

pf_ctx * pf_load(const char * gguf_path, const char * device, int n_threads) {
    if (!gguf_path) return nullptr;
    auto * ctx = new pf_ctx();
    if (!ctx->m.load(gguf_path, device ? device : "cpu", n_threads)) {
        ctx->error = ctx->m.error;
    } else if (!ctx->tok.load(ctx->m.file.guf)) {
        ctx->error = ctx->tok.error;
    } else {
        ctx->lt = pf::ner::build_label_table(ctx->m.file.labels);
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

int pf_classify(pf_ctx * ctx, const char * text, size_t len, float threshold,
                pf_entity ** out, size_t * n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    if (!ctx || !text || !out || !n_out) return -1;
    if (!ctx->error.empty()) return -1;

    const std::vector<pf::token_span> toks = ctx->tok.encode(text, len);
    if (toks.empty()) return 0;

    std::vector<int32_t> ids(toks.size());
    for (size_t i = 0; i < toks.size(); i++) ids[i] = toks[i].id;

    std::vector<pf::ner::tok_span> spans;
    std::string err;
    if (!pf::ner::classify_tokens(ctx->m, ids.data(), (int) ids.size(),
                                  ctx->max_forward_tokens, threshold, spans, err)) {
        ctx->error = err;
        return -1;
    }

    std::vector<pf_entity> ents;
    ents.reserve(spans.size());
    for (const auto & sp : spans) {
        int32_t bstart = toks[sp.tok_begin].start;
        int32_t bend   = toks[sp.tok_end].end;
        if (bstart < 0 || bend > (int32_t) len || bstart >= bend) continue;
        // trim ASCII whitespace: o200k folds the leading space into the piece
        while (bstart < bend && (unsigned char) text[bstart] <= ' ') bstart++;
        while (bend > bstart && (unsigned char) text[bend - 1] <= ' ') bend--;
        if (bstart >= bend) continue;
        ents.push_back({ bstart, bend, sp.score, ctx->lt.category_name(sp.cat).c_str() });
    }

    *out = (pf_entity *) malloc(std::max<size_t>(ents.size(), 1) * sizeof(pf_entity));
    if (!*out) return -1;
    std::memcpy(*out, ents.data(), ents.size() * sizeof(pf_entity));
    *n_out = ents.size();
    return 0;
}

void pf_entities_free(pf_entity * ents, size_t) { free(ents); }

int pf_tokenize(pf_ctx * ctx, const char * text, size_t len,
                int32_t ** ids, int32_t ** offsets, size_t * n) {
    if (ids) *ids = nullptr;
    if (offsets) *offsets = nullptr;
    if (n) *n = 0;
    if (!ctx || !text || !ids || !offsets || !n) return -1;
    if (!ctx->error.empty()) return -1;

    const std::vector<pf::token_span> toks = ctx->tok.encode(text, len);
    *ids     = (int32_t *) malloc(std::max<size_t>(toks.size(), 1) * sizeof(int32_t));
    *offsets = (int32_t *) malloc(std::max<size_t>(toks.size(), 1) * 2 * sizeof(int32_t));
    if (!*ids || !*offsets) return -1;
    for (size_t i = 0; i < toks.size(); i++) {
        (*ids)[i] = toks[i].id;
        (*offsets)[2 * i]     = toks[i].start;
        (*offsets)[2 * i + 1] = toks[i].end;
    }
    *n = toks.size();
    return 0;
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
