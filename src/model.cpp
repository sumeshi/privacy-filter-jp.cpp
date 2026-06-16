#include "model.h"

#include <ggml.h>

#include <cmath>
#include <cstring>

namespace pf {

void fill_swa_mask(float * dst, int64_t n, int64_t radius) {
    for (int64_t p1 = 0; p1 < n; p1++) {
        for (int64_t p0 = 0; p0 < n; p0++) {
            dst[p1 * n + p0] = swa_visible(p0, p1, radius) ? 0.0f : -INFINITY;
        }
    }
}

void yarn_freq_factors(const hparams & hp, std::vector<float> & ff, float & attn_factor) {
    const int    half   = hp.n_rot / 2;
    const double base   = hp.rope_freq_base;
    const double factor = hp.yarn_factor;

    ff.assign(half, 1.0f);
    attn_factor = 1.0f;
    if (factor <= 1.0f) {
        return;
    }

    // Ramp corner dims (transformers find_correction_range). The model trains
    // with truncate=false: corners stay fractional. floor/ceil only when the
    // GGUF carries rope.scaling.yarn_truncate=true.
    auto corr_dim = [&](double beta) {
        return hp.n_rot * std::log(hp.n_ctx_orig / (beta * 2.0 * M_PI)) / (2.0 * std::log(base));
    };
    double low  = corr_dim(hp.yarn_beta_fast);
    double high = corr_dim(hp.yarn_beta_slow);
    if (hp.yarn_truncate) {
        low  = std::floor(low);
        high = std::ceil(high);
    }
    low  = std::max(low, 0.0);
    high = std::min(high, (double) hp.n_rot - 1);

    for (int j = 0; j < half; j++) {
        const double extrap = std::pow(base, -2.0 * j / hp.n_rot);  // original inv_freq
        const double interp = extrap / factor;
        // ramp 0 at j<=low (keep original freq), 1 at j>=high (interpolate)
        double ramp = (j - low) / std::max(high - low, 1e-3);
        ramp = std::min(std::max(ramp, 0.0), 1.0);
        const double inv_freq = interp * ramp + extrap * (1.0 - ramp);
        // ggml_rope_ext computes theta = pos * base^(-2j/n) / ff[j]
        ff[j] = (float) (extrap / inv_freq);
    }
    attn_factor = (float) (0.1 * std::log(factor) + 1.0);
}

bool model::load(const std::string & gguf_path, const std::string & device, int n_threads) {
    release();

    if (!file.open(gguf_path, /*with_data =*/ true)) {
        error = file.error;
        return false;
    }
    if (!be.init(device, n_threads)) {
        error = be.error;
        return false;
    }
    if (!map_tensors() || !realize_weights()) {
        return false;
    }
    yarn_freq_factors(file.hp, freq_factors, attn_factor);
    return true;
}

void model::release() {
    if (weights_buf) { ggml_backend_buffer_free(weights_buf); weights_buf = nullptr; }
    if (device_ctx)  { ggml_free(device_ctx); device_ctx = nullptr; }
    be.release();
    layers.clear();
    file.close();
    error.clear();
}

bool model::map_tensors() {
    auto t = [&](const std::string & name) { return file.require(name.c_str()); };

    tok_embd    = t("token_embd.weight");
    output_norm = t("output_norm.weight");
    cls_w       = t("cls.output.weight");
    cls_b       = t("cls.output.bias");

    layers.resize(file.hp.n_layer);
    for (int i = 0; i < file.hp.n_layer; i++) {
        const std::string p = "blk." + std::to_string(i) + ".";
        layer_weights & l = layers[i];
        l.attn_norm   = t(p + "attn_norm.weight");
        l.wq          = t(p + "attn_q.weight");      l.bq = t(p + "attn_q.bias");
        l.wk          = t(p + "attn_k.weight");      l.bk = t(p + "attn_k.bias");
        l.wv          = t(p + "attn_v.weight");      l.bv = t(p + "attn_v.bias");
        l.wo          = t(p + "attn_output.weight"); l.bo = t(p + "attn_output.bias");
        l.sinks       = t(p + "attn_sinks.weight");
        l.post_norm   = t(p + "post_attention_norm.weight");
        l.router_w    = t(p + "ffn_gate_inp.weight");
        l.router_b    = t(p + "ffn_gate_inp.bias");
        l.gate_exps   = t(p + "ffn_gate_exps.weight");
        l.gate_exps_b = t(p + "ffn_gate_exps.bias");
        l.up_exps     = t(p + "ffn_up_exps.weight");
        l.up_exps_b   = t(p + "ffn_up_exps.bias");
        l.down_exps   = t(p + "ffn_down_exps.weight");
        l.down_exps_b = t(p + "ffn_down_exps.bias");
    }
    if (!file.error.empty()) {
        error = file.error;
        return false;
    }
    return true;
}

bool model::realize_weights() {
    if (be.is_cpu()) {
        // Zero-copy: the GGUF loaded with no_alloc=false, so all tensor data
        // lives in one contiguous ctx buffer. Wrap it as a CPU backend buffer
        // so graphs can reference loader tensors directly as leaves.
        void * base  = ggml_get_mem_buffer(file.ctx);
        size_t size  = ggml_get_mem_size(file.ctx);
        weights_buf = ggml_backend_cpu_buffer_from_ptr(base, size);
        if (!weights_buf) {
            error = "cpu weight buffer wrap failed";
            return false;
        }
        for (ggml_tensor * t = ggml_get_first_tensor(file.ctx); t; t = ggml_get_next_tensor(file.ctx, t)) {
            t->buffer = weights_buf;
        }
        return true;
    }

    // Device path: mirror weights into a no_alloc ctx, allocate on the
    // backend, upload, and repoint the layer struct at the device tensors.
    size_t n_tensors = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(file.ctx); t; t = ggml_get_next_tensor(file.ctx, t)) {
        n_tensors++;
    }
    ggml_init_params dp = { ggml_tensor_overhead() * (n_tensors + 8), nullptr, /*no_alloc =*/ true };
    device_ctx = ggml_init(dp);
    if (!device_ctx) {
        error = "device ctx init failed";
        return false;
    }
    for (ggml_tensor * s = ggml_get_first_tensor(file.ctx); s; s = ggml_get_next_tensor(file.ctx, s)) {
        ggml_tensor * d = ggml_new_tensor(device_ctx, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, ggml_get_name(s));
    }
    weights_buf = ggml_backend_alloc_ctx_tensors(device_ctx, be.be);
    if (!weights_buf) {
        error = "device weight alloc failed (out of device memory?)";
        return false;
    }
    for (ggml_tensor * s = ggml_get_first_tensor(file.ctx); s; s = ggml_get_next_tensor(file.ctx, s)) {
        ggml_tensor * d = ggml_get_tensor(device_ctx, ggml_get_name(s));
        ggml_backend_tensor_set(d, s->data, 0, ggml_nbytes(s));
    }
    // Repoint the weight map at the device tensors (file.ctx stays as source).
    auto remap = [&](ggml_tensor *& t) { if (t) t = ggml_get_tensor(device_ctx, ggml_get_name(t)); };
    remap(tok_embd); remap(output_norm); remap(cls_w); remap(cls_b);
    for (auto & l : layers) {
        remap(l.attn_norm);
        remap(l.wq); remap(l.bq); remap(l.wk); remap(l.bk); remap(l.wv); remap(l.bv);
        remap(l.wo); remap(l.bo); remap(l.sinks); remap(l.post_norm);
        remap(l.router_w); remap(l.router_b);
        remap(l.gate_exps); remap(l.gate_exps_b);
        remap(l.up_exps);   remap(l.up_exps_b);
        remap(l.down_exps); remap(l.down_exps_b);
    }
    return true;
}

bool model::forward(const int32_t * ids, int64_t n, std::vector<float> & logits, tap_map * taps) {
    const hparams & h = file.hp;
    const int64_t n_embd = h.n_embd, n_head = h.n_head, n_head_kv = h.n_head_kv, n_rot = h.n_rot;

    // ~45 nodes/layer * 8 layers + inputs/head; generous fixed bound.
    const size_t  graph_nodes = 1024;
    ggml_init_params gp = {
        ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
        nullptr,
        /*no_alloc =*/ true,
    };
    ggml_context * ctx = ggml_init(gp);
    if (!ctx) {
        error = "graph ctx init failed";
        return false;
    }

    std::vector<ggml_tensor *> tap_list;
    auto tap = [&](ggml_tensor * t, const std::string & name) {
        ggml_set_name(t, name.c_str());
        if (taps) {
            // Keep: gallocr must not reuse this buffer before readback. For
            // views (argsort_top_k's [k,n] slice) the bytes live in view_src —
            // flag it too or the parent gets recycled under the view.
            ggml_set_output(t);
            if (t->view_src) ggml_set_output(t->view_src);
            tap_list.push_back(t);
        }
        return t;
    };

    // inputs (data written after alloc)
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_tensor * inp_pos    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_tensor * kq_mask    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * ff         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_rot / 2);
    ggml_set_input(inp_tokens);
    ggml_set_input(inp_pos);
    ggml_set_input(kq_mask);
    ggml_set_input(ff);

    ggml_tensor * cur = ggml_get_rows(ctx, tok_embd, inp_tokens);  // [640, n]
    tap(cur, "embd");

    auto rms = [&](ggml_tensor * x, ggml_tensor * w) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, h.rms_eps), w);
    };
    auto rope = [&](ggml_tensor * x) {
        // YaRN folded into ff at load: ext_factor=0, freq_scale=1, exact
        // frequencies via freq_factors, cos/sin scale via attn_factor.
        // mode 0 = "normal" rope: interleaved (2i, 2i+1) pairs, matching the
        // HF reference's _apply_rotary_emb (NOT the NEOX rotate-half layout)
        return ggml_rope_ext(ctx, x, inp_pos, ff, (int) n_rot, /*mode =*/ 0,
                             h.n_ctx_orig, h.rope_freq_base, 1.0f,
                             0.0f, attn_factor, h.yarn_beta_fast, h.yarn_beta_slow);
    };

    // Ablation profiling hooks (PF_PROF): skip a block to attribute wall-time.
    //   noattn  -> skip self-attention; nomoe -> skip the MoE FFN.
    // The residual still runs on the (cheap rms) input, so the delta vs the full
    // forward is that block's cost. Build-time only; no effect unset.
    const char * prof = std::getenv("PF_PROF");
    const bool prof_noattn = prof && std::strstr(prof, "noattn");
    const bool prof_nomoe  = prof && std::strstr(prof, "nomoe");

    for (int il = 0; il < h.n_layer; il++) {
        const layer_weights & l = layers[il];
        const std::string L = "l" + std::to_string(il);
        ggml_tensor * resid = cur;

        cur = tap(rms(cur, l.attn_norm), L + ".attn_norm");

        // self-attention
        if (!prof_noattn) {
            ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, l.wq, cur), l.bq);  // [896, n]
            ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, l.wk, cur), l.bk);  // [128, n]
            ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, l.wv, cur), l.bv);  // [128, n]

            q = ggml_reshape_3d(ctx, q, n_rot, n_head, n);
            k = ggml_reshape_3d(ctx, k, n_rot, n_head_kv, n);
            v = ggml_reshape_3d(ctx, v, n_rot, n_head_kv, n);

            q = tap(rope(q), L + ".q_rope");  // [64, 14, n]
            k = tap(rope(k), L + ".k_rope");  // [64,  2, n]

            ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);                 // [64, n, 14]
            ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);                 // [64, n,  2]
            ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [n, 64,  2]

            ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);                        // [n, n, 14] (GQA broadcast)
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, kq_mask, 1.0f / std::sqrt((float) n_rot), 0.0f);
            ggml_soft_max_add_sinks(kq, l.sinks);

            ggml_tensor * kqv = ggml_mul_mat(ctx, vp, kq);                       // [64, n, 14]
            kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);                            // [64, 14, n]
            kqv = ggml_cont_2d(ctx, kqv, n_head * n_rot, n);                     // [896, n]

            cur = ggml_add(ctx, ggml_mul_mat(ctx, l.wo, kqv), l.bo);             // [640, n]
            tap(cur, L + ".attn_out");
        }

        cur = ggml_add(ctx, cur, resid);
        tap(cur, L + ".ffn_inp");
        resid = cur;

        cur = tap(rms(cur, l.post_norm), L + ".post_norm");

        // MoE FFN (softmax-after-top-k gating; the HF reference's /top_k and
        // *top_k cancel, so plain softmax weights are the trained semantics)
        if (!prof_nomoe) {
            const int64_t n_exp = h.n_expert, n_used = h.n_expert_used;

            ggml_tensor * rl = ggml_add(ctx, ggml_mul_mat(ctx, l.router_w, cur), l.router_b);  // [128, n]
            tap(rl, L + ".moe_logits");

            ggml_tensor * sel = tap(ggml_argsort_top_k(ctx, rl, (int) n_used), L + ".moe_topk");  // i32 [4, n]

            ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, rl, 1, n_exp, n), sel);     // [1, 4, n]
            w = ggml_soft_max(ctx, ggml_reshape_2d(ctx, w, n_used, n));
            tap(w, L + ".moe_weights");
            w = ggml_reshape_3d(ctx, w, 1, n_used, n);

            ggml_tensor * x3   = ggml_reshape_3d(ctx, cur, n_embd, 1, n);
            ggml_tensor * up   = ggml_mul_mat_id(ctx, l.up_exps, x3, sel);                        // [640, 4, n]
            up                 = ggml_add_id(ctx, up, l.up_exps_b, sel);
            ggml_tensor * gate = ggml_mul_mat_id(ctx, l.gate_exps, x3, sel);
            gate               = ggml_add_id(ctx, gate, l.gate_exps_b, sel);

            ggml_tensor * hms  = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f);                    // [640, 4, n]

            ggml_tensor * out  = ggml_mul_mat_id(ctx, l.down_exps, hms, sel);                     // [640, 4, n]
            out                = ggml_add_id(ctx, out, l.down_exps_b, sel);
            out                = ggml_mul(ctx, out, w);

            ggml_tensor * moe = nullptr;
            for (int64_t e = 0; e < n_used; e++) {
                ggml_tensor * slice = ggml_view_2d(ctx, out, n_embd, n, out->nb[2], e * out->nb[1]);
                moe = moe ? ggml_add(ctx, moe, slice) : slice;
            }
            cur = tap(ggml_cont(ctx, moe), L + ".moe_out");
        }

        cur = ggml_add(ctx, cur, resid);
        tap(cur, L + ".l_out");
    }

    cur = tap(rms(cur, output_norm), "result_norm");
    cur = ggml_add(ctx, ggml_mul_mat(ctx, cls_w, cur), cls_b);  // [217, n]
    tap(cur, "logits");
    ggml_set_output(cur);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_gallocr_alloc_graph(be.galloc, gf)) {
        error = "graph alloc failed";
        ggml_free(ctx);
        return false;
    }

    // inputs AFTER alloc. Guard each set on ->buffer: a PF_PROF ablation can
    // prune a block, orphaning its inputs (gallocr leaves them unallocated);
    // unset, every input is live so this is a no-op.
    ggml_backend_tensor_set(inp_tokens, ids, 0, n * sizeof(int32_t));
    {
        std::vector<int32_t> pos(n);
        for (int64_t i = 0; i < n; i++) pos[i] = (int32_t) i;
        if (inp_pos->buffer) ggml_backend_tensor_set(inp_pos, pos.data(), 0, n * sizeof(int32_t));

        std::vector<float> mask(n * n);
        fill_swa_mask(mask.data(), n, h.swa_radius);
        if (kq_mask->buffer) ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
        if (ff->buffer)      ggml_backend_tensor_set(ff, freq_factors.data(), 0, freq_factors.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(be.be, gf) != GGML_STATUS_SUCCESS) {
        error = "graph compute failed";
        ggml_free(ctx);
        return false;
    }

    logits.resize(n * h.n_cls);
    ggml_backend_tensor_get(cur, logits.data(), 0, logits.size() * sizeof(float));

    if (taps) {
        for (ggml_tensor * t : tap_list) {
            tap_data & d = (*taps)[ggml_get_name(t)];
            // ggml [D, n] (or [d0, d1, n]) -> row-major [n, D]
            d.n_rows = t->ne[ggml_n_dims(t) - 1];
            d.n_cols = ggml_nelements(t) / d.n_rows;
            const size_t esz = t->type == GGML_TYPE_I32 ? sizeof(int32_t) : sizeof(float);
            void * dst;
            if (t->type == GGML_TYPE_I32) {
                d.i32.resize(ggml_nelements(t));
                dst = d.i32.data();
            } else {
                d.f32.resize(ggml_nelements(t));
                dst = d.f32.data();
            }
            if (ggml_is_contiguous(t)) {
                ggml_backend_tensor_get(t, dst, 0, ggml_nelements(t) * esz);
            } else {
                // 2-D view (e.g. argsort_top_k's [k, n] slice of [n_expert, n]):
                // copy row by row using the parent stride
                for (int64_t r = 0; r < t->ne[1]; r++) {
                    ggml_backend_tensor_get(t, (char *) dst + r * t->ne[0] * esz,
                                            r * t->nb[1], t->ne[0] * esz);
                }
            }
        }
    }

    ggml_free(ctx);
    return true;
}

} // namespace pf
