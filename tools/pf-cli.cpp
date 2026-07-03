// pf-cli — inspect and run the privacy-filter model.
//   cat text.txt | pf-cli redact <model.gguf> [--threshold 0.5] [--device cpu] [--labels]
//   cat text.txt | pf-cli classify <model.gguf> [--threshold 0.5] [--device cpu]
//   pf-cli info <model.gguf>
#include "gguf_loader.h"
#include "pf.h"
#include "model.h"
#include "ner.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int info(const char * path) {
    pf::model_file mf;
    if (!mf.open(path, /*with_data =*/ false)) {
        std::fprintf(stderr, "error: %s\n", mf.error.c_str());
        return 1;
    }
    const pf::hparams & h = mf.hp;
    std::printf("arch            openai-privacy-filter\n");
    std::printf("kv pairs        %lld\n", (long long) gguf_get_n_kv(mf.guf));
    std::printf("tensors         %lld\n", (long long) gguf_get_n_tensors(mf.guf));
    std::printf("layers          %d\n", h.n_layer);
    std::printf("d_model         %d\n", h.n_embd);
    std::printf("heads           %d q / %d kv, head_dim %d\n", h.n_head, h.n_head_kv, h.n_rot);
    std::printf("experts         %d, top-%d, ff %d\n", h.n_expert, h.n_expert_used, h.n_ff_exp);
    std::printf("swa radius      %d (band diameter %d)\n", h.swa_radius, 2 * h.swa_radius);
    std::printf("rope            theta %.0f, yarn x%.0f (beta %.0f/%.0f, orig_ctx %d, truncate %s)\n",
                h.rope_freq_base, h.yarn_factor, h.yarn_beta_fast, h.yarn_beta_slow,
                h.n_ctx_orig, h.yarn_truncate ? "true" : "false");
    std::printf("rms_eps         %g\n", h.rms_eps);
    std::printf("labels          %d (%s ... %s)\n", h.n_cls,
                mf.labels.front().c_str(), mf.labels.back().c_str());
    return 0;
}

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

static bool dump_taps(const pf::tap_map & taps, int64_t n_tok, const std::string & dir) {
    std::string meta = "{\n \"n_tok\": " + std::to_string(n_tok) + ",\n \"taps\": {";
    bool first = true;
    for (const auto & [name, d] : taps) {
        const bool is_i32 = !d.i32.empty();
        const std::string path = dir + "/" + name + (is_i32 ? ".i32" : ".f32");
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            return false;
        }
        if (is_i32) std::fwrite(d.i32.data(), sizeof(int32_t), d.i32.size(), f);
        else        std::fwrite(d.f32.data(), sizeof(float),   d.f32.size(), f);
        std::fclose(f);
        meta += std::string(first ? "" : ",") + "\n  \"" + name + "\": {\"shape\": [" +
                std::to_string(d.n_rows) + ", " + std::to_string(d.n_cols) +
                "], \"dtype\": \"" + (is_i32 ? "i32" : "f32") + "\"}";
        first = false;
    }
    meta += "\n }\n}\n";
    FILE * f = std::fopen((dir + "/meta.json").c_str(), "wb");
    if (!f) return false;
    std::fwrite(meta.data(), 1, meta.size(), f);
    std::fclose(f);
    return true;
}

// Differential-test workhorse: read a pack of texts ([u32 n] then per text
// [u32 len][bytes]), tokenize each (vocab only, no weights), write a pack of
// results ([u32 n_tok][ids i32 x n][offsets i32 x 2n] per text).
static int tok_batch(const char * gguf, const char * in_path, const char * out_path) {
    pf::model_file mf;
    if (!mf.open(gguf, /*with_data =*/ false)) {
        std::fprintf(stderr, "%s\n", mf.error.c_str());
        return 1;
    }
    pf::tokenizer tk;
    if (!tk.load(mf.guf)) {
        std::fprintf(stderr, "%s\n", tk.error.c_str());
        return 1;
    }
    FILE * in = std::fopen(in_path, "rb");
    FILE * out = std::fopen(out_path, "wb");
    if (!in || !out) {
        std::fprintf(stderr, "cannot open %s / %s\n", in_path, out_path);
        return 1;
    }
    uint32_t n_texts = 0;
    if (std::fread(&n_texts, 4, 1, in) != 1) return 1;
    std::vector<char> buf;
    for (uint32_t i = 0; i < n_texts; i++) {
        uint32_t len = 0;
        if (std::fread(&len, 4, 1, in) != 1) return 1;
        buf.resize(len);
        if (len && std::fread(buf.data(), 1, len, in) != len) return 1;
        const auto toks = tk.encode(buf.data(), len);
        const uint32_t n = (uint32_t) toks.size();
        std::fwrite(&n, 4, 1, out);
        for (const auto & t : toks) std::fwrite(&t.id, 4, 1, out);
        for (const auto & t : toks) {
            std::fwrite(&t.start, 4, 1, out);
            std::fwrite(&t.end, 4, 1, out);
        }
    }
    std::fclose(in);
    std::fclose(out);
    return 0;
}

static bool label_is(const char * label, const char * want) {
    return label && std::strcmp(label, want) == 0;
}

static const char * redaction_for_label(const char * label) {
    if (label_is(label, "private_person") || label_is(label, "FIRSTNAME") ||
        label_is(label, "LASTNAME") || label_is(label, "MIDDLENAME") ||
        label_is(label, "JOBTITLE")) {
        return "[PERSON]";
    }
    if (label_is(label, "private_address") || label_is(label, "STREET") ||
        label_is(label, "BUILDINGNUMBER") || label_is(label, "CITY") ||
        label_is(label, "ZIPCODE") || label_is(label, "STATE") ||
        label_is(label, "COUNTY") || label_is(label, "SECONDARYADDRESS")) {
        return "[ADDRESS]";
    }
    if (label_is(label, "private_email") || label_is(label, "EMAIL")) {
        return "[EMAIL]";
    }
    if (label_is(label, "private_phone") || label_is(label, "PHONE") ||
        label_is(label, "PHONENUMBER")) {
        return "[PHONE]";
    }
    if (label_is(label, "private_date") || label_is(label, "DATE") ||
        label_is(label, "DATEOFBIRTH") || label_is(label, "AGE") ||
        label_is(label, "TIME")) {
        return "[DATE]";
    }
    if (label_is(label, "private_url") || label_is(label, "URL")) {
        return "[URL]";
    }
    if (label_is(label, "secret") || label_is(label, "PASSWORD")) {
        return "[SECRET]";
    }
    if (label_is(label, "account_number") || label_is(label, "ACCOUNTNUMBER") ||
        label_is(label, "BANKACCOUNT") || label_is(label, "ACCOUNTNAME") ||
        label_is(label, "USERNAME") || label_is(label, "IBAN") ||
        label_is(label, "SSN") || label_is(label, "PIN") ||
        label_is(label, "CVV") || label_is(label, "CREDITCARDNUMBER") ||
        label_is(label, "BIC") || label_is(label, "VRM") ||
        label_is(label, "IPADDRESS") || label_is(label, "MAC") ||
        label_is(label, "AMOUNT") || label_is(label, "HEIGHT")) {
        return "[ACCOUNT]";
    }
    return "[REDACTED]";
}

static void print_redacted(const std::string & text, pf_entity * ents, size_t n_ents,
                           bool labels, const std::string & replacement) {
    if (n_ents == 0 || !ents) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fputc('\n', stdout);
        return;
    }

    std::vector<pf_entity> sorted(ents, ents + n_ents);
    std::sort(sorted.begin(), sorted.end(), [](const pf_entity & a, const pf_entity & b) {
        if (a.start != b.start) return a.start < b.start;
        return a.end > b.end;
    });

    size_t cursor = 0;
    for (const pf_entity & e : sorted) {
        if (e.start < 0 || e.end <= e.start || (size_t) e.end > text.size()) continue;
        const size_t start = (size_t) e.start;
        const size_t end = (size_t) e.end;
        if (end <= cursor) continue;
        if (start > cursor) {
            std::fwrite(text.data() + cursor, 1, start - cursor, stdout);
        }
        std::fputs(labels ? redaction_for_label(e.label) : replacement.c_str(), stdout);
        cursor = end;
    }
    if (cursor < text.size()) {
        std::fwrite(text.data() + cursor, 1, text.size() - cursor, stdout);
    }
    std::fputc('\n', stdout);
}

static int usage(int rc = 2) {
    FILE * out = rc == 0 ? stdout : stderr;
    std::fprintf(out,
        "usage:\n"
        "  cat input.txt | pf-cli redact <model.gguf> [--threshold 0.5] [--device cpu] [--labels]\n"
        "  cat input.txt | pf-cli classify <model.gguf> [--threshold 0.5] [--device cpu]\n"
        "  pf-cli info <model.gguf>\n"
        "\n"
        "redact output:\n"
        "  default       replace each detected span with ***\n"
        "  --labels      replace spans with labels like [PERSON] and [ADDRESS]\n"
        "\n"
        "devices:\n"
        "  cpu, gpu, cuda, vulkan, cuda:N, vulkan:N\n"
        "\n"
        "examples:\n"
        "  echo 'お問い合わせは山田太郎（taro@example.com）まで' | pf-cli redact privacy-filter-jp-f16.gguf\n"
        "  echo 'お問い合わせは山田太郎（taro@example.com）まで' | pf-cli redact privacy-filter-jp-f16.gguf --labels\n"
        "  echo '...' | pf-cli classify privacy-filter-jp-f16.gguf --threshold 0.5\n"
        "\n"
        "note:\n"
        "  stdin is read to EOF before inference; output is emitted after spans are known.\n"
        "\n"
        "developer:\n"
        "  pf-cli --model <model.gguf> --tokens <tokens.i32> [--device cpu]\n");
    return rc;
}

struct text_cli_args {
    std::string model;
    std::string device = "cpu";
    std::string replacement = "***";
    float threshold = 0.5f;
    bool labels = false;
};

static bool parse_float(const char * s, float & out) {
    if (!s || !*s) return false;
    char * end = nullptr;
    const float v = std::strtof(s, &end);
    if (!end || *end != '\0') return false;
    out = v;
    return true;
}

static bool parse_text_args(int argc, char ** argv, int model_i, text_cli_args & out) {
    if (model_i >= argc) return false;
    out.model = argv[model_i];
    for (int i = model_i + 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--threshold" || a == "-t") {
            if (++i >= argc || !parse_float(argv[i], out.threshold)) return false;
        } else if (a == "--device" || a == "-d") {
            if (++i >= argc) return false;
            out.device = argv[i];
        } else if (a == "--labels") {
            out.labels = true;
        } else if (a == "--replacement") {
            if (++i >= argc) return false;
            out.replacement = argv[i];
        } else if (a == "--help" || a == "-h") {
            return false;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        } else {
            float threshold = 0.0f;
            if (parse_float(a.c_str(), threshold)) out.threshold = threshold;
            else out.device = a;
        }
    }
    return true;
}

static bool read_stdin(std::string & text) {
    char buf[65536];
    size_t r;
    while ((r = std::fread(buf, 1, sizeof(buf), stdin)) > 0) text.append(buf, r);
    if (std::ferror(stdin)) return false;
    if (!text.empty() && text.back() == '\n') text.pop_back();
    return true;
}

static int run_text_mode(bool classify_mode, int argc, char ** argv, int model_i) {
    text_cli_args a;
    if (!parse_text_args(argc, argv, model_i, a)) return usage();

    std::string text;
    if (!read_stdin(text)) {
        std::fprintf(stderr, "failed to read stdin\n");
        return 1;
    }

    pf_ctx * ctx = pf_load(a.model.c_str(), a.device.c_str(), 0);
    if (pf_last_error(ctx)) {
        std::fprintf(stderr, "load: %s\n", pf_last_error(ctx));
        pf_free(ctx);
        return 1;
    }

    pf_entity * ents = nullptr;
    size_t n_ents = 0;
    if (pf_classify(ctx, text.data(), text.size(), a.threshold, &ents, &n_ents) != 0) {
        std::fprintf(stderr, "classify: %s\n", pf_last_error(ctx));
        pf_free(ctx);
        return 1;
    }

    if (!classify_mode) {
        print_redacted(text, ents, n_ents, a.labels, a.replacement);
        pf_entities_free(ents, n_ents);
        pf_free(ctx);
        return 0;
    }

    std::printf("[");
    for (size_t i = 0; i < n_ents; i++) {
        std::printf("%s\n  {\"entity_group\": \"%s\", \"start\": %d, \"end\": %d, "
                    "\"score\": %.4f, \"text\": \"%.*s\"}",
                    i ? "," : "", ents[i].label, ents[i].start, ents[i].end,
                    ents[i].score, ents[i].end - ents[i].start, text.data() + ents[i].start);
    }
    std::printf("\n]\n");

    pf_entities_free(ents, n_ents);
    pf_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) {
        return usage(argc < 2 ? 2 : 0);
    }
    if (argc == 3 && (std::strcmp(argv[1], "info") == 0 || std::strcmp(argv[1], "--info") == 0)) {
        return info(argv[2]);
    }
    if (std::strcmp(argv[1], "redact") == 0 || std::strcmp(argv[1], "--redact") == 0) {
        return run_text_mode(false, argc, argv, 2);
    }
    if (std::strcmp(argv[1], "classify") == 0 || std::strcmp(argv[1], "--classify") == 0) {
        return run_text_mode(true, argc, argv, 2);
    }
    if (argc == 3 && std::strcmp(argv[1], "--info") == 0) {
        return info(argv[2]);
    }
    if (argc == 5 && std::strcmp(argv[1], "--tok-batch") == 0) {
        return tok_batch(argv[2], argv[3], argv[4]);
    }
    // Legacy flag-style entry points, kept for older scripts.
    //   pf-cli --classify <model> <threshold> [device] < text.txt
    //   pf-cli --redact   <model> <threshold> [device] < text.txt
    const char * cmd = argc > 1 ? argv[1] : "";
    const bool classify_mode = std::strcmp(cmd, "--classify") == 0;
    const bool redact_mode = std::strcmp(cmd, "--redact") == 0;
    if ((argc == 4 || argc == 5) && (classify_mode || redact_mode)) {
        std::string text;
        char buf[65536];
        size_t r;
        while ((r = std::fread(buf, 1, sizeof(buf), stdin)) > 0) text.append(buf, r);
        if (!text.empty() && text.back() == '\n') text.pop_back();

        pf_ctx * ctx = pf_load(argv[2], argc == 5 ? argv[4] : nullptr, 0);
        if (pf_last_error(ctx)) {
            std::fprintf(stderr, "load: %s\n", pf_last_error(ctx));
            return 1;
        }
        pf_entity * ents = nullptr;
        size_t n_ents = 0;
        const float thr = argc >= 4 ? (float) std::atof(argv[3]) : 0.0f;
        if (pf_classify(ctx, text.data(), text.size(), thr, &ents, &n_ents) != 0) {
            std::fprintf(stderr, "classify: %s\n", pf_last_error(ctx));
            return 1;
        }
        if (redact_mode) {
            print_redacted(text, ents, n_ents, false, "***");
            pf_entities_free(ents, n_ents);
            pf_free(ctx);
            return 0;
        }
        std::printf("[");
        for (size_t i = 0; i < n_ents; i++) {
            std::printf("%s\n {\"entity_group\": \"%s\", \"start\": %d, \"end\": %d, "
                        "\"score\": %.4f, \"text\": \"%.*s\"}",
                        i ? "," : "", ents[i].label, ents[i].start, ents[i].end,
                        ents[i].score, ents[i].end - ents[i].start, text.data() + ents[i].start);
        }
        std::printf("\n]\n");
        pf_entities_free(ents, n_ents);
        pf_free(ctx);
        return 0;
    }

    std::string model_path, tokens_path, taps_dir, logits_path, offsets_path, text_path;
    std::string device = "cpu";
    int threads = 0, window = 4096;
    float threshold = 0.0f;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if      (a == "--model")      model_path   = argv[i + 1];
        else if (a == "--tokens")     tokens_path  = argv[i + 1];
        else if (a == "--dump-taps")  taps_dir     = argv[i + 1];
        else if (a == "--logits-out") logits_path  = argv[i + 1];
        else if (a == "--offsets")    offsets_path = argv[i + 1];
        else if (a == "--text")       text_path    = argv[i + 1];
        else if (a == "--device")     device       = argv[i + 1];
        else if (a == "--threads")    threads      = std::atoi(argv[i + 1]);
        else if (a == "--window")     window       = std::atoi(argv[i + 1]);
        else if (a == "--threshold")  threshold    = (float) std::atof(argv[i + 1]);
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || tokens_path.empty()) {
        std::fprintf(stderr,
            "usage: pf-cli --info <model.gguf>\n"
            "       pf-cli --classify <model.gguf> <threshold> [cpu|gpu|cuda|vulkan|cuda:N] < text.txt\n"
            "       pf-cli --redact   <model.gguf> <threshold> [cpu|gpu|cuda|vulkan|cuda:N] < text.txt\n"
            "       pf-cli --model <model.gguf> --tokens <tokens.i32> [--device cpu|gpu|cuda|vulkan|cuda:N]\n"
            "              [--threads N] [--window N] [--dump-taps DIR] [--logits-out FILE]\n"
            "              [--offsets <offsets.i32> --text <text.txt> [--threshold F]]\n");
        return 2;
    }

    std::vector<int32_t> ids;
    if (!read_i32(tokens_path, ids)) {
        std::fprintf(stderr, "cannot read %s\n", tokens_path.c_str());
        return 1;
    }

    pf::model m;
    if (!m.load(model_path, device, threads)) {
        std::fprintf(stderr, "load error: %s\n", m.error.c_str());
        return 1;
    }
    std::fprintf(stderr, "loaded %s on %s, %zu tokens\n",
                 model_path.c_str(), m.be.device.c_str(), ids.size());

    pf::tap_map taps;
    std::vector<float> logits;
    if (!m.forward(ids.data(), (int64_t) ids.size(), logits, taps_dir.empty() ? nullptr : &taps)) {
        std::fprintf(stderr, "forward error: %s\n", m.error.c_str());
        return 1;
    }

    if (!taps_dir.empty() && !dump_taps(taps, (int64_t) ids.size(), taps_dir)) {
        return 1;
    }
    if (!logits_path.empty()) {
        FILE * f = std::fopen(logits_path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", logits_path.c_str()); return 1; }
        std::fwrite(logits.data(), sizeof(float), logits.size(), f);
        std::fclose(f);
    }

    // Span mode: byte offsets from file (the tokenizer lands in P4), text for
    // span extraction. Mirrors the LocalAI TokenClassify response assembly.
    if (!offsets_path.empty() && !text_path.empty()) {
        std::vector<int32_t> offs;
        std::vector<char> text;
        FILE * tf = std::fopen(text_path.c_str(), "rb");
        if (!read_i32(offsets_path, offs) || !tf) {
            std::fprintf(stderr, "cannot read offsets/text\n");
            return 1;
        }
        std::fseek(tf, 0, SEEK_END);
        text.resize(std::ftell(tf));
        std::fseek(tf, 0, SEEK_SET);
        if (std::fread(text.data(), 1, text.size(), tf) != text.size()) { std::fclose(tf); return 1; }
        std::fclose(tf);

        std::vector<pf::ner::tok_span> spans;
        std::string err;
        if (!pf::ner::classify_tokens(m, ids.data(), (int) ids.size(), window, threshold, spans, err)) {
            std::fprintf(stderr, "classify error: %s\n", err.c_str());
            return 1;
        }
        const pf::ner::label_table lt = pf::ner::build_label_table(m.file.labels);
        std::printf("[");
        bool first = true;
        for (const auto & sp : spans) {
            int bstart = offs[2 * sp.tok_begin];
            int bend   = offs[2 * sp.tok_end + 1];
            if (bstart < 0 || bend > (int) text.size() || bstart >= bend) continue;
            // trim leading/trailing ASCII whitespace (o200k folds the leading
            // space into the piece; mask the trimmed form)
            while (bstart < bend && (unsigned char) text[bstart] <= ' ') bstart++;
            while (bend > bstart && (unsigned char) text[bend - 1] <= ' ') bend--;
            if (bstart >= bend) continue;
            std::printf("%s\n {\"entity_group\": \"%s\", \"start\": %d, \"end\": %d, "
                        "\"score\": %.4f, \"text\": \"%.*s\"}",
                        first ? "" : ",", lt.category_name(sp.cat).c_str(), bstart, bend,
                        sp.score, bend - bstart, text.data() + bstart);
            first = false;
        }
        std::printf("\n]\n");
        return 0;
    }

    // print argmax label per token as a quick eyeball
    const int n_cls = m.hp().n_cls;
    for (size_t t = 0; t < ids.size() && t < 32; t++) {
        int best = 0;
        for (int c = 1; c < n_cls; c++) {
            if (logits[t * n_cls + c] > logits[t * n_cls + best]) best = c;
        }
        std::printf("%3zu %-6d %s\n", t, ids[t], m.file.labels[best].c_str());
    }
    return 0;
}
