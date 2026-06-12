// Tokenizer tests. Scanner unit tests run without assets; the fixture
// replay (ids AND byte offsets equal to HF tokenizers across all cases)
// needs the GGUF vocab — it self-skips without PF_GGUF_DIR (label: model
// covers it in CI tier 2; locally both halves run in one invocation).
#include "tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK_MSG(cond, ...) do { \
    if (!(cond)) { failures++; std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                   std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } \
} while (0)

static std::vector<std::string> pretok(const std::string & s) {
    std::vector<std::string> out;
    for (const auto & [b, e] : pf::pretokenize(s.data(), s.size())) {
        out.push_back(s.substr(b, e - b));
    }
    return out;
}

static void expect_pretok(const std::string & text, const std::vector<std::string> & want) {
    const auto got = pretok(text);
    bool ok = got.size() == want.size();
    for (size_t i = 0; ok && i < got.size(); i++) ok = got[i] == want[i];
    if (!ok) {
        failures++;
        std::fprintf(stderr, "FAIL pretok '%s':\n  got : [", text.c_str());
        for (const auto & g : got) std::fprintf(stderr, "'%s' ", g.c_str());
        std::fprintf(stderr, "]\n  want: [");
        for (const auto & w : want) std::fprintf(stderr, "'%s' ", w.c_str());
        std::fprintf(stderr, "]\n");
    }
}

static void test_scanner() {
    // words, leading-space folding, case transitions (alt 1 vs alt 2)
    expect_pretok("Contact John", { "Contact", " John" });
    expect_pretok("USA", { "USA" });
    expect_pretok("McDonald", { "Mc", "Donald" });
    expect_pretok("iPhone", { "i", "Phone" });
    // contractions: the (?i:'s|'t|...) suffix is PART of alts 1/2 in o200k
    // (unlike GPT-2's separate alternative), so the word keeps its suffix
    expect_pretok("don't they'LL", { "don't", " they'LL" });
    // digits in groups of <= 3
    expect_pretok("12345", { "123", "45" });
    expect_pretok("a 1234", { "a", " ", "123", "4" });
    // punctuation with optional leading space and trailing slashes/newlines
    expect_pretok("x ?!", { "x", " ?!" });
    // alt 1's optional one-char prefix [^\r\n\p{L}\p{N}] wins over alt 4:
    // '.' and '/' attach to the following word (verified vs HF tokenizers)
    expect_pretok("a.b/c", { "a", ".b", "/c" });
    // whitespace: run before word leaves last space for the word (alt 6)
    expect_pretok("a   b", { "a", "  ", " b" });
    // trailing whitespace at end of input is one token (alt 6 at EOI)
    expect_pretok("a   ", { "a", "   " });
    // newlines: \s*[\r\n]+ groups trailing newline block (alt 5)
    expect_pretok("a \n\nb", { "a", " \n\n", "b" });
    expect_pretok("a\n \nb", { "a", "\n \n", "b" });
    // multibyte: ö is Ll, 北 is Lo (in both bracket classes)
    expect_pretok("wörld", { "wörld" });
    expect_pretok("北京", { "北京" });
}

static bool read_file(const std::string & path, std::vector<char> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz);
    size_t got = std::fread(out.data(), 1, sz, f);
    std::fclose(f);
    return got == (size_t) sz;
}

static void test_fixture_replay(const char * gguf_dir, const char * fixtures) {
    gguf_context * meta_ctx = nullptr;
    ggml_context * gctx = nullptr;
    gguf_init_params params = { /*no_alloc =*/ true, &gctx };
    const std::string gguf = std::string(gguf_dir) + "/pf-rope2-f16.gguf";
    meta_ctx = gguf_init_from_file(gguf.c_str(), params);
    if (!meta_ctx) {
        CHECK_MSG(false, "cannot open %s", gguf.c_str());
        return;
    }
    pf::tokenizer tk;
    if (!tk.load(meta_ctx)) {
        CHECK_MSG(false, "tokenizer load: %s", tk.error.c_str());
        return;
    }

    for (const char * cs : { "short-en", "multilingual", "pii-dense", "long-3k" }) {
        const std::string dir = std::string(fixtures) + "/" + cs;
        std::vector<char> text, ids_raw, offs_raw;
        if (!read_file(dir + "/text.txt", text) || !read_file(dir + "/tokens.i32", ids_raw) ||
            !read_file(dir + "/offsets.i32", offs_raw)) {
            std::fprintf(stderr, "note: fixture %s missing, skipping\n", cs);
            continue;
        }
        const size_t n_ref = ids_raw.size() / 4;
        const int32_t * ref_ids = (const int32_t *) ids_raw.data();
        const int32_t * ref_off = (const int32_t *) offs_raw.data();

        const auto got = tk.encode(text.data(), text.size());
        bool ok = got.size() == n_ref;
        size_t first_bad = got.size();
        for (size_t i = 0; ok && i < n_ref; i++) {
            if (got[i].id != ref_ids[i] || got[i].start != ref_off[2 * i] ||
                got[i].end != ref_off[2 * i + 1]) {
                ok = false;
                first_bad = i;
            }
        }
        std::printf("tokenizer %-12s %zu tokens %s\n", cs, got.size(), ok ? "OK" : "MISMATCH");
        if (!ok) {
            failures++;
            const size_t i = std::min(first_bad, got.size() - 1);
            std::fprintf(stderr, "  count got %zu want %zu; first diff at %zu: "
                         "got id=%d [%d,%d) want id=%d [%d,%d)\n",
                         got.size(), n_ref, i,
                         i < got.size() ? got[i].id : -1,
                         i < got.size() ? got[i].start : -1,
                         i < got.size() ? got[i].end : -1,
                         i < n_ref ? ref_ids[i] : -1,
                         i < n_ref ? ref_off[2 * i] : -1,
                         i < n_ref ? ref_off[2 * i + 1] : -1);
        }
    }
    gguf_free(meta_ctx);
    ggml_free(gctx);
}

// Committed torture corpus (tests/fixtures/tok_corpus.bin, written by
// hf_tok_diff.py --save-corpus): per text, the HF-reference ids and byte
// offsets. Replayed without Python.
static void test_corpus_replay(const std::string & gguf, const char * corpus_path) {
    std::vector<char> pack;
    if (!read_file(corpus_path, pack)) {
        std::fprintf(stderr, "note: %s missing, skipping corpus replay\n", corpus_path);
        return;
    }
    gguf_context * meta_ctx = nullptr;
    ggml_context * gctx = nullptr;
    gguf_init_params params = { true, &gctx };
    meta_ctx = gguf_init_from_file(gguf.c_str(), params);
    pf::tokenizer tk;
    if (!meta_ctx || !tk.load(meta_ctx)) {
        CHECK_MSG(false, "corpus replay: cannot load vocab");
        return;
    }
    const char * p = pack.data();
    auto rd_u32 = [&]() { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
    const uint32_t n_texts = rd_u32();
    int bad = 0;
    for (uint32_t i = 0; i < n_texts; i++) {
        const uint32_t len = rd_u32();
        const char * text = p;
        p += len;
        const uint32_t n_tok = rd_u32();
        const int32_t * ids = (const int32_t *) p;
        p += 4 * n_tok;
        const int32_t * offs = (const int32_t *) p;
        p += 8 * n_tok;

        const auto got = tk.encode(text, len);
        bool ok = got.size() == n_tok;
        for (size_t k = 0; ok && k < n_tok; k++) {
            ok = got[k].id == ids[k] && got[k].start == offs[2 * k] && got[k].end == offs[2 * k + 1];
        }
        if (!ok) {
            bad++;
            CHECK_MSG(false, "corpus text %u (%u bytes) mismatch", i, len);
        }
    }
    std::printf("corpus replay: %u texts, %d mismatches\n", n_texts, bad);
    gguf_free(meta_ctx);
    ggml_free(gctx);
}

int main() {
    test_scanner();

    const char * gguf_dir = std::getenv("PF_GGUF_DIR");
    const char * fixtures = std::getenv("PF_FIXTURES");
    if (gguf_dir && fixtures) {
        test_fixture_replay(gguf_dir, fixtures);
        test_corpus_replay(std::string(gguf_dir) + "/pf-rope2-f16.gguf",
                           (std::string(fixtures) + "/../tok_corpus.bin").c_str());
    } else {
        std::printf("note: fixture replay skipped (PF_GGUF_DIR/PF_FIXTURES unset)\n");
    }

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
