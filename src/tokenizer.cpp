#include "tokenizer.h"

#include <queue>

namespace pf {

#include "unicode_data.inc"

enum cls_mask : uint8_t {
    CLS_LU = 1, CLS_LL = 2, CLS_LX = 4, CLS_M = 8, CLS_N = 16, CLS_WS = 32,
};
static constexpr uint8_t LETTER   = CLS_LU | CLS_LL | CLS_LX;
static constexpr uint8_t UPPERISH = CLS_LU | CLS_LX | CLS_M;  // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
static constexpr uint8_t LOWERISH = CLS_LL | CLS_LX | CLS_M;  // [\p{Ll}\p{Lm}\p{Lo}\p{M}]

static uint8_t cp_class(uint32_t cp) {
    if (cp > 0x10FFFF) return 0;  // invalid-byte sentinel: "other"
    int lo = 0, hi = (int) (sizeof(unicode_ranges) / sizeof(unicode_ranges[0])) - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (cp < unicode_ranges[mid].lo)      hi = mid - 1;
        else if (cp > unicode_ranges[mid].hi) lo = mid + 1;
        else return unicode_ranges[mid].mask;
    }
    return 0;
}

// Decoded view: codepoints, their classes, and each codepoint's byte offset.
// Invalid UTF-8: each bad byte is one sentinel codepoint of class "other"
// (deterministic; excluded from the HF differential, which is valid-UTF-8).
struct decoded {
    std::vector<uint32_t> cp;
    std::vector<uint8_t>  cls;
    std::vector<int32_t>  off;  // size n+1: off[i] = byte offset of cp i, off[n] = len
};

static decoded decode_utf8(const char * text, size_t len) {
    decoded d;
    d.cp.reserve(len);
    size_t i = 0;
    while (i < len) {
        d.off.push_back((int32_t) i);
        const uint8_t b0 = (uint8_t) text[i];
        uint32_t cp = 0;
        size_t n = 0;
        if      (b0 < 0x80)                { cp = b0; n = 1; }
        else if ((b0 & 0xE0) == 0xC0)      { cp = b0 & 0x1F; n = 2; }
        else if ((b0 & 0xF0) == 0xE0)      { cp = b0 & 0x0F; n = 3; }
        else if ((b0 & 0xF8) == 0xF0)      { cp = b0 & 0x07; n = 4; }
        if (n == 0 || i + n > len) {
            d.cp.push_back(0x110000u + b0);  // invalid lead byte
            i += 1;
        } else {
            bool ok = true;
            for (size_t k = 1; k < n; k++) {
                const uint8_t bk = (uint8_t) text[i + k];
                if ((bk & 0xC0) != 0x80) { ok = false; break; }
                cp = (cp << 6) | (bk & 0x3F);
            }
            // reject overlongs/surrogates/out-of-range as invalid bytes
            if (ok && n == 2 && cp < 0x80)     ok = false;
            if (ok && n == 3 && cp < 0x800)    ok = false;
            if (ok && n == 4 && cp < 0x10000)  ok = false;
            if (ok && (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) ok = false;
            if (ok) {
                d.cp.push_back(cp);
                i += n;
            } else {
                d.cp.push_back(0x110000u + b0);
                i += 1;
            }
        }
    }
    d.off.push_back((int32_t) len);
    d.cls.resize(d.cp.size());
    for (size_t k = 0; k < d.cp.size(); k++) d.cls[k] = cp_class(d.cp[k]);
    return d;
}

static inline bool is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// (?i:'s|'t|'re|'ve|'m|'ll|'d) — returns matched length in codepoints (0 if none)
static size_t match_contraction(const decoded & d, size_t i) {
    const size_t n = d.cp.size();
    if (i >= n || d.cp[i] != '\'') return 0;
    if (i + 1 >= n) return 0;
    const uint32_t c1 = d.cp[i + 1] | 0x20;  // ascii lowercase
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (i + 2 < n) {
        const uint32_t c2 = d.cp[i + 2] | 0x20;
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return 3;
    }
    return 0;
}

// Core of alternatives 1/2: UPPERISH*/+ then LOWERISH+/*, greedy with the
// single give-back the overlapping classes (Lm/Lo/M in both) require.
// upper_star=true:  [U]* [L]+  (alt 1) — needs >= 1 LOWERISH char overall
// upper_star=false: [U]+ [L]*  (alt 2) — needs >= 1 UPPERISH char first
static size_t match_word(const decoded & d, size_t i, bool upper_star) {
    const size_t n = d.cp.size();
    size_t u = i;
    while (u < n && (d.cls[u] & UPPERISH)) u++;
    if (!upper_star && u == i) return 0;
    size_t l = u;
    while (l < n && (d.cls[l] & LOWERISH)) l++;
    size_t end;
    if (upper_star && l == u) {
        // Greedy U* consumed everything and L+ has nothing. Backtracking
        // retries shorter U* prefixes from the longest down, so the match
        // ends right after the LAST char in [i, u) that is also LOWERISH
        // (the chars beyond it are U-only and fall out of the match).
        size_t p = u;
        while (p > i && !(d.cls[p - 1] & LOWERISH)) p--;
        if (p == i) return 0;  // no LOWERISH char anywhere: alt 1 fails
        end = p;
    } else {
        end = l;
    }
    end += match_contraction(d, end);
    return end - i;
}

// One pre-token starting at codepoint i; returns its length in codepoints
// (>= 1; some alternative always matches). Alternatives in pattern order.
static size_t pretok_next(const decoded & d, size_t i) {
    const size_t n = d.cp.size();
    const uint8_t c = d.cls[i];
    const bool prefix_ok = !is_crlf(d.cp[i]) && !(c & (LETTER | CLS_N));

    // alt 1: [^\r\n\p{L}\p{N}]? [U]* [L]+ (?i:suffix)?
    // alt 2: [^\r\n\p{L}\p{N}]? [U]+ [L]* (?i:suffix)?
    for (const bool upper_star : { true, false }) {
        if (prefix_ok && i + 1 < n) {
            const size_t r = match_word(d, i + 1, upper_star);
            if (r) return 1 + r;
        }
        const size_t r = match_word(d, i, upper_star);
        if (r) return r;
    }

    // alt 3: \p{N}{1,3}
    if (c & CLS_N) {
        size_t r = 1;
        while (r < 3 && i + r < n && (d.cls[i + r] & CLS_N)) r++;
        return r;
    }

    // alt 4: " ?" [^\s\p{L}\p{N}]+ [\r\n/]*
    {
        size_t j = i;
        if (d.cp[j] == ' ' && j + 1 < n) j++;
        size_t k = j;
        while (k < n && !(d.cls[k] & (CLS_WS | LETTER | CLS_N))) k++;
        if (k > j) {
            while (k < n && (is_crlf(d.cp[k]) || d.cp[k] == '/')) k++;
            return k - i;
        }
    }

    // whitespace run for alts 5-7
    size_t w = i;
    while (w < n && (d.cls[w] & CLS_WS)) w++;
    if (w > i) {
        // alt 5: \s*[\r\n]+ — match up to the end of the LAST contiguous
        // CR/LF block inside the run (greedy \s* backtracks to it)
        size_t last_end = 0;
        for (size_t k = w; k > i; k--) {
            if (is_crlf(d.cp[k - 1])) { last_end = k; break; }
        }
        if (last_end) return last_end - i;
        // alt 6: \s+(?!\S) — full run at end of input, else run minus one
        if (w == n) return w - i;
        if (w - i >= 2) return w - i - 1;
        // alt 7: \s+
        return w - i;
    }

    return 1;  // unreachable for any class, but guarantee progress
}

std::vector<std::pair<int32_t, int32_t>> pretokenize(const char * text, size_t len) {
    const decoded d = decode_utf8(text, len);
    std::vector<std::pair<int32_t, int32_t>> out;
    size_t i = 0;
    while (i < d.cp.size()) {
        const size_t r = pretok_next(d, i);
        out.emplace_back(d.off[i], d.off[i + r]);
        i += r;
    }
    return out;
}

bool tokenizer::load(const gguf_context * g) {
    // GPT-2 byte<->unicode: printable ranges map to themselves, the rest to
    // 0x100+n in order. Build cp -> byte for decoding the GGUF strings.
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;
    {
        auto is_direct = [](int b) {
            return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        };
        int shift = 0;
        for (int b = 0; b < 256; b++) {
            if (is_direct(b)) {
                cp_to_byte[(uint32_t) b] = (uint8_t) b;
            } else {
                cp_to_byte[0x100u + shift++] = (uint8_t) b;
            }
        }
    }
    auto decode_gpt2 = [&](const char * s) {
        std::string out;
        const size_t len = std::char_traits<char>::length(s);
        size_t i = 0;
        while (i < len) {
            const uint8_t b0 = (uint8_t) s[i];
            uint32_t cp;
            size_t n;
            if      (b0 < 0x80)           { cp = b0; n = 1; }
            else if ((b0 & 0xE0) == 0xC0) { cp = ((uint32_t)(b0 & 0x1F) << 6) | ((uint8_t) s[i+1] & 0x3F); n = 2; }
            else                          { cp = 0xFFFD; n = 1; }  // vocab strings are 1-2 byte cps
            auto it = cp_to_byte.find(cp);
            out.push_back(it != cp_to_byte.end() ? (char) it->second : (char) 0xFF);
            i += n;
        }
        return out;
    };

    const int64_t tok_id = gguf_find_key(g, "tokenizer.ggml.tokens");
    const int64_t mrg_id = gguf_find_key(g, "tokenizer.ggml.merges");
    if (tok_id < 0 || mrg_id < 0) {
        error = "GGUF lacks tokenizer.ggml.tokens / merges";
        return false;
    }
    const size_t n_tok = gguf_get_arr_n(g, tok_id);
    vocab.reserve(n_tok);
    for (size_t i = 0; i < n_tok; i++) {
        vocab.emplace(decode_gpt2(gguf_get_arr_str(g, tok_id, i)), (int32_t) i);
    }
    const size_t n_mrg = gguf_get_arr_n(g, mrg_id);
    merge_rank.reserve(n_mrg);
    for (size_t i = 0; i < n_mrg; i++) {
        const std::string m = gguf_get_arr_str(g, mrg_id, i);
        const size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        merge_rank.emplace(pair_key(decode_gpt2(m.substr(0, sp).c_str()),
                                    decode_gpt2(m.substr(sp + 1).c_str())),
                           (int32_t) i);
    }
    return true;
}

// BPE over one pre-token's raw bytes: linked-list symbols + lazy min-heap of
// merge candidates (rank, position, stamp). O(m log m) — long punctuation
// runs are single pre-tokens, so the naive quadratic loop is a fuzz DoS.
static void bpe_encode(const tokenizer & tk, const char * bytes, int32_t start, int32_t end,
                       std::vector<token_span> & out) {
    const int m = end - start;
    if (m <= 0) return;

    {   // ignore_merges=true: whole-pre-token vocab hit wins outright
        auto it = tk.vocab.find(std::string(bytes + start, bytes + end));
        if (it != tk.vocab.end()) {
            out.push_back({ it->second, start, end });
            return;
        }
    }

    struct sym { int32_t b, e; int prev, next; };
    std::vector<sym> syms(m);
    for (int i = 0; i < m; i++) {
        syms[i] = { start + i, start + i + 1, i - 1, i + 1 };
    }

    struct cand {
        int32_t rank;
        int     left;
        int32_t len_l, len_r;  // staleness stamp
        bool operator>(const cand & o) const {
            return rank != o.rank ? rank > o.rank : left > o.left;
        }
    };
    std::priority_queue<cand, std::vector<cand>, std::greater<cand>> heap;

    auto piece = [&](const sym & s) { return std::string(bytes + s.b, bytes + s.e); };
    auto push_pair = [&](int li) {
        const int ri = syms[li].next;
        if (ri >= m) return;
        auto it = tk.merge_rank.find(tokenizer::pair_key(piece(syms[li]), piece(syms[ri])));
        if (it != tk.merge_rank.end()) {
            heap.push({ it->second, li, syms[li].e - syms[li].b, syms[ri].e - syms[ri].b });
        }
    };
    for (int i = 0; i + 1 < m; i++) push_pair(i);

    while (!heap.empty()) {
        const cand c = heap.top();
        heap.pop();
        const int li = c.left;
        const int ri = syms[li].next;
        if (syms[li].b < 0 || ri >= m ||
            syms[li].e - syms[li].b != c.len_l || syms[ri].e - syms[ri].b != c.len_r) {
            continue;  // stale
        }
        // merge ri into li
        syms[li].e = syms[ri].e;
        syms[li].next = syms[ri].next;
        if (syms[ri].next < m) syms[syms[ri].next].prev = li;
        syms[ri].b = -1;  // dead
        if (syms[li].prev >= 0) push_pair(syms[li].prev);
        push_pair(li);
    }

    for (int i = 0; i < m && i >= 0;) {
        const sym & s = syms[i];
        auto it = tk.vocab.find(piece(s));
        if (it != tk.vocab.end()) {
            out.push_back({ it->second, s.b, s.e });
        } else {
            // unmergeable byte(s) outside the vocab: emit per-byte fallback
            for (int32_t b = s.b; b < s.e; b++) {
                auto bit = tk.vocab.find(std::string(bytes + b, bytes + b + 1));
                out.push_back({ bit != tk.vocab.end() ? bit->second : 0, b, b + 1 });
            }
        }
        i = s.next;
    }
}

std::vector<token_span> tokenizer::encode(const char * text, size_t len) const {
    std::vector<token_span> out;
    for (const auto & [s, e] : pretokenize(text, len)) {
        bpe_encode(*this, text, s, e, out);
    }
    // Widen token offsets to UTF-8 character boundaries: when BPE splits
    // inside a multibyte character, every piece reports the whole character
    // (HF tokenizers semantics, and what consumers validating offsets at
    // char boundaries need). Tokens may then overlap on that character.
    auto is_cont = [&](int32_t i) { return ((uint8_t) text[i] & 0xC0) == 0x80; };
    for (auto & t : out) {
        while (t.start > 0 && is_cont(t.start)) t.start--;
        while (t.end < (int32_t) len && is_cont(t.end)) t.end++;
    }
    return out;
}

} // namespace pf
