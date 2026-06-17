#!/usr/bin/env python3
"""
pii_scan.py — a single-document PII scan, rendered with rich and captured by
recorder-for-agents into an MP4.

The left pane streams one document at the engine's REAL measured throughput
(tok/s): the view scrolls a page at a time as tokens are processed and each PII
span snaps to ████ as the frontier passes it. The right pane is the live NER
feed — every entity the model finds, in document order, with its category and
byte range. A bottom strip tracks tokens processed, elapsed time and tok/s.

Data (built by gen_scan.py from real pf-cli output):
  <scene>/content.json -> {"document","n_tokens","note","entities":[{type,start,end,text,score}]}
  <scene>/engines.json -> [{"label","device","tps"}]

  python3 pii_scan.py --scene traces/scan --dilate 1
"""
import argparse, bisect, json, time
from pathlib import Path

from rich import box
from rich.console import Group, Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

# ---- palette ---------------------------------------------------------------
INK   = "#d7dde5"
DIM   = "#6e7681"
FAINT = "#3b424c"
RULE  = "#222b34"
ACC, ACC_EMPTY = "#3ec8e0", "#1d6c7b"     # teal frontier / bar
GREEN = "#46c266"
GOLD  = "#e3b341"

# entity_group -> redaction/label colour, grouped by kind (names, contact,
# location, financial/id, secrets/misc). Unknown categories fall back to slate.
CAT = {
    "FIRSTNAME": "#e0709a", "LASTNAME": "#e0709a", "MIDDLENAME": "#e0709a",
    "ACCOUNTNAME": "#e0709a", "USERNAME": "#d98ca0", "JOBTITLE": "#d98ca0",
    "EMAIL": "#7aa2f7", "URL": "#7aa2f7", "IPADDRESS": "#6ab0f3", "MAC": "#6ab0f3",
    "PHONE": "#e0a458", "PHONENUMBER": "#e0a458",
    "DATE": "#9d7cd8", "DATEOFBIRTH": "#b48ee8", "AGE": "#b48ee8", "TIME": "#9d7cd8",
    "STREET": "#98c379", "BUILDINGNUMBER": "#98c379", "CITY": "#8ac06f",
    "ZIPCODE": "#7fb86a", "STATE": "#a6cc8a", "COUNTY": "#a6cc8a", "SECONDARYADDRESS": "#98c379",
    "IBAN": "#56b6c2", "AMOUNT": "#56b6c2", "PIN": "#5fc9d6", "SSN": "#4fb0bd",
    "CVV": "#4fb0bd", "CREDITCARDNUMBER": "#56b6c2", "ACCOUNTNUMBER": "#56b6c2",
    "PASSWORD": "#e3b341", "VRM": "#e3b341", "HEIGHT": "#cbb05a", "BIC": "#56b6c2",
}
DEFAULT_CAT = "#9aa4af"


def cat_colour(t):
    return CAT.get(t, DEFAULT_CAT)


# ---- layout helpers (shared shape with pii_duel.py) ------------------------
def wrap_ranges(s, width):
    """Greedy word-wrap over the source string -> (start, end) char ranges."""
    n = len(s); width = max(8, width)
    out, i = [], 0
    while i < n:
        nl = s.find("\n", i, i + width)
        if nl != -1:
            out.append((i, nl)); i = nl + 1; continue
        if i + width >= n:
            out.append((i, n)); break
        sp = s.rfind(" ", i, i + width)
        if sp > i:
            out.append((i, sp)); i = sp + 1
        else:
            out.append((i, i + width)); i += width
    return out or [(0, 0)]


def prep(content, inner_w):
    doc = content["document"]
    ents = sorted(content["entities"], key=lambda e: e["start"])
    ranges = wrap_ranges(doc, inner_w)
    return {"doc": doc, "ents": ents,
            "starts": [e["start"] for e in ents],
            "ranges": ranges, "line_starts": [s for s, _ in ranges],
            "n_tokens": content["n_tokens"], "inner_w": inner_w}


def _emit_norm(t, doc, a, b, playhead, cursor):
    if a >= b:
        return
    if playhead <= a:
        t.append(doc[a:b], style=FAINT)
    elif playhead >= b:
        t.append(doc[a:b], style=INK)
    else:
        t.append(doc[a:playhead], style=INK)
        t.append(doc[playhead], style=cursor)
        t.append(doc[playhead + 1:b], style=FAINT)


def render_page(P, page_lines, playhead):
    doc, ents, starts, n = P["doc"], P["ents"], P["starts"], len(P["ents"])
    cursor = f"{ACC} reverse"
    lines = []
    for s, e in page_lines:
        t = Text(no_wrap=True, overflow="crop")
        k = bisect.bisect_left(starts, s)
        if k > 0 and ents[k - 1]["end"] > s:
            k -= 1
        cur = s
        while k < n and ents[k]["start"] < e:
            en = ents[k]; k += 1
            a, b = max(s, en["start"]), min(e, en["end"])
            if b <= cur:
                continue
            _emit_norm(t, doc, cur, a, playhead, cursor)
            if playhead >= en["start"]:
                t.append("█" * (b - a), style=cat_colour(en["type"]))
            else:
                t.append(doc[a:b], style=FAINT)
            cur = b
        _emit_norm(t, doc, cur, e, playhead, cursor)
        lines.append(t)
    return lines


def bar(frac, width, fill, empty, label):
    f = max(0, min(width, round(width * frac)))
    t = Text()
    t.append("━" * f, style=fill)
    t.append("━" * (width - f), style=empty)
    t.append(f"  {label}", style=fill)
    return t


def doc_pane(P, playhead, height):
    body_lines = max(4, height - 4)
    cur_line = max(0, bisect.bisect_right(P["line_starts"], playhead) - 1)
    page0 = (cur_line // body_lines) * body_lines
    lines = render_page(P, P["ranges"][page0:page0 + body_lines], playhead)
    body = Text("\n").join(lines)
    pad = body_lines - len(lines)
    if pad > 0:
        body.append("\n" * pad)
    page_n = page0 // body_lines + 1
    pages = (len(P["ranges"]) + body_lines - 1) // body_lines
    title = Text(" document ", style=f"bold {ACC}")
    title.append(f" scanning · page {page_n}/{pages} ", style=DIM)
    return Panel(body, title=title, title_align="left", border_style=ACC_EMPTY,
                 box=box.ROUNDED, padding=(1, 2), height=height)


def findings_pane(P, playhead, height, done):
    """Right column: the live NER feed (newest at the bottom), tail-limited."""
    found = bisect.bisect_right(P["starts"], playhead)   # entities discovered so far
    rows = max(3, height - 4)
    ents = P["ents"][:found]
    shown = ents[-rows:]
    cat_w = 14
    t = Text()
    if found > rows:
        t.append(f"  ⋮ +{found - rows} earlier\n", style=FAINT)
    for e in shown:
        col = cat_colour(e["type"])
        lab = (e["type"][:cat_w]).ljust(cat_w)
        t.append(lab + " ", style=f"bold {col}")
        txt = e.get("text", "")
        avail = max(6, P["find_w"] - cat_w - 1 - 11)
        if len(txt) > avail:
            txt = txt[:avail - 1] + "…"
        t.append(txt.ljust(avail), style=INK)
        t.append(f" {e['start']:>4}-{e['end']:<4}\n", style=FAINT)
    pad = rows - len(shown) - (1 if found > rows else 0)
    if pad > 0:
        t.append("\n" * pad)
    title = Text(" PII found ", style=f"bold {GOLD}")
    title.append(f" {found} spans ", style=DIM)
    border = GREEN if done else (GOLD if found else FAINT)
    return Panel(t, title=title, title_align="left", border_style=border,
                 box=box.ROUNDED, padding=(1, 1), height=height)


def status_strip(P, eng, playhead, tokens_done, elapsed_ms, frac, done, cols):
    found = bisect.bisect_right(P["starts"], playhead)
    line = Text()
    line.append(f"{tokens_done:,}", style=INK)
    line.append(f" / {P['n_tokens']:,} tok", style=DIM)
    line.append("    ")
    if done:
        line.append("✓ ", style=f"bold {GREEN}")
        line.append(f"{eng['proc_s'] * 1000:,.0f} ms", style=f"bold {INK}")
    else:
        line.append("▸ ", style=ACC)
        line.append(f"{elapsed_ms:,.0f} ms", style=f"bold {INK}")
    line.append(f"   {eng['tps']:,.0f} tok/s", style=DIM)
    line.append(f"   ·   {found} entities", style=DIM)
    blabel = "100%" if done else f"{int(round(frac * 100)):>3d}%"
    return Group(bar(frac, max(10, cols - 10), ACC, ACC_EMPTY, blabel), line)


def header(eng, note, dilate, cols):
    left = Text()
    left.append(eng["label"], style=f"bold {ACC}")
    left.append("   " + eng["device"], style=DIM)
    right = Text(justify="right")
    if note:
        right.append(note, style=INK)
    if dilate > 1.01:
        right.append(f"   ·   slowed {dilate:g}×", style=FAINT)
    g = Table.grid(expand=True)
    g.add_column(justify="left"); g.add_column(justify="right")
    g.add_row(left, right)
    return Group(g, Text("─" * cols, style=RULE))


def view(P, eng, note, elapsed, dilate, cols, rows):
    proc = eng["proc_s"]
    frac = 0.0 if proc <= 0 else max(0.0, min(1.0, elapsed / proc))
    done = elapsed >= proc
    playhead = round(frac * len(P["doc"]))
    tokens_done = round(frac * P["n_tokens"])
    body_h = max(10, rows - 5)
    g = Table.grid(expand=True, padding=(0, 1))
    g.add_column(ratio=63); g.add_column(ratio=37)
    g.add_row(doc_pane(P, playhead, body_h), findings_pane(P, playhead, body_h, done))
    return Group(header(eng, note, dilate, cols), g,
                 status_strip(P, eng, playhead, tokens_done, elapsed * 1000, frac, done, cols))


def end_card(P, eng, note, link):
    from collections import Counter
    cats = Counter(e["type"] for e in P["ents"])
    g = Text()
    if note:
        g.append(note.upper() + "\n\n", style=f"bold {DIM}")
    g.append(f"{P['n_tokens']:,} tokens", style=f"bold {ACC}")
    g.append(f"  scanned in {eng['proc_s'] * 1000:,.0f} ms", style=f"bold {INK}")
    g.append(f"   ·   {eng['tps']:,.0f} tok/s\n", style=DIM)
    g.append(f"{len(P['ents'])} PII spans", style=f"bold {GOLD}")
    g.append(f"  across {len(cats)} categories\n\n", style=INK)
    top = cats.most_common(8)
    for i, (c, k) in enumerate(top):
        if i and i % 4 == 0:
            g.append("\n")
        g.append(f"{c} ", style=f"bold {cat_colour(c)}")
        g.append(f"{k}   ", style=DIM)
    g.append(f"\n\n{eng['device']}", style=DIM)
    if link:
        g.append(f"\n{link}", style=DIM)
    return Panel(Align.center(g, vertical="middle"), border_style=ACC_EMPTY,
                 box=box.ROUNDED, padding=(2, 6))


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--scene", default=str(here / "traces/scan"))
    ap.add_argument("--link", default="github.com/richiejp/privacy-filter.cpp")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dilate", type=float, default=1.0, help="time scale (1 = real time)")
    ap.add_argument("--hold", type=float, default=1.4)
    ap.add_argument("--card", type=float, default=3.5)
    a = ap.parse_args()

    scene = Path(a.scene)
    content = json.load(open(scene / "content.json"))
    eng = json.load(open(scene / "engines.json"))[0]
    eng["proc_s"] = content["n_tokens"] / eng["tps"]

    console = Console()
    cols, rows = console.size
    doc_w = max(24, int((cols - 4) * 0.63) - 4)
    P = prep(content, doc_w)
    P["find_w"] = max(20, int((cols - 4) * 0.37) - 4)

    wall_end = eng["proc_s"] * a.dilate + a.hold
    dt = 1.0 / a.fps
    note = content.get("note", "")
    with Live(console=console, refresh_per_second=a.fps, screen=True) as live:
        t0 = time.perf_counter()
        while (w := time.perf_counter() - t0) < wall_end:
            live.update(view(P, eng, note, w / a.dilate, a.dilate, cols, rows))
            time.sleep(dt)
        live.update(view(P, eng, note, eng["proc_s"], a.dilate, cols, rows))
        time.sleep(0.7)
        live.update(Panel(Align.center(end_card(P, eng, note, a.link), vertical="middle"),
                          border_style="black", box=box.SIMPLE, height=rows))
        time.sleep(a.card)


if __name__ == "__main__":
    main()
