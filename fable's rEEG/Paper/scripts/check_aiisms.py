#!/usr/bin/env python3
"""
check_aiisms.py — scan section tex files for forbidden AI-ism phrases and
flag overly uniform sentence lengths.

Forbidden phrases (from CLAUDE.md AI-ism watchlist):
    delve, leverage (as verb), it is worth noting, in conclusion,
    moreover (overuse — flag if appears >2 times in a section),
    Furthermore (overuse — flag if appears >2 times in a section)

Sentence-length rule:
    Flag any section where the AVERAGE sentence word-count exceeds 30 words
    (proxy for monotonously long, dense prose).

Usage:
    python scripts/check_aiisms.py [--strict]

Exit codes:
    0 — no issues found
    1 — one or more issues found (use --strict to enforce exit 1)
"""

import re
import sys
from pathlib import Path

SECTIONS_DIR = Path(__file__).parent.parent / "sections"

# ---------------------------------------------------------------------------
# Forbidden phrase patterns (case-insensitive)
# ---------------------------------------------------------------------------
ALWAYS_FORBIDDEN = [
    r"\bdelve\b",
    r"\bit is worth noting\b",
    r"\bin conclusion\b",
]

# Overuse: flag if count > threshold in a single file
OVERUSE = {
    r"\bmoreover\b": 2,
    r"\bfurthermore\b": 2,
    r"\bleverage\b": 1,   # flag any use as a verb
}

# ---------------------------------------------------------------------------
# Sentence splitter (naive but sufficient for LaTeX prose)
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """Split on . ! ? preceded by a word character, ignoring common abbreviations."""
    # Strip LaTeX commands for cleaner word counting
    clean = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)   # \cmd{arg}
    clean = re.sub(r'\\[a-zA-Z]+', ' ', clean)            # \cmd
    clean = re.sub(r'%.*', '', clean)                      # comments
    clean = re.sub(r'\s+', ' ', clean)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean)
    return [s.strip() for s in sentences if s.strip()]

def word_count(sentence: str) -> int:
    return len(sentence.split())

# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------
issues: list[str] = []

tex_files = sorted(SECTIONS_DIR.glob("*.tex"))
if not tex_files:
    print(f"No .tex files found in {SECTIONS_DIR}")
    sys.exit(0)

for tf in tex_files:
    text = tf.read_text(encoding="utf-8", errors="replace")
    # Strip comment lines for phrase scanning
    text_no_comments = re.sub(r'(?m)^%.*$', '', text)
    fname = tf.name

    # -- Always-forbidden phrases --
    for pattern in ALWAYS_FORBIDDEN:
        for m in re.finditer(pattern, text_no_comments, re.IGNORECASE):
            snippet = text_no_comments[max(0, m.start()-30):m.end()+30].replace('\n', ' ')
            issues.append(f"[{fname}] FORBIDDEN '{m.group()}': ...{snippet}...")

    # -- Overuse patterns --
    for pattern, threshold in OVERUSE.items():
        matches = list(re.finditer(pattern, text_no_comments, re.IGNORECASE))
        if len(matches) > threshold:
            issues.append(
                f"[{fname}] OVERUSE '{pattern}' appears {len(matches)} times "
                f"(threshold: {threshold})"
            )

    # -- Average sentence length --
    sentences = split_sentences(text_no_comments)
    prose_sentences = [s for s in sentences if word_count(s) > 3]
    if prose_sentences:
        avg_len = sum(word_count(s) for s in prose_sentences) / len(prose_sentences)
        if avg_len > 30:
            issues.append(
                f"[{fname}] HIGH avg sentence length: {avg_len:.1f} words "
                f"(threshold: 30) — vary short declarative with longer analytical"
            )

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print(f"Scanned {len(tex_files)} section file(s).")
if issues:
    print(f"\nFOUND {len(issues)} issue(s):\n")
    for issue in issues:
        print(f"  {issue}")
    if "--strict" in sys.argv:
        sys.exit(1)
    else:
        print("\n(Run with --strict to exit 1 on issues.)")
        sys.exit(0)
else:
    print("PASS — no AI-isms or sentence-length issues detected.")
    sys.exit(0)
