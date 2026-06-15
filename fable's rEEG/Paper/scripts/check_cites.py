#!/usr/bin/env python3
"""
check_cites.py — verify every \\cite{} key in the paper exists in refs.bib.

Usage:
    python scripts/check_cites.py

Exit codes:
    0 — all cite keys resolved
    1 — one or more cite keys missing from refs.bib

Run this before every commit (CLAUDE.md Rule 1).
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root — Paper/)
# ---------------------------------------------------------------------------
PAPER_ROOT = Path(__file__).parent.parent          # Paper/
SECTIONS_DIR = PAPER_ROOT / "sections"
MAIN_TEX = PAPER_ROOT / "main.tex"
REFS_BIB = PAPER_ROOT / "refs.bib"

# ---------------------------------------------------------------------------
# Step 1: collect all \cite{...} and \citep{...} keys from tex files
# ---------------------------------------------------------------------------
CITE_RE = re.compile(r'\\cite[pt]?\{([^}]+)\}')

def extract_cite_keys(tex_path: Path) -> set[str]:
    """Return all citation keys referenced in a .tex file."""
    keys: set[str] = set()
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    for match in CITE_RE.finditer(text):
        # A single \cite{} can hold comma-separated keys
        for key in match.group(1).split(","):
            keys.add(key.strip())
    return keys

tex_files = [MAIN_TEX] + sorted(SECTIONS_DIR.glob("*.tex"))
all_cite_keys: set[str] = set()
for tf in tex_files:
    if tf.exists():
        all_cite_keys |= extract_cite_keys(tf)

# ---------------------------------------------------------------------------
# Step 2: collect all @TYPE{key, ...} entries from refs.bib
# ---------------------------------------------------------------------------
BIB_KEY_RE = re.compile(r'@\w+\{([^,\s]+)\s*,', re.IGNORECASE)

def extract_bib_keys(bib_path: Path) -> set[str]:
    """Return all entry keys defined in a .bib file."""
    keys: set[str] = set()
    if not bib_path.exists():
        return keys
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    for match in BIB_KEY_RE.finditer(text):
        keys.add(match.group(1).strip())
    return keys

bib_keys = extract_bib_keys(REFS_BIB)

# ---------------------------------------------------------------------------
# Step 3: diff and report
# ---------------------------------------------------------------------------
missing = sorted(all_cite_keys - bib_keys)

if not all_cite_keys:
    print("INFO: No \\cite{} keys found in tex files.")
    sys.exit(0)

print(f"Cite keys found in tex files : {len(all_cite_keys)}")
print(f"Keys defined in refs.bib     : {len(bib_keys)}")

if missing:
    print(f"\nFAIL — {len(missing)} missing key(s):")
    for k in missing:
        print(f"  MISSING: {k}")
    sys.exit(1)
else:
    print("\nPASS — all cite keys resolved.")
    sys.exit(0)
