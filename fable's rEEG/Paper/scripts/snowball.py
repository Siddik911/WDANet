#!/usr/bin/env python3
"""
snowball.py — citation chaining via OpenAlex to find primary sources for claims.

USAGE
-----
  # Find primary sources for a specific claim via one seed paper:
  python3 snowball.py --key shen2026wdanet --claim "riemannian outperforms euclidean cross-subject"

  # Expand all seeds in seed_dois.txt and write candidates:
  python3 snowball.py --all

  # Add a new seed DOI interactively:
  python3 snowball.py --add-seed 10.3389/fnhum.2017.00568

RULES (enforced)
----------------
  - Only snowball to verify/find primary sources for claims already in claims.csv
  - Never snowball to discover new claims to add to the paper
  - Stop at 2 citation hops — do not recurse further
  - Cite the primary source where possible; if unverifiable, mark SECONDARY in claims.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR    = os.path.dirname(SCRIPT_DIR)
REFS_BIB     = os.path.join(PAPER_DIR, "refs.bib")
CLAIMS_CSV   = os.path.join(PAPER_DIR, "claims.csv")
SEED_FILE    = os.path.join(SCRIPT_DIR, "seed_dois.txt")
CANDIDATES   = os.path.join(SCRIPT_DIR, "snowball_candidates.csv")

MAILTO = "sihabhasan4567@gmail.com"

# EEG/depression/Riemannian relevance keywords (title or abstract match → keep)
RELEVANCE_KEYWORDS = {
    "eeg", "electroencephalogram", "depression", "riemannian", "spd",
    "covariance", "brain-computer", "bci", "domain adaptation", "cross-subject",
    "cross-dataset", "transfer learning", "manifold", "positive definite",
    "dann", "mental health", "affective", "neural", "motor imagery",
}

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def safe_get(url, timeout=12):
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"snowball/1.0 (mailto:{MAILTO})"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return None


def fetch_json(url):
    raw = safe_get(url)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# refs.bib helpers
# ---------------------------------------------------------------------------

def load_bib_keys():
    """Return dict: key -> DOI (lowercase, no https://doi.org/)"""
    keys = {}
    if not os.path.exists(REFS_BIB):
        return keys
    with open(REFS_BIB) as f:
        content = f.read()
    for m in re.finditer(r'@\w+\{(\w+),.*?(?=\n@|\Z)', content, re.DOTALL):
        block = m.group(0)
        key = m.group(1)
        dm = re.search(r'DOI\s*=\s*\{([^}]+)\}', block, re.I)
        ep = re.search(r'eprint\s*=\s*\{([^}]+)\}', block, re.I)
        doi = dm.group(1).lower().strip() if dm else None
        arxiv = ep.group(1).strip() if ep else None
        keys[key] = {"doi": doi, "arxiv": arxiv}
    return keys


def doi_to_bib_key(doi, bib_keys):
    """Find existing bib key for a DOI (case-insensitive)."""
    doi_norm = doi.lower().strip()
    for key, meta in bib_keys.items():
        if meta.get("doi") and meta["doi"] == doi_norm:
            return key
    return None

# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def openalex_work(doi):
    """Fetch OpenAlex work metadata by DOI."""
    doi_enc = urllib.parse.quote(doi, safe="")
    url = f"https://api.openalex.org/works/doi:{doi_enc}?mailto={MAILTO}"
    return fetch_json(url)


def openalex_work_by_id(oa_id):
    """Fetch OpenAlex work metadata by OpenAlex ID (e.g. W1234567890)."""
    short = oa_id.split("/")[-1]
    url = f"https://api.openalex.org/works/{short}?mailto={MAILTO}"
    return fetch_json(url)


def is_relevant(work):
    """Return True if the work matches our relevance keywords."""
    text = " ".join([
        work.get("title") or "",
        (work.get("abstract_inverted_index") and " ".join(work["abstract_inverted_index"].keys())) or "",
        " ".join(c.get("display_name","") for c in work.get("concepts",[]))
    ]).lower()
    return any(kw in text for kw in RELEVANCE_KEYWORDS)


def get_references(doi):
    """
    Fetch all references of a paper (1-hop backward snowball).
    Returns list of dicts: {oa_id, doi, title, year, cited_by_count, relevant}
    """
    work = openalex_work(doi)
    if not work:
        print(f"  [!] OpenAlex: no record for DOI {doi}")
        return []

    ref_ids = work.get("referenced_works", [])
    print(f"  Found {len(ref_ids)} references — fetching relevant ones...")

    results = []
    for i, oa_id in enumerate(ref_ids):
        time.sleep(0.12)  # polite pool: ~8 req/s
        ref = openalex_work_by_id(oa_id)
        if not ref:
            continue
        ref_doi = (ref.get("doi") or "").replace("https://doi.org/", "").lower()
        title = ref.get("title") or ""
        year = ref.get("publication_year") or ""
        cbc = ref.get("cited_by_count") or 0
        relevant = is_relevant(ref)
        results.append({
            "oa_id": oa_id,
            "doi": ref_doi,
            "title": title,
            "year": year,
            "cited_by_count": cbc,
            "relevant": relevant,
        })

    results.sort(key=lambda x: -x["cited_by_count"])
    return results

# ---------------------------------------------------------------------------
# Claim matching
# ---------------------------------------------------------------------------

def jaccard(a, b):
    sa = set(re.sub(r'[^a-z0-9 ]', '', a.lower()).split())
    sb = set(re.sub(r'[^a-z0-9 ]', '', b.lower()).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def match_claim_to_references(claim_text, references, top_n=5):
    """
    Rank references by title similarity to the claim.
    Real citation-context matching requires full-text; this is a proxy.
    """
    scored = []
    for ref in references:
        if not ref["relevant"]:
            continue
        score = jaccard(claim_text, ref["title"])
        scored.append((score, ref))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_n]

# ---------------------------------------------------------------------------
# BibTeX fetch helper
# ---------------------------------------------------------------------------

STOP = {"a","an","the","on","of","for","in","to","with","via","and","from","by","at"}

def make_key(surname, year, title):
    words = re.sub(r'[^a-z0-9 ]', '', title.lower()).split()
    fw = next((w for w in words if w not in STOP), words[0] if words else "x")
    sn = re.sub(r'[^a-z]', '', surname.lower())
    return f"{sn}{year}{fw}"


def fetch_bibtex_doi(doi):
    url = f"https://doi.org/{doi}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/x-bibtex",
                 "User-Agent": f"snowball/1.0 (mailto:{MAILTO})"}
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def rekey(bib, key):
    return re.sub(r'(@\w+\{)[^,]+,', rf'\g<1>{key},', bib, count=1)


def add_to_refs_bib(doi, label="snowball"):
    """Fetch BibTeX for doi and append to refs.bib. Returns the key used."""
    bib = fetch_bibtex_doi(doi)
    if not bib or "@" not in bib:
        return None
    # Extract author/year/title for key generation
    m_author = re.search(r'author\s*=\s*\{([^}]+)\}', bib, re.I)
    m_year   = re.search(r'year\s*=\s*\{?(\d{4})', bib, re.I)
    m_title  = re.search(r'title\s*=\s*\{([^}]+)\}', bib, re.I)
    surname = "unknown"
    if m_author:
        first = m_author.group(1).split(" and ")[0]
        surname = first.split(",")[0].strip().split()[-1]
    year  = m_year.group(1) if m_year else "2026"
    title = m_title.group(1) if m_title else doi
    key = make_key(surname, year, title)
    bib = rekey(bib.strip(), key)
    with open(REFS_BIB, "a") as f:
        f.write(f"\n% Added by snowball.py ({label})\n{bib}\n")
    return key

# ---------------------------------------------------------------------------
# Seed file helpers
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [
    # These are the most-cited foundational papers in our refs.bib
    ("10.1109/tbme.2011.2172210",  "barachant2012multiclass",   "Barachant 2012 MDM classifier"),
    ("10.3389/fnhum.2021.595723",  "wu2021new",                 "Wu 2021 tangent space BCI"),
    ("10.1109/taffc.2025.3646189", "shen2026wdanet",            "WDANet depression cross-domain"),
    ("10.3390/math14081327",       "feng2026decoding",           "EEG-RCformer depression MODMA"),
    ("10.1016/j.neunet.2026.109076","cheng2026spd",             "SPD-DANN domain adaptation"),
]


def load_seeds():
    seeds = list(DEFAULT_SEEDS)
    if os.path.exists(SEED_FILE):
        with open(SEED_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("|", 2)
                    doi = parts[0].strip()
                    key = parts[1].strip() if len(parts) > 1 else ""
                    note = parts[2].strip() if len(parts) > 2 else ""
                    seeds.append((doi, key, note))
    return seeds


def save_seed(doi, key="", note=""):
    with open(SEED_FILE, "a") as f:
        f.write(f"{doi} | {key} | {note}\n")

# ---------------------------------------------------------------------------
# Claims CSV
# ---------------------------------------------------------------------------

def load_claims():
    if not os.path.exists(CLAIMS_CSV):
        return []
    with open(CLAIMS_CSV, newline="") as f:
        return list(csv.DictReader(f))

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_claim(bib_key, claim_text):
    """
    Given a bib key and claim text, find which references of that paper
    best support the claim. Suggest whether to cite primary or keep secondary.
    """
    bib_keys = load_bib_keys()
    meta = bib_keys.get(bib_key)
    if not meta:
        print(f"[!] Key '{bib_key}' not found in refs.bib")
        sys.exit(1)
    doi = meta.get("doi")
    if not doi:
        print(f"[!] Key '{bib_key}' has no DOI — cannot snowball arXiv-only entries")
        sys.exit(1)

    print(f"\nSnowballing: {bib_key} (DOI: {doi})")
    print(f"Claim: \"{claim_text}\"\n")

    refs = get_references(doi)
    if not refs:
        print("No references retrieved.")
        return

    relevant = [r for r in refs if r["relevant"]]
    print(f"Relevant references (EEG/Riemannian/depression): {len(relevant)} of {len(refs)}\n")

    matches = match_claim_to_references(claim_text, relevant, top_n=8)

    print("=" * 70)
    print("TOP MATCHES FOR CLAIM")
    print("=" * 70)
    for score, ref in matches:
        in_bib = doi_to_bib_key(ref["doi"], bib_keys)
        status = f"✓ IN refs.bib → \\cite{{{in_bib}}}" if in_bib else "✗ NOT in refs.bib"
        print(f"\n  [{score:.2f}] {ref['title'][:65]}")
        print(f"         Year: {ref['year']} | Cited by: {ref['cited_by_count']} | DOI: {ref['doi']}")
        print(f"         {status}")
        if not in_bib and ref["doi"] and score > 0.1:
            print(f"         → Add: python3 snowball.py --add-doi {ref['doi']}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    already_have = [(s, r) for s, r in matches if doi_to_bib_key(r["doi"], bib_keys)]
    not_have     = [(s, r) for s, r in matches if not doi_to_bib_key(r["doi"], bib_keys) and r["doi"]]

    if already_have:
        best_score, best_ref = already_have[0]
        best_key = doi_to_bib_key(best_ref["doi"], bib_keys)
        print(f"  Primary source already in refs.bib: \\cite{{{best_key}}}")
        print(f"  → Use \\cite{{{best_key}}} instead of \\cite{{{bib_key}}} for this claim")
        if best_score < 0.15:
            print(f"  ⚠ Low similarity ({best_score:.2f}) — verify manually that this paper")
            print(f"    actually supports \"{claim_text[:50]}...\"")
    elif not_have:
        best_score, best_ref = not_have[0]
        print(f"  Best primary candidate NOT in refs.bib:")
        print(f"    \"{best_ref['title'][:60]}\"")
        print(f"    DOI: {best_ref['doi']} | Year: {best_ref['year']} | Cited: {best_ref['cited_by_count']}")
        print(f"  → Verify abstract supports the claim, then run:")
        print(f"    python3 snowball.py --add-doi {best_ref['doi']}")
    else:
        print(f"  No strong primary source found. Keep \\cite{{{bib_key}}} as secondary.")
        print(f"  Mark as SECONDARY in claims.csv.")


def cmd_all():
    """
    Expand all seeds, collect relevant references, write snowball_candidates.csv.
    Use this to discover candidate papers for the literature review — but only
    add them to refs.bib if they are needed for a specific claim.
    """
    seeds = load_seeds()
    bib_keys = load_bib_keys()
    all_refs = {}  # doi -> ref dict

    for doi, key, note in seeds:
        print(f"\n[seed] {key or doi} — {note}")
        refs = get_references(doi)
        time.sleep(0.3)
        for ref in refs:
            if ref["doi"] and ref["doi"] not in all_refs:
                all_refs[ref["doi"]] = ref

    relevant = [r for r in all_refs.values() if r["relevant"]]
    relevant.sort(key=lambda x: -x["cited_by_count"])

    with open(CANDIDATES, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doi","title","year","cited_by_count","in_refs_bib"])
        writer.writeheader()
        for ref in relevant:
            in_bib = doi_to_bib_key(ref["doi"], bib_keys) or ""
            writer.writerow({
                "doi": ref["doi"],
                "title": ref["title"][:100],
                "year": ref["year"],
                "cited_by_count": ref["cited_by_count"],
                "in_refs_bib": in_bib,
            })

    already = sum(1 for r in relevant if doi_to_bib_key(r["doi"], bib_keys))
    print(f"\n{'='*60}")
    print(f"Relevant candidates found: {len(relevant)}")
    print(f"Already in refs.bib:       {already}")
    print(f"New candidates:            {len(relevant) - already}")
    print(f"Written to: {CANDIDATES}")
    print(f"\n⚠  Only add new candidates if needed for a specific claim in claims.csv.")


def cmd_add_doi(doi):
    """Fetch BibTeX for a DOI and append to refs.bib."""
    bib_keys = load_bib_keys()
    existing = doi_to_bib_key(doi, bib_keys)
    if existing:
        print(f"Already in refs.bib as: {existing}")
        return
    print(f"Fetching BibTeX for {doi}...")
    key = add_to_refs_bib(doi, label="snowball --add-doi")
    if key:
        print(f"✓ Added to refs.bib as: {key}")
        print(f"  Remember to add a claims.csv row before using \\cite{{{key}}}")
    else:
        print("✗ Could not fetch BibTeX. Try manually:")
        print(f"  curl -LH 'Accept: application/x-bibtex' https://doi.org/{doi}")


def cmd_add_seed(doi, note=""):
    save_seed(doi, note=note)
    print(f"Added seed: {doi}")


def cmd_status():
    """Show current snowball state: seeds, refs.bib count, candidates."""
    bib_keys = load_bib_keys()
    seeds = load_seeds()
    claims = load_claims()
    print(f"refs.bib entries:        {len(bib_keys)}")
    print(f"Seed DOIs:               {len(seeds)}")
    print(f"claims.csv rows:         {len(claims)}")
    has_cands = os.path.exists(CANDIDATES)
    if has_cands:
        with open(CANDIDATES) as f:
            n = sum(1 for _ in f) - 1
        print(f"snowball_candidates.csv: {n} rows (run --all to refresh)")
    else:
        print(f"snowball_candidates.csv: not yet generated (run --all)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Citation chaining via OpenAlex — find primary sources for claims"
    )
    sub = parser.add_subparsers(dest="cmd")

    # --key / --claim (most common)
    p1 = parser.add_argument_group("Claim mode (most common)")
    parser.add_argument("--key",   metavar="BIB_KEY",
                        help="bib key of the paper you're currently citing (e.g. shen2026wdanet)")
    parser.add_argument("--claim", metavar="TEXT",
                        help="the claim sentence you need to support")

    # --all
    parser.add_argument("--all", action="store_true",
                        help="Expand all seed DOIs → write snowball_candidates.csv")

    # --add-doi
    parser.add_argument("--add-doi", metavar="DOI",
                        help="Fetch BibTeX for DOI and append to refs.bib")

    # --add-seed
    parser.add_argument("--add-seed", metavar="DOI",
                        help="Add a DOI to the persistent seed list")
    parser.add_argument("--note", metavar="TEXT", default="",
                        help="Note for --add-seed")

    # --status
    parser.add_argument("--status", action="store_true",
                        help="Show current state")

    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.add_doi:
        cmd_add_doi(args.add_doi)
    elif args.add_seed:
        cmd_add_seed(args.add_seed, note=args.note)
    elif args.all:
        cmd_all()
    elif args.key and args.claim:
        cmd_claim(args.key, args.claim)
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python3 snowball.py --key shen2026wdanet --claim "riemannian outperforms euclidean cross-subject"')
        print('  python3 snowball.py --key feng2026decoding --claim "SPD covariance features depression detection"')
        print("  python3 snowball.py --all")
        print("  python3 snowball.py --add-doi 10.3389/fnhum.2017.00568")
        print("  python3 snowball.py --status")

if __name__ == "__main__":
    main()
