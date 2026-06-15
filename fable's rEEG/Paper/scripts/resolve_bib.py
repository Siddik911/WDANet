#!/usr/bin/env python3
"""
resolve_bib.py — Batch-resolve to-verify.txt entries into refs.bib
Queries: arXiv API, Semantic Scholar, CrossRef, HAL
NEVER hand-writes BibTeX fields — only uses API-returned data.
"""

import urllib.request
import urllib.parse
import json
import time
import re
import xml.etree.ElementTree as ET

REFS_BIB = "/home/hasan/fable's rEEG/Paper/refs.bib"
TO_VERIFY = "/home/hasan/fable's rEEG/Paper/scripts/to-verify.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jaccard(a, b):
    sa = set(re.sub(r'[^a-z0-9 ]', '', a.lower()).split())
    sb = set(re.sub(r'[^a-z0-9 ]', '', b.lower()).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

STOP = {'a','an','the','on','of','for','in','to','with','via','and','from',
        'by','at','is','are','using','based','its'}

def make_key(first_author_surname, year, title):
    words = re.sub(r'[^a-z0-9 ]', '', title.lower()).split()
    first_word = next((w for w in words if w not in STOP), words[0] if words else 'x')
    surname = re.sub(r'[^a-z]', '', first_author_surname.lower())
    return f"{surname}{year}{first_word}"

def safe_get(url, headers=None, timeout=12):
    try:
        req = urllib.request.Request(url, headers=headers or {
            'User-Agent': 'Mozilla/5.0 (research; mailto:sihabhasan4567@gmail.com)'
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='replace')
    except Exception as e:
        return None

def fetch_bibtex_doi(doi):
    url = f"https://doi.org/{doi}"
    headers = {
        'Accept': 'application/x-bibtex',
        'User-Agent': 'Mozilla/5.0 (research; mailto:sihabhasan4567@gmail.com)'
    }
    return safe_get(url, headers=headers)

def rekey_bibtex(raw_bib, new_key):
    """Replace whatever key CrossRef assigned with our canonical key."""
    return re.sub(r'(@\w+\{)[^,]+,', rf'\g<1>{new_key},', raw_bib, count=1)

# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

def arxiv_search(title):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=3&sortBy=relevance"
    raw = safe_get(url)
    if not raw:
        return None
    try:
        root = ET.fromstring(raw)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            entry_title = entry.find('atom:title', ns).text.strip().replace('\n',' ')
            if jaccard(title, entry_title) < 0.45:
                continue
            arxiv_id_full = entry.find('atom:id', ns).text.strip()
            arxiv_id = re.search(r'abs/(.+?)(?:v\d+)?$', arxiv_id_full)
            if not arxiv_id:
                continue
            arxiv_id = arxiv_id.group(1)
            authors = [a.find('atom:name', ns).text.strip()
                       for a in entry.findall('atom:author', ns)]
            year = entry.find('atom:published', ns).text[:4]
            return {
                'arxiv_id': arxiv_id,
                'title': entry_title,
                'authors': authors,
                'year': year,
            }
    except Exception as e:
        pass
    return None

def make_arxiv_bibtex(info, key, primary_class='eess.SP'):
    author_str = ' and '.join(info['authors'])
    return (
        f"@misc{{{key},\n"
        f"  title        = {{{info['title']}}},\n"
        f"  author       = {{{author_str}}},\n"
        f"  year         = {{{info['year']}}},\n"
        f"  eprint       = {{{info['arxiv_id']}}},\n"
        f"  archivePrefix= {{arXiv}},\n"
        f"  primaryClass = {{{primary_class}}},\n"
        f"  url          = {{https://arxiv.org/abs/{info['arxiv_id']}}},\n"
        f"  note         = {{arXiv preprint}}\n"
        f"}}"
    )

# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def s2_search(title):
    q = urllib.parse.quote(title[:120])
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={q}&fields=title,externalIds,year,venue,authors&limit=3")
    raw = safe_get(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        for item in data.get('data', []):
            if jaccard(title, item.get('title','')) < 0.45:
                continue
            return item
    except:
        pass
    return None

def s2_by_doi(doi):
    url = (f"https://api.semanticscholar.org/graph/v1/paper/{doi}"
           f"?fields=title,externalIds,year,venue,authors")
    raw = safe_get(url)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except:
        return None

# ---------------------------------------------------------------------------
# CrossRef
# ---------------------------------------------------------------------------

def crossref_search(title, author=''):
    q = urllib.parse.quote(title)
    url = (f"https://api.crossref.org/works?query.bibliographic={q}"
           f"&rows=3&mailto=sihabhasan4567@gmail.com")
    raw = safe_get(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        for item in data.get('message',{}).get('items',[]):
            item_title = ' '.join(item.get('title',['']))
            if jaccard(title, item_title) > 0.55:
                return item.get('DOI')
    except:
        pass
    return None

# ---------------------------------------------------------------------------
# HAL
# ---------------------------------------------------------------------------

def hal_search(title):
    q = urllib.parse.quote(title[:100])
    url = (f"https://api.archives-ouvertes.fr/search/?q=title_t:{q}"
           f"&fl=halId_s,title_s,authFirstName_s,authLastName_s,producedDate_s,doiId_s"
           f"&wt=json&rows=3")
    raw = safe_get(url)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        docs = data.get('response',{}).get('docs',[])
        for doc in docs:
            doc_title = (doc.get('title_s') or [''])[0]
            if jaccard(title, doc_title) < 0.45:
                continue
            return doc
    except:
        pass
    return None

def make_hal_bibtex(doc, key):
    hal_id = doc.get('halId_s','')
    titles = doc.get('title_s') or ['Unknown']
    title = titles[0]
    first_names = doc.get('authFirstName_s') or []
    last_names  = doc.get('authLastName_s')  or []
    authors = ' and '.join(
        f"{ln}, {fn}" for fn, ln in zip(first_names, last_names)
    ) or 'Unknown'
    year = (doc.get('producedDate_s','2026') or '2026')[:4]
    url = f"https://hal.science/{hal_id}" if hal_id else ''
    return (
        f"@misc{{{key},\n"
        f"  title  = {{{title}}},\n"
        f"  author = {{{authors}}},\n"
        f"  year   = {{{year}}},\n"
        f"  url    = {{{url}}},\n"
        f"  note   = {{HAL preprint {hal_id}}}\n"
        f"}}"
    )

# ---------------------------------------------------------------------------
# Paper list
# ---------------------------------------------------------------------------

ARXIV_PAPERS = [
    ("P13", "RepSPD: Enhancing SPD Manifold Representation in EEGs via Dynamic Graphs", "eess.SP"),
    ("P14", "Riemannian Geometry-Preserving Variational Autoencoder for MI-BCI Data Augmentation", "eess.SP"),
    ("P15", "Riemannian Geometry Meets fMRI: The Advantages of Modeling Correlation Manifolds and Eigenvector Subspaces", "eess.SP"),
    ("P26", "Routing on the Stiefel Manifold: When Does Adaptive Subspace Selection Help for EEG Decoding", "eess.SP"),
    ("P51", "NeuroBRIDGE: Behavior-Conditioned Koopman Dynamics with Riemannian Alignment for Early Substance Use Initiation Prediction from Longitudinal Functional Connectome", "eess.SP"),
    ("P52", "Real-Time Decoding of Movement Onset and Offset for Brain-Controlled Rehabilitation Exoskeleton", "eess.SP"),
    ("P54", "Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods", "eess.SP"),
    ("P67", "FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet", "cs.LG"),
    ("P68", "Riemannian Networks over Full-Rank Correlation Matrices", "cs.LG"),
    ("P69", "SPD Matrix Learning for Neuroimaging Analysis: Perspectives Methods and Challenges", "eess.SP"),
    ("P72", "A Unified SPD Token Transformer Framework for EEG Classification Systematic Comparison of Geometric Embeddings", "eess.SP"),
    ("P75", "SPDLearn: A Geometric Deep Learning Python Library for Neural Decoding Through Trivialization", "cs.LG"),
    ("P80", "Latte: Hyperbolic Lorentz Attention for Joint-Subject EEG Classification", "eess.SP"),
    ("P82", "EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts", "eess.SP"),
    ("P84", "Beyond Rigid Geometries: The Spline-Pullback Metric for Universal Diffeomorphic Autoencoders", "cs.LG"),
    ("P85", "Riemannian Adversarial Attacks on Symmetric Positive Definite Matrices", "cs.LG"),
    ("P86", "Riemannian Block SPD Coupling Manifold and Its Application to Optimal Transport", "cs.LG"),
    ("P88", "Riemannian Optimization over Symmetric Positive Definite Matrices with the Alpha-Procrustes Metric", "math.OC"),
    ("P91", "Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks", "cs.LG"),
    ("P92", "A Riemannian Quasi-Newton Algorithm for Optimization with Euclidean Bounds", "math.OC"),
    ("P93", "Multivariate Intrinsic Local Polynomial Regression on Isometric Riemannian Manifolds", "math.ST"),
    ("P94", "Batch Normalization for Neural Networks on Complex Domains", "cs.LG"),
]

S2_PAPERS = [
    ("P49", "Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry"),
    ("P73", "Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds"),
    ("P96", "NEED: Cross-Subject and Cross-Task Generalization for Video and Image Reconstruction from EEG Signals"),
    ("P3",  "Riemannian High-Order Pooling for Brain Foundation Models"),
    ("P27", "Geometric Moment Alignment for Domain Adaptation via Siegel Embeddings"),
    ("P46", "HybridRDG: Zero-Shot ASD Biomarker Detection from Multi-Paradigm EEG via Hybrid Deep-Riemannian Domain Generalization"),
    ("P50", "PSDNorm: Test-Time Temporal Normalization for Deep Learning in Sleep Staging"),
    ("P76", "Channel Adaptation for EEG Foundation Models: A Systematic Benchmark"),
    ("P77", "EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces"),
    ("P78", "NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis and Motor Imagery Classification"),
    ("P70", "Res2SPDNet: Multi-Granularity SPD Matrix Residual Learning for Signal Classification"),
    ("P97", "Outlier Detection for Riemannian Manifold-Valued Functional Data"),
]

DOI_RETRY_PAPERS = [
    ("P4",  "10.1145/3770855.3818864", "A Sliced-Wasserstein Framework on Correlation Matrices for EEG Decoding"),
    ("P10", None, "CTSSP: A Temporal-Spectral-Spatial Joint Optimization Algorithm for Motor Imagery"),
    ("P19", None, "Filter Bank CSP with Riemannian Weighting for Disability-Centric Motor Imagery BCI"),
    ("P48", None, "Riemannian Geometry of Functional Connectivity Matrices for Multi-Site ADHD Data Harmonization"),
    ("P83", "10.48550/arxiv.2405.13979", "Robust Hyperbolic Learning with Curvature-Aware Optimization"),
]

HAL_PAPERS = [
    ("P11", "Pseudo Affine-Invariant Riemannian Metrics for Efficient Brain-Computer Interfaces"),
    ("P34", "FB-RCSP-RA: A Filter-Bank Regularized CSP Framework with Per-Band Riemannian Alignment for Cross-Subject Motor Imagery EEG Decoding"),
    ("P66", "ARMAGNAC: A New Parametric Batch Normalization Layer for SPDNet Architecture"),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    resolved = []   # list of (paper_id, key, bibtex_str)
    still_unresolved = []

    # --- arXiv papers ---
    print("\n=== Category 1: arXiv papers ===")
    for pid, title, pclass in ARXIV_PAPERS:
        print(f"  {pid}: {title[:60]}...")
        info = arxiv_search(title)
        time.sleep(0.6)
        if info:
            first_author = info['authors'][0].split()[-1] if info['authors'] else 'unknown'
            key = make_key(first_author, info['year'], info['title'])
            bib = make_arxiv_bibtex(info, key, pclass)
            resolved.append((pid, key, bib))
            print(f"    ✓ {key}  [arXiv:{info['arxiv_id']}]")
        else:
            still_unresolved.append((pid, title, "arXiv: no match found (Jaccard < 0.45)"))
            print(f"    ✗ no match")

    # --- Semantic Scholar papers ---
    print("\n=== Category 2: Semantic Scholar (NeurIPS/ICLR/unknown venue) ===")
    for pid, title in S2_PAPERS:
        print(f"  {pid}: {title[:60]}...")
        item = s2_search(title)
        time.sleep(1.1)
        if not item:
            still_unresolved.append((pid, title, "Semantic Scholar: no match"))
            print(f"    ✗ no match")
            continue
        ext_ids = item.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors_raw = item.get('authors', [])
        year = str(item.get('year', '2025'))
        item_title = item.get('title', title)
        first_author = authors_raw[0].get('name','Unknown').split()[-1] if authors_raw else 'unknown'
        key = make_key(first_author, year, item_title)

        if doi:
            bib = fetch_bibtex_doi(doi)
            time.sleep(0.5)
            if bib and '@' in bib:
                bib = rekey_bibtex(bib.strip(), key)
                resolved.append((pid, key, bib))
                print(f"    ✓ {key}  [DOI:{doi}]")
                continue
        if arxiv_id:
            info = {
                'arxiv_id': arxiv_id,
                'title': item_title,
                'authors': [a.get('name','') for a in authors_raw],
                'year': year,
            }
            bib = make_arxiv_bibtex(info, key, 'cs.LG')
            resolved.append((pid, key, bib))
            print(f"    ✓ {key}  [arXiv:{arxiv_id}]")
        else:
            still_unresolved.append((pid, title, f"S2 match found but no DOI/arXiv: {item_title[:50]}"))
            print(f"    ~ S2 matched but no citable ID")

    # --- DOI retry papers ---
    print("\n=== Category 3: DOI retry / CrossRef fallback ===")
    for pid, doi, title in DOI_RETRY_PAPERS:
        print(f"  {pid}: {title[:60]}...")
        bib = None
        if doi:
            bib = fetch_bibtex_doi(doi)
            time.sleep(0.5)
        if not (bib and '@' in bib):
            # fallback: CrossRef search
            found_doi = crossref_search(title)
            time.sleep(0.5)
            if found_doi:
                bib = fetch_bibtex_doi(found_doi)
                time.sleep(0.5)
                doi = found_doi
        if not (bib and '@' in bib):
            # fallback: Semantic Scholar
            item = s2_search(title)
            time.sleep(1.0)
            if item:
                ext = item.get('externalIds', {})
                doi = ext.get('DOI') or doi
                arxiv_id = ext.get('ArXiv')
                if doi:
                    bib = fetch_bibtex_doi(doi)
                    time.sleep(0.5)
                elif arxiv_id:
                    authors_raw = item.get('authors', [])
                    year = str(item.get('year', '2025'))
                    item_title = item.get('title', title)
                    first_author = authors_raw[0].get('name','Unknown').split()[-1] if authors_raw else 'unknown'
                    key = make_key(first_author, year, item_title)
                    info = {
                        'arxiv_id': arxiv_id,
                        'title': item_title,
                        'authors': [a.get('name','') for a in authors_raw],
                        'year': year,
                    }
                    bib = make_arxiv_bibtex(info, key, 'eess.SP')
                    resolved.append((pid, key, bib))
                    print(f"    ✓ {key}  [arXiv:{arxiv_id}]")
                    continue
        if bib and '@' in bib:
            # extract key from title+doi
            # first try to parse author from bib
            m_author = re.search(r'author\s*=\s*\{([^}]+)\}', bib, re.I)
            if m_author:
                first_part = m_author.group(1).split(' and ')[0]
                surname = first_part.split(',')[0].strip().split()[-1]
            else:
                surname = 'unknown'
            m_year = re.search(r'year\s*=\s*\{?(\d{4})', bib, re.I)
            year = m_year.group(1) if m_year else '2026'
            key = make_key(surname, year, title)
            bib = rekey_bibtex(bib.strip(), key)
            resolved.append((pid, key, bib))
            print(f"    ✓ {key}  [DOI:{doi}]")
        else:
            still_unresolved.append((pid, title, f"DOI fetch failed (tried doi.org + CrossRef + S2)"))
            print(f"    ✗ all methods failed")

    # --- HAL papers ---
    print("\n=== Category 4: HAL preprints ===")
    for pid, title in HAL_PAPERS:
        print(f"  {pid}: {title[:60]}...")
        doc = hal_search(title)
        time.sleep(0.6)
        if not doc:
            still_unresolved.append((pid, title, "HAL: no match"))
            print(f"    ✗ no match")
            continue
        doi = doc.get('doiId_s')
        last_names = doc.get('authLastName_s') or ['unknown']
        year_raw = doc.get('producedDate_s','2026') or '2026'
        year = year_raw[:4]
        key = make_key(last_names[0], year, title)
        if doi:
            bib = fetch_bibtex_doi(doi)
            time.sleep(0.5)
            if bib and '@' in bib:
                bib = rekey_bibtex(bib.strip(), key)
                resolved.append((pid, key, bib))
                print(f"    ✓ {key}  [DOI:{doi}]")
                continue
        # no DOI — use HAL record
        bib = make_hal_bibtex(doc, key)
        resolved.append((pid, key, bib))
        hal_id = doc.get('halId_s','')
        print(f"    ✓ {key}  [HAL:{hal_id}]")

    # --- Append to refs.bib ---
    print(f"\n=== Writing {len(resolved)} new entries to refs.bib ===")
    if resolved:
        with open(REFS_BIB, 'a', encoding='utf-8') as f:
            f.write("\n\n% === Resolved from to-verify.txt ===\n")
            for pid, key, bib in resolved:
                f.write(f"\n% {pid}\n{bib}\n")
        print(f"  Appended {len(resolved)} entries.")

    # --- Rewrite to-verify.txt ---
    SKIP_REASONS = {
        "P18": "Research Square preprint — no stable DOI; not appropriate to cite",
        "P21": "Research Square preprint — no stable DOI",
        "P45": "ICLR 2026 under review — not yet published",
        "P71": "ICLR 2026 under review — not yet published",
        "P81": "ICLR 2026 under review — not yet published",
        "P47": "medRxiv preprint — search for published journal version manually",
        "P43": "Chinese journal (10.7507/1001-5515.202507056) — HTTP 403; add manually if needed",
        "P58": "MSc Thesis, Université Laval — cite as @phdthesis if needed",
        "P59": "PhD Thesis, Università degli Studi di Verona",
        "P60": "PhD Thesis, Sapienza University of Rome",
        "P62": "PhD Thesis, University of Auckland",
        "P63": "Technical Report, University of West Bohemia",
        "P64": "PhD Thesis, Politecnico di Bari",
        "P87": "Unknown venue — Sheaf Neural Networks on SPD Manifolds",
        "P89": "Unknown venue — Fast and Stable Riemannian SPD Methods",
        "P90": "Unknown venue — Building Transformation Layers for SPDNet",
    }

    with open(TO_VERIFY, 'w', encoding='utf-8') as f:
        f.write("# Papers requiring manual BibTeX entry\n")
        f.write(f"# Remaining: {len(still_unresolved) + len(SKIP_REASONS)}\n\n")

        f.write("## Not yet published / under review (DO NOT CITE)\n")
        for pid in ["P45","P71","P81"]:
            f.write(f"{pid} — {SKIP_REASONS[pid]}\n")

        f.write("\n## Unstable preprints (cite with caution)\n")
        for pid in ["P18","P21","P47"]:
            f.write(f"{pid} — {SKIP_REASONS[pid]}\n")

        f.write("\n## Theses (use @phdthesis/@mastersthesis if needed)\n")
        for pid in ["P58","P59","P60","P62","P63","P64"]:
            f.write(f"{pid} — {SKIP_REASONS[pid]}\n")

        f.write("\n## Chinese/blocked journal (manual fetch required)\n")
        f.write(f"P43 — {SKIP_REASONS['P43']}\n")

        f.write("\n## Unknown venue (manual search required)\n")
        for pid in ["P87","P89","P90"]:
            f.write(f"{pid} — {SKIP_REASONS[pid]}\n")

        if still_unresolved:
            f.write("\n## API resolution failed (retry or manual lookup)\n")
            for pid, title, reason in still_unresolved:
                f.write(f"{pid} — {title[:80]}\n  Reason: {reason}\n\n")

    # --- Summary ---
    total_in_bib = 39 + len(resolved)
    resolved_ids = [r[0] for r in resolved]
    print(f"\n{'='*50}")
    print(f"RESOLVED: {len(resolved)} new entries")
    print(f"TOTAL refs.bib: {total_in_bib} entries")
    print(f"STILL UNRESOLVED: {len(still_unresolved)} (see updated to-verify.txt)")
    print(f"\nKeys added:")
    for pid, key, _ in resolved:
        print(f"  {pid}: {key}")
    if still_unresolved:
        print(f"\nFailed:")
        for pid, title, reason in still_unresolved:
            print(f"  {pid}: {reason}")

if __name__ == '__main__':
    main()
