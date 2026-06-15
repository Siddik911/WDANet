#!/usr/bin/env python3
"""Fetch BibTeX via Semantic Scholar for Cat2, Cat3, Cat4."""
import urllib.request
import urllib.parse
import json
import time
import re
import subprocess

def jaccard(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def make_key(authors, year, title):
    if not authors:
        return f"unknown{year}key"
    first_author = authors[0]
    surname = first_author.get('name', 'unknown').strip().split()[-1]
    surname = re.sub(r'[^a-z]', '', surname.lower())

    skip = {'a', 'an', 'the', 'on', 'of', 'for', 'in', 'to', 'and', 'with', 'via', 'by', 'from', 'using'}
    words = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().split()
    firstword = 'paper'
    for w in words:
        if w not in skip and len(w) > 1:
            firstword = re.sub(r'[^a-z0-9]', '', w)
            break
    return f"{surname}{year}{firstword}"

def make_key_str(author_str, year, title):
    """For string author format."""
    parts = author_str.strip().split()
    surname = re.sub(r'[^a-z]', '', parts[-1].lower()) if parts else 'unknown'
    skip = {'a', 'an', 'the', 'on', 'of', 'for', 'in', 'to', 'and', 'with', 'via', 'by', 'from', 'using'}
    words = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().split()
    firstword = 'paper'
    for w in words:
        if w not in skip and len(w) > 1:
            firstword = re.sub(r'[^a-z0-9]', '', w)
            break
    return f"{surname}{year}{firstword}"

def ss_search(title, limit=3):
    """Semantic Scholar title search."""
    encoded = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&fields=title,externalIds,year,venue,authors,publicationVenue&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('data', [])
    except Exception as e:
        print(f"  SS search ERROR for '{title[:50]}': {e}")
        return []

def ss_by_doi(doi):
    """Semantic Scholar lookup by DOI."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(doi)}?fields=title,authors,year,venue,externalIds,publicationVenue"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  SS DOI lookup ERROR for '{doi}': {e}")
        return None

def fetch_doi_bibtex(doi):
    """Fetch BibTeX via doi.org content negotiation."""
    url = f"https://doi.org/{doi}"
    try:
        req = urllib.request.Request(url, headers={
            'Accept': 'application/x-bibtex',
            'User-Agent': 'Mozilla/5.0'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode('utf-8').strip()
    except Exception as e:
        print(f"  DOI BibTeX ERROR for '{doi}': {e}")
        return None

def crossref_search(query, rows=3):
    """Search CrossRef."""
    encoded = urllib.parse.quote(query)
    url = f"https://api.crossref.org/works?query.bibliographic={encoded}&rows={rows}"
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'MyApp/1.0 (mailto:research@example.com)'
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('message', {}).get('items', [])
    except Exception as e:
        print(f"  CrossRef ERROR for '{query[:50]}': {e}")
        return []

def make_misc_bibtex(key, title, authors, year, arxiv_id, primary_class='cs.LG'):
    """Generate @misc for arXiv."""
    if isinstance(authors, list) and authors and isinstance(authors[0], dict):
        author_str = ' and '.join(a.get('name', '') for a in authors)
    elif isinstance(authors, list):
        author_str = ' and '.join(authors)
    else:
        author_str = str(authors)
    return f"""@misc{{{key},
  title        = {{{title}}},
  author       = {{{author_str}}},
  year         = {{{year}}},
  eprint       = {{{arxiv_id}}},
  archivePrefix= {{arXiv}},
  primaryClass = {{{primary_class}}},
  url          = {{https://arxiv.org/abs/{arxiv_id}}},
  note         = {{arXiv preprint}}
}}"""

resolved = {}
unresolved_details = {}

print("=" * 60)
print("CATEGORY 2: NeurIPS 2025 / ICLR 2026 via Semantic Scholar")
print("=" * 60)

# P83 - known arXiv ID, generate directly
print("\nP83: Robust Hyperbolic Learning with Curvature-Aware Optimization")
bib_p83 = make_misc_bibtex(
    'gao2025robust',
    'Robust Hyperbolic Learning with Curvature-Aware Optimization',
    [{'name': 'Gao unknown'}],  # will be replaced by SS lookup
    '2025', '2405.13979', 'cs.LG'
)
# Look up actual authors from SS
r = ss_by_doi('arXiv:2405.13979')
if not r:
    # try search
    results = ss_search('Robust Hyperbolic Learning Curvature-Aware Optimization')
    if results:
        r = results[0]
if r:
    authors = r.get('authors', [])
    title = r.get('title', 'Robust Hyperbolic Learning with Curvature-Aware Optimization')
    year = str(r.get('year', 2025))
    arxiv_id = r.get('externalIds', {}).get('ArXiv', '2405.13979')
    key = make_key(authors, year, title)
    bib_p83 = make_misc_bibtex(key, title, authors, year, arxiv_id, 'cs.LG')
    print(f"  -> RESOLVED: key={key}, arXiv={arxiv_id}")
    resolved['P83'] = {'key': key, 'bib': bib_p83, 'title': title}
else:
    print(f"  -> Using fallback with eprint=2405.13979")
    resolved['P83'] = {'key': 'gao2025robust', 'bib': bib_p83, 'title': 'Robust Hyperbolic Learning with Curvature-Aware Optimization'}
time.sleep(1)

# Search all other Cat2 papers
cat2_papers = [
    ('P49', 'Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry', 'NeurIPS 2025', 'cs.LG'),
    ('P73', 'Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds', 'NeurIPS 2025', 'cs.LG'),
    ('P96', 'NEED: Cross-Subject and Cross-Task Generalization for Video and Image Reconstruction from EEG Signals', 'NeurIPS 2025', 'eess.SP'),
    ('P3', 'Riemannian High-Order Pooling for Brain Foundation Models', 'ICLR 2026', 'eess.SP'),
    ('P27', 'Geometric Moment Alignment for Domain Adaptation via Siegel Embeddings', 'ICLR 2026', 'cs.LG'),
    ('P46', 'HybridRDG: Zero-Shot ASD Biomarker Detection from Multi-Paradigm EEG via Hybrid Deep-Riemannian Domain Generalization', 'CVPR Workshop 2026', 'eess.SP'),
    ('P50', 'PSDNorm: Temporal Normalization for Deep Learning in Sleep Staging', 'ICLR 2026', 'eess.SP'),
]

for paper_id, title, venue, pclass in cat2_papers:
    print(f"\n{paper_id}: {title[:60]}...")
    results = ss_search(title)
    time.sleep(1)

    best = None
    best_sim = 0
    for r in results:
        sim = jaccard(title, r.get('title', ''))
        if sim > best_sim:
            best_sim = sim
            best = r

    if best and best_sim > 0.5:
        print(f"  Best: '{best.get('title','')[:60]}' (sim={best_sim:.3f})")
        ext_ids = best.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = best.get('authors', [])
        year = str(best.get('year', '2025'))
        ret_title = best.get('title', title)
        key = make_key(authors, year, ret_title)

        if doi:
            print(f"  Found DOI: {doi}, fetching BibTeX...")
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI BibTeX: key to be parsed")
                resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title, 'doi': doi}
            else:
                # Construct from SS data
                print(f"  -> DOI fetch failed, using SS data")
                unresolved_details[paper_id] = f"SS found (doi={doi}) but BibTeX fetch failed"
        elif arxiv_id:
            print(f"  Found arXiv: {arxiv_id}")
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, pclass)
            resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title, 'arxiv': arxiv_id}
            print(f"  -> RESOLVED via arXiv: key={key}")
        else:
            print(f"  -> Found on SS but no DOI/arXiv ID available")
            unresolved_details[paper_id] = f"SS found (no DOI/arXiv), venue={best.get('venue','?')}, year={year}"
    else:
        print(f"  -> NOT FOUND (best sim={best_sim:.3f})")
        unresolved_details[paper_id] = "Not found on Semantic Scholar"

    time.sleep(1)

# Also try CrossRef for P50
print("\nP50: Trying CrossRef for PSDNorm sleep staging...")
cr_items = crossref_search("PSDNorm sleep staging temporal normalization")
time.sleep(1)
for item in cr_items:
    cr_title = ' '.join(item.get('title', ['']))
    sim = jaccard('PSDNorm Temporal Normalization for Deep Learning in Sleep Staging', cr_title)
    print(f"  CrossRef: '{cr_title[:60]}' sim={sim:.3f}")
    if sim > 0.5:
        doi = item.get('DOI')
        if doi:
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via CrossRef: DOI={doi}")
                if 'P50' not in resolved:
                    resolved['P50'] = {'key': 'psdnorm', 'bib': bib, 'title': cr_title, 'doi': doi}
                break

print("\n" + "=" * 60)
print("CATEGORY 3: Priority papers with known DOIs")
print("=" * 60)

# P4 - CorSW KDD 2026
print("\nP4: CorSW KDD 2026, DOI: 10.1145/3770855.3818864")
bib = fetch_doi_bibtex("10.1145/3770855.3818864")
time.sleep(1)
if bib and len(bib) > 50:
    print(f"  -> RESOLVED via DOI BibTeX")
    resolved['P4'] = {'key': 'corsw2026', 'bib': bib, 'title': 'CorSW'}
else:
    print(f"  -> DOI fetch failed, trying SS...")
    r = ss_by_doi("10.1145/3770855.3818864")
    time.sleep(1)
    if r and r.get('title'):
        print(f"  -> SS found: '{r['title']}'")
        authors = r.get('authors', [])
        year = str(r.get('year', '2026'))
        title_r = r.get('title', '')
        key = make_key(authors, year, title_r)
        unresolved_details['P4'] = f"SS found title='{title_r}' but no BibTeX DOI negotiation failed"
        # Try arxiv
        arxiv_id = r.get('externalIds', {}).get('ArXiv')
        if arxiv_id:
            bib = make_misc_bibtex(key, title_r, authors, year, arxiv_id, 'cs.LG')
            resolved['P4'] = {'key': key, 'bib': bib, 'title': title_r}
            print(f"  -> RESOLVED via arXiv on SS: key={key}")
    else:
        print(f"  -> NOT FOUND")
        unresolved_details['P4'] = "DOI 10.1145/3770855.3818864 — content negotiation failed, SS lookup failed"

# P48 - ADHD harmonization
print("\nP48: Riemannian Geometry Functional Connectivity ADHD Harmonization")
results = ss_search("Riemannian Geometry Functional Connectivity Matrices Multi-Site ADHD Data Harmonization")
time.sleep(1)
for r in results:
    sim = jaccard("Riemannian Geometry of Functional Connectivity Matrices for Multi-Site ADHD Data Harmonization", r.get('title', ''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.5:
        ext_ids = r.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = r.get('authors', [])
        year = str(r.get('year', '2022'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if doi:
            bib = fetch_doi_bibtex(doi)
            time.sleep(1)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI: {doi}")
                resolved['P48'] = {'key': key, 'bib': bib, 'title': ret_title}
                break
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
            print(f"  -> RESOLVED via arXiv: {arxiv_id}")
            resolved['P48'] = {'key': key, 'bib': bib, 'title': ret_title}
            break
if 'P48' not in resolved:
    # Also try CrossRef
    cr_items = crossref_search("Riemannian Geometry Functional Connectivity ADHD Harmonization Frontiers Neuroinformatics")
    time.sleep(1)
    for item in cr_items:
        cr_title = ' '.join(item.get('title', ['']))
        sim = jaccard("Riemannian Geometry of Functional Connectivity Matrices for Multi-Site ADHD Data Harmonization", cr_title)
        print(f"  CrossRef: '{cr_title[:60]}' sim={sim:.3f}")
        if sim > 0.45:
            doi = item.get('DOI')
            if doi:
                bib = fetch_doi_bibtex(doi)
                time.sleep(1)
                if bib and len(bib) > 50:
                    print(f"  -> RESOLVED via CrossRef: DOI={doi}")
                    resolved['P48'] = {'key': 'adhd2022riemannian', 'bib': bib, 'title': cr_title}
                    break

# P10 - CTSSP Journal of Neural Engineering
print("\nP10: CTSSP temporal spectral spatial motor imagery")
results = ss_search("CTSSP temporal spectral spatial joint optimization motor imagery")
time.sleep(1)
for r in results:
    sim = jaccard("CTSSP A Temporal-Spectral-Spatial Joint Optimization Algorithm for Motor Imagery", r.get('title', ''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.4:
        ext_ids = r.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = r.get('authors', [])
        year = str(r.get('year', '2026'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if doi:
            bib = fetch_doi_bibtex(doi)
            time.sleep(1)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI: {doi}")
                resolved['P10'] = {'key': key, 'bib': bib, 'title': ret_title}
                break
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
            print(f"  -> RESOLVED via arXiv: {arxiv_id}")
            resolved['P10'] = {'key': key, 'bib': bib, 'title': ret_title}
            break

if 'P10' not in resolved:
    cr_items = crossref_search("CTSSP temporal spectral spatial motor imagery Journal Neural Engineering")
    time.sleep(1)
    for item in cr_items:
        cr_title = ' '.join(item.get('title', ['']))
        sim = jaccard("CTSSP Temporal-Spectral-Spatial Joint Optimization Algorithm Motor Imagery", cr_title)
        print(f"  CrossRef: '{cr_title[:60]}' sim={sim:.3f}")
        if sim > 0.4:
            doi = item.get('DOI')
            if doi:
                bib = fetch_doi_bibtex(doi)
                time.sleep(1)
                if bib and len(bib) > 50:
                    print(f"  -> RESOLVED via CrossRef: DOI={doi}")
                    resolved['P10'] = {'key': 'ctssp2026temporal', 'bib': bib, 'title': cr_title}
                    break

# P43 - Chinese biomedical journal
print("\nP43: Chinese biomedical journal, DOI: 10.7507/1001-5515.202507056")
r = ss_by_doi("10.7507/1001-5515.202507056")
time.sleep(1)
if r and r.get('title'):
    title_r = r.get('title', '')
    authors = r.get('authors', [])
    year = str(r.get('year', '2026'))
    print(f"  SS found: '{title_r}'")
    key = make_key(authors, year, title_r)
    # construct article bib from SS data
    venue = r.get('venue', 'Journal of Biomedical Engineering')
    author_str = ' and '.join(a.get('name', '') for a in authors)
    bib = f"""@article{{{key},
  title   = {{{title_r}}},
  author  = {{{author_str}}},
  year    = {{{year}}},
  journal = {{{venue}}},
  doi     = {{10.7507/1001-5515.202507056}},
  url     = {{https://doi.org/10.7507/1001-5515.202507056}},
  note    = {{Chinese biomedical journal}}
}}"""
    resolved['P43'] = {'key': key, 'bib': bib, 'title': title_r}
    print(f"  -> RESOLVED from SS data: key={key}")
else:
    print(f"  -> NOT FOUND on SS")
    unresolved_details['P43'] = "DOI 10.7507/1001-5515.202507056 — Chinese journal, not in SS"

# P19 - Filter Bank CSP Brain Informatics
print("\nP19: Filter Bank CSP Riemannian Weighting Disability Motor Imagery Brain Informatics")
results = ss_search("Filter Bank CSP Riemannian Weighting Disability Motor Imagery BCI")
time.sleep(1)
for r in results:
    sim = jaccard("Filter Bank CSP with Riemannian Weighting for Disability-Centric Motor Imagery BCI", r.get('title', ''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.5:
        ext_ids = r.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = r.get('authors', [])
        year = str(r.get('year', '2026'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if doi:
            bib = fetch_doi_bibtex(doi)
            time.sleep(1)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI: {doi}")
                resolved['P19'] = {'key': key, 'bib': bib, 'title': ret_title}
                break
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
            print(f"  -> RESOLVED via arXiv: {arxiv_id}")
            resolved['P19'] = {'key': key, 'bib': bib, 'title': ret_title}
            break

if 'P19' not in resolved:
    cr_items = crossref_search("Filter Bank CSP Riemannian Weighting disability motor imagery Brain Informatics")
    time.sleep(1)
    for item in cr_items:
        cr_title = ' '.join(item.get('title', ['']))
        sim = jaccard("Filter Bank CSP with Riemannian Weighting for Disability-Centric Motor Imagery BCI", cr_title)
        print(f"  CrossRef: '{cr_title[:60]}' sim={sim:.3f}")
        if sim > 0.4:
            doi = item.get('DOI')
            if doi:
                bib = fetch_doi_bibtex(doi)
                time.sleep(1)
                if bib and len(bib) > 50:
                    print(f"  -> RESOLVED via CrossRef: DOI={doi}")
                    resolved['P19'] = {'key': 'filtercsp2026', 'bib': bib, 'title': cr_title}
                    break

print("\n" + "=" * 60)
print("CATEGORY 4: Unknown venue papers via Semantic Scholar")
print("=" * 60)

cat4_papers = [
    ('P76', 'Channel Adaptation for EEG Foundation Models: A Systematic Benchmark', 'eess.SP'),
    ('P77', 'EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces', 'eess.SP'),
    ('P78', 'NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis', 'eess.SP'),
    ('P70', 'Res2SPDNet: Multi-Granularity SPD Matrix Residual Learning for Signal Classification', 'cs.LG'),
    ('P97', 'Outlier Detection for Riemannian Manifold-Valued Functional Data', 'stat.ME'),
]

for paper_id, title, pclass in cat4_papers:
    print(f"\n{paper_id}: {title[:60]}...")
    results = ss_search(title)
    time.sleep(1)

    best = None
    best_sim = 0
    for r in results:
        sim = jaccard(title, r.get('title', ''))
        if sim > best_sim:
            best_sim = sim
            best = r

    if best and best_sim > 0.6:
        print(f"  Best: '{best.get('title','')[:60]}' sim={best_sim:.3f}")
        ext_ids = best.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = best.get('authors', [])
        year = str(best.get('year', '2026'))
        ret_title = best.get('title', title)
        key = make_key(authors, year, ret_title)

        if doi:
            bib = fetch_doi_bibtex(doi)
            time.sleep(1)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI: {doi}")
                resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
                continue
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, pclass)
            print(f"  -> RESOLVED via arXiv: {arxiv_id}")
            resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
        elif doi:
            unresolved_details[paper_id] = f"SS found (doi={doi}) but BibTeX fetch failed"
            print(f"  -> SS found but can't get BibTeX")
        else:
            unresolved_details[paper_id] = f"SS found (no DOI/arXiv), title='{ret_title}'"
            print(f"  -> SS found but no DOI/arXiv")
    else:
        print(f"  -> NOT FOUND (best sim={best_sim:.3f})")
        unresolved_details[paper_id] = f"Not found on SS (best sim={best_sim:.3f})"

print("\n" + "=" * 60)
print(f"TOTAL RESOLVED in this script: {len(resolved)}")
for pid, v in resolved.items():
    print(f"  {pid}: {v['key']}")
print(f"\nUNRESOLVED: {list(unresolved_details.keys())}")
for pid, reason in unresolved_details.items():
    print(f"  {pid}: {reason}")

# Save
output = {
    'resolved': resolved,
    'unresolved': unresolved_details
}
with open('/tmp/semantic_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved to /tmp/semantic_results.json")
