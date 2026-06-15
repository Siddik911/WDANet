#!/usr/bin/env python3
"""Retry rate-limited SS requests with longer delays, and do HAL searches."""
import urllib.request
import urllib.parse
import json
import time
import re

def jaccard(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def make_key(authors, year, title):
    if not authors:
        return f"unknown{year}key"
    first_author = authors[0] if isinstance(authors[0], str) else authors[0].get('name', 'unknown')
    parts = first_author.strip().split()
    if ',' in first_author:
        surname = first_author.split(',')[0].strip()
    else:
        surname = parts[-1] if parts else 'unknown'
    surname = re.sub(r'[^a-z]', '', surname.lower())
    skip = {'a', 'an', 'the', 'on', 'of', 'for', 'in', 'to', 'and', 'with', 'via', 'by', 'from', 'using'}
    words = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().split()
    firstword = 'paper'
    for w in words:
        if w not in skip and len(w) > 1:
            firstword = re.sub(r'[^a-z0-9]', '', w)
            break
    return f"{surname}{year}{firstword}"

def ss_search(title, max_retries=3, sleep_sec=5):
    encoded = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&fields=title,externalIds,year,venue,authors,publicationVenue&limit=3"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            return data.get('data', [])
        except Exception as e:
            print(f"  SS attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(sleep_sec)
    return []

def fetch_doi_bibtex(doi):
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

def make_misc_bibtex(key, title, authors, year, arxiv_id, primary_class='cs.LG'):
    if isinstance(authors, list) and authors and isinstance(authors[0], dict):
        author_str = ' and '.join(a.get('name', '') for a in authors)
    elif isinstance(authors, list):
        author_str = ' and '.join(str(a) for a in authors)
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
print("RETRY: Rate-limited papers")
print("=" * 60)

# Papers that got 429 errors, retry with delays
retry_papers = [
    ('P49', 'Riemannian Flow Matching for Brain Connectivity Matrices via Pullback Geometry', 'cs.LG'),
    ('P73', 'Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds', 'cs.LG'),
    ('P96', 'NEED: Cross-Subject and Cross-Task Generalization for Video and Image Reconstruction from EEG Signals', 'eess.SP'),
    ('P3', 'Riemannian High-Order Pooling for Brain Foundation Models', 'eess.SP'),
    ('P46', 'HybridRDG: Zero-Shot ASD Biomarker Detection from Multi-Paradigm EEG via Hybrid Deep-Riemannian Domain Generalization', 'eess.SP'),
    ('P50', 'PSDNorm: Temporal Normalization for Deep Learning in Sleep Staging', 'eess.SP'),
    ('P76', 'Channel Adaptation for EEG Foundation Models: A Systematic Benchmark', 'eess.SP'),
    ('P77', 'EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces', 'eess.SP'),
    ('P78', 'NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis', 'eess.SP'),
    ('P70', 'Res2SPDNet: Multi-Granularity SPD Matrix Residual Learning for Signal Classification', 'cs.LG'),
    ('P97', 'Outlier Detection for Riemannian Manifold-Valued Functional Data', 'stat.ME'),
]

for paper_id, title, pclass in retry_papers:
    print(f"\n{paper_id}: {title[:60]}...")
    results = ss_search(title, max_retries=3, sleep_sec=8)
    time.sleep(3)

    best = None
    best_sim = 0
    for r in results:
        sim = jaccard(title, r.get('title', ''))
        if sim > best_sim:
            best_sim = sim
            best = r

    if best and best_sim > 0.5:
        print(f"  Best: '{best.get('title','')[:60]}' sim={best_sim:.3f}")
        ext_ids = best.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = best.get('authors', [])
        year = str(best.get('year', '2025'))
        ret_title = best.get('title', title)
        key = make_key(authors, year, ret_title)

        if doi and not doi.startswith('10.48550'):  # Skip arXiv DOIs
            bib = fetch_doi_bibtex(doi)
            time.sleep(2)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI: {doi}")
                resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
                continue

        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, pclass)
            print(f"  -> RESOLVED via arXiv: {arxiv_id}")
            resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
        elif doi:
            # It's on SS, has DOI but couldn't fetch BibTeX
            print(f"  -> SS found, has DOI={doi} but BibTeX unavailable")
            unresolved_details[paper_id] = f"SS found, doi={doi}, no arXiv"
        else:
            print(f"  -> SS found but no DOI/arXiv")
            unresolved_details[paper_id] = f"SS found (no DOI/arXiv): {ret_title[:60]}"
    else:
        print(f"  -> NOT FOUND on SS (best sim={best_sim:.3f})")
        unresolved_details[paper_id] = f"Not found on SS"

print("\n" + "=" * 60)
print("CATEGORY 5: HAL preprints")
print("=" * 60)

def hal_search(title_query, rows=3):
    encoded = urllib.parse.quote(title_query)
    url = f"https://api.archives-ouvertes.fr/search/?q=title_t:{encoded}&fl=halId_s,title_s,authFirstName_s,authLastName_s,producedDate_s,doiId_s&wt=json&rows={rows}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('response', {}).get('docs', [])
    except Exception as e:
        print(f"  HAL ERROR for '{title_query[:50]}': {e}")
        return []

hal_papers = [
    ('P11', 'Pseudo Affine-Invariant Riemannian Metrics for Efficient Brain-Computer Interfaces', 'eess.SP'),
    ('P34', 'FB-RCSP-RA: A Filter-Bank Regularized CSP Framework with Per-Band Riemannian Alignment for Cross-Subject Motor Imagery EEG Decoding', 'eess.SP'),
    ('P66', 'ARMAGNAC: A New Parametric Batch Normalization Layer for SPDNet Architecture', 'cs.LG'),
]

for paper_id, title, pclass in hal_papers:
    print(f"\n{paper_id}: {title[:60]}...")
    docs = hal_search(title)
    time.sleep(1)

    best = None
    best_sim = 0
    for doc in docs:
        hal_titles = doc.get('title_s', [])
        hal_title = hal_titles[0] if hal_titles else ''
        sim = jaccard(title, hal_title)
        if sim > best_sim:
            best_sim = sim
            best = doc
            best['_matched_title'] = hal_title

    if best and best_sim > 0.4:
        hal_id = best.get('halId_s', '')
        hal_title = best.get('_matched_title', title)
        doi = best.get('doiId_s', '')
        first_names = best.get('authFirstName_s', [])
        last_names = best.get('authLastName_s', [])
        produced_date = best.get('producedDate_s', '')
        year = produced_date[:4] if produced_date else '2026'

        # Build author string
        authors_list = []
        for fn, ln in zip(first_names, last_names):
            authors_list.append(f"{fn} {ln}")
        author_str = ' and '.join(authors_list) if authors_list else 'Unknown Author'

        print(f"  HAL found: '{hal_title[:60]}' (sim={best_sim:.3f})")
        print(f"  halId={hal_id}, doi={doi}, year={year}")

        key = make_key(
            [authors_list[0]] if authors_list else ['Unknown'],
            year, hal_title
        )

        if doi:
            print(f"  Found DOI: {doi}, fetching BibTeX...")
            bib = fetch_doi_bibtex(doi)
            time.sleep(1)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI")
                resolved[paper_id] = {'key': key, 'bib': bib, 'title': hal_title}
                continue

        # Generate HAL misc entry
        bib = f"""@misc{{{key},
  title  = {{{hal_title}}},
  author = {{{author_str}}},
  year   = {{{year}}},
  url    = {{https://hal.science/{hal_id}}},
  note   = {{HAL preprint {hal_id}}}
}}"""
        print(f"  -> RESOLVED via HAL: key={key}, halId={hal_id}")
        resolved[paper_id] = {'key': key, 'bib': bib, 'title': hal_title}
    else:
        print(f"  -> NOT FOUND on HAL (best sim={best_sim:.3f})")
        unresolved_details[paper_id] = f"Not found on HAL"

print("\n" + "=" * 60)
print(f"ADDITIONAL RESOLVED: {len(resolved)}")
for pid, v in resolved.items():
    print(f"  {pid}: {v['key']}")
print(f"\nUNRESOLVED: {list(unresolved_details.keys())}")
for pid, reason in unresolved_details.items():
    print(f"  {pid}: {reason}")

output = {
    'resolved': resolved,
    'unresolved': unresolved_details
}
with open('/tmp/retry_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to /tmp/retry_results.json")
