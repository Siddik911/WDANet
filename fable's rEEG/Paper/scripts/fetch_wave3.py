#!/usr/bin/env python3
"""Wave 3: Very long SS delays + arXiv fallbacks for NeurIPS/ICLR papers + P96 details."""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
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
    if ',' in first_author:
        surname = first_author.split(',')[0].strip()
    else:
        parts = first_author.strip().split()
        surname = parts[-1] if parts else 'unknown'
    surname = re.sub(r'[^a-z]', '', surname.lower())
    skip = {'a', 'an', 'the', 'on', 'of', 'for', 'in', 'to', 'and', 'with', 'via', 'by', 'from', 'using', 'its', 'towards', 'general'}
    words = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().split()
    firstword = 'paper'
    for w in words:
        if w not in skip and len(w) > 2:
            firstword = re.sub(r'[^a-z0-9]', '', w)
            break
    return f"{surname}{year}{firstword}"

def ss_search_slow(title, sleep_before=20):
    time.sleep(sleep_before)
    encoded = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&fields=title,externalIds,year,venue,authors,publicationVenue&limit=3"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = data.get('data', [])
        print(f"  SS OK: {len(results)} results")
        return results
    except Exception as e:
        print(f"  SS ERROR: {e}")
        return []

def search_arxiv(title, sleep_before=1.0):
    time.sleep(sleep_before)
    encoded = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=ti:{encoded}&max_results=3&sortBy=relevance"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_data = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  arXiv ERROR: {e}")
        return []
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    try:
        root = ET.fromstring(xml_data)
    except:
        return []
    results = []
    for entry in root.findall('atom:entry', ns):
        arxiv_id_full = entry.find('atom:id', ns)
        ret_title = entry.find('atom:title', ns)
        published = entry.find('atom:published', ns)
        authors_el = entry.findall('atom:author', ns)
        if arxiv_id_full is None or ret_title is None:
            continue
        arxiv_url = arxiv_id_full.text.strip()
        arxiv_id = re.sub(r'v\d+$', '', arxiv_url.split('abs/')[-1])
        title_clean = ' '.join(ret_title.text.strip().split())
        year = published.text[:4] if published is not None else '2026'
        author_names = [a.find('atom:name', ns).text.strip()
                        for a in authors_el if a.find('atom:name', ns) is not None]
        sim = jaccard(title, title_clean)
        results.append({'arxiv_id': arxiv_id, 'title': title_clean, 'year': year,
                        'authors': author_names, 'similarity': sim})
    return results

def fetch_doi_bibtex(doi, sleep_before=2):
    time.sleep(sleep_before)
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
print("WAVE 3: Try arXiv for NeurIPS/ICLR papers first (faster)")
print("=" * 60)

# These are conference papers that likely have arXiv preprints
arxiv_for_conf = [
    ('P73', 'Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds', 'cs.LG', 0.55),
    ('P3', 'Riemannian High-Order Pooling Brain Foundation Models', 'eess.SP', 0.5),
    ('P46', 'HybridRDG Zero-Shot ASD EEG Hybrid Deep Riemannian Domain Generalization', 'eess.SP', 0.40),
    ('P50', 'PSDNorm temporal normalization sleep staging EEG', 'eess.SP', 0.45),
    ('P96', 'NEED Cross-Subject Cross-Task EEG Video Image Reconstruction', 'eess.SP', 0.5),
]

for paper_id, title, pclass, threshold in arxiv_for_conf:
    print(f"\n{paper_id}: {title[:60]}...")
    results = search_arxiv(title, sleep_before=0.8)
    if not results:
        print(f"  -> No arXiv results")
        unresolved_details[paper_id] = "Not on arXiv"
        continue
    best = max(results, key=lambda r: r['similarity'])
    print(f"  Best: '{best['title'][:60]}' sim={best['similarity']:.3f}")
    if best['similarity'] > threshold:
        authors = best['authors']
        year = best['year']
        title_found = best['title']
        arxiv_id = best['arxiv_id']
        key = make_key(authors, year, title_found)
        bib = make_misc_bibtex(key, title_found, authors, year, arxiv_id, pclass)
        print(f"  -> RESOLVED via arXiv: key={key}")
        resolved[paper_id] = {'key': key, 'bib': bib, 'title': title_found}
    else:
        print(f"  -> REJECTED from arXiv")
        unresolved_details[paper_id] = f"arXiv best sim={best['similarity']:.3f}"

print("\n" + "=" * 60)
print("WAVE 3: SS with very long delays for Cat4 papers")
print("=" * 60)

cat4_slow = [
    ('P76', 'Channel Adaptation for EEG Foundation Models: A Systematic Benchmark', 'eess.SP'),
    ('P77', 'EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces', 'eess.SP'),
    ('P78', 'NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis', 'eess.SP'),
    ('P70', 'Res2SPDNet: Multi-Granularity SPD Matrix Residual Learning for Signal Classification', 'cs.LG'),
    ('P97', 'Outlier Detection for Riemannian Manifold-Valued Functional Data', 'stat.ME'),
]

for paper_id, title, pclass in cat4_slow:
    print(f"\n{paper_id}: {title[:60]}...")
    # First try arXiv
    results = search_arxiv(title, sleep_before=0.5)
    if results:
        best = max(results, key=lambda r: r['similarity'])
        print(f"  arXiv best: '{best['title'][:60]}' sim={best['similarity']:.3f}")
        if best['similarity'] > 0.55:
            authors = best['authors']
            year = best['year']
            title_found = best['title']
            arxiv_id = best['arxiv_id']
            key = make_key(authors, year, title_found)
            bib = make_misc_bibtex(key, title_found, authors, year, arxiv_id, pclass)
            print(f"  -> RESOLVED via arXiv: key={key}")
            resolved[paper_id] = {'key': key, 'bib': bib, 'title': title_found}
            continue

    # Then SS
    results_ss = ss_search_slow(title, sleep_before=25)
    best = None
    best_sim = 0
    for r in results_ss:
        sim = jaccard(title, r.get('title', ''))
        if sim > best_sim:
            best_sim = sim
            best = r

    if best and best_sim > 0.55:
        print(f"  SS best: '{best.get('title','')[:60]}' sim={best_sim:.3f}")
        ext_ids = best.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = best.get('authors', [])
        year = str(best.get('year', '2026'))
        ret_title = best.get('title', title)
        key = make_key(authors, year, ret_title)

        if doi and not doi.startswith('10.48550'):
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI")
                resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
                continue
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, pclass)
            print(f"  -> RESOLVED via SS arXiv: {arxiv_id}")
            resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
        else:
            print(f"  -> SS found but no citable ID")
            unresolved_details[paper_id] = f"SS found '{ret_title[:50]}' but no DOI/arXiv"
    else:
        print(f"  -> NOT FOUND (SS sim={best_sim:.3f})")
        unresolved_details[paper_id] = f"Not found (arXiv+SS)"

print("\n" + "=" * 60)
print("WAVE 3: SS slow for remaining unresolved arXiv Cat1 papers")
print("=" * 60)

cat1_ss_fallback = [
    ('P54', 'Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods', 'eess.SP'),
    ('P67', 'FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet', 'cs.LG'),
    ('P75', 'SPDLearn: A Geometric Deep Learning Python Library for Neural Decoding Through Trivialization', 'cs.LG'),
    ('P85', 'Riemannian Adversarial Attacks on Symmetric Positive Definite Matrices', 'cs.LG'),
    ('P91', 'Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks', 'cs.LG'),
    ('P94', 'Batch Normalization for Neural Networks on Complex Domains', 'cs.LG'),
]

for paper_id, title, pclass in cat1_ss_fallback:
    print(f"\n{paper_id}: {title[:60]}...")
    results_ss = ss_search_slow(title, sleep_before=22)
    best = None
    best_sim = 0
    for r in results_ss:
        sim = jaccard(title, r.get('title', ''))
        if sim > best_sim:
            best_sim = sim
            best = r

    if best and best_sim > 0.5:
        print(f"  SS best: '{best.get('title','')[:60]}' sim={best_sim:.3f}")
        ext_ids = best.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = best.get('authors', [])
        year = str(best.get('year', '2026'))
        ret_title = best.get('title', title)
        key = make_key(authors, year, ret_title)

        if doi and not doi.startswith('10.48550'):
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI")
                resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
                continue
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, pclass)
            print(f"  -> RESOLVED via SS arXiv: {arxiv_id}")
            resolved[paper_id] = {'key': key, 'bib': bib, 'title': ret_title}
        else:
            print(f"  -> SS found but no IDs")
            unresolved_details[paper_id] = f"SS found '{ret_title[:50]}' but no DOI/arXiv"
    else:
        print(f"  -> NOT FOUND (SS sim={best_sim:.3f})")
        unresolved_details[paper_id] = f"Not found on SS"

# Also try P73 via SS
print("\nP73 final SS attempt...")
results_ss = ss_search_slow('Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds', sleep_before=25)
for r in results_ss:
    sim = jaccard('Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds', r.get('title', ''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.5:
        ext_ids = r.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = r.get('authors', [])
        year = str(r.get('year', '2025'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'cs.LG')
            print(f"  -> RESOLVED P73 via SS arXiv: {arxiv_id}")
            resolved['P73'] = {'key': key, 'bib': bib, 'title': ret_title}
            break
        elif doi and not doi.startswith('10.48550'):
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                resolved['P73'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P73 via DOI")
                break

if 'P73' not in resolved:
    unresolved_details['P73'] = "Not found on SS or arXiv"

print("\n" + "=" * 60)
print(f"WAVE 3 RESOLVED: {len(resolved)}")
for pid, v in resolved.items():
    print(f"  {pid}: {v['key']}")
print(f"\nSTILL UNRESOLVED: {list(unresolved_details.keys())}")
for pid, reason in unresolved_details.items():
    print(f"  {pid}: {reason}")

output = {'resolved': resolved, 'unresolved': unresolved_details}
with open('/tmp/wave3_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to /tmp/wave3_results.json")
