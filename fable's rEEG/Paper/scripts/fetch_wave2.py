#!/usr/bin/env python3
"""Wave 2: Long-delay SS retries + alternate arXiv searches + HAL alternate queries."""
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
    skip = {'a', 'an', 'the', 'on', 'of', 'for', 'in', 'to', 'and', 'with', 'via', 'by', 'from', 'using', 'its'}
    words = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().split()
    firstword = 'paper'
    for w in words:
        if w not in skip and len(w) > 1:
            firstword = re.sub(r'[^a-z0-9]', '', w)
            break
    return f"{surname}{year}{firstword}"

def ss_search(title, sleep_before=10):
    time.sleep(sleep_before)
    encoded = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&fields=title,externalIds,year,venue,authors,publicationVenue&limit=3"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('data', [])
    except Exception as e:
        print(f"  SS ERROR: {e}")
        return []

def search_arxiv(title, sleep_before=0.5):
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
    except Exception as e:
        print(f"  XML parse ERROR: {e}")
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
        author_names = []
        for a in authors_el:
            name_el = a.find('atom:name', ns)
            if name_el is not None:
                author_names.append(name_el.text.strip())
        sim = jaccard(title, title_clean)
        results.append({'arxiv_id': arxiv_id, 'title': title_clean, 'year': year,
                        'authors': author_names, 'similarity': sim})
    return results

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

resolved = {}
unresolved_details = {}

print("=" * 60)
print("WAVE 2: SS retries with long delays")
print("=" * 60)

# These all need SS retries - use 12s sleep between each
wave2_ss = [
    ('P73', 'Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds', 'cs.LG'),
    ('P3', 'Riemannian High-Order Pooling for Brain Foundation Models', 'eess.SP'),
    ('P46', 'HybridRDG: Zero-Shot ASD Biomarker Detection from Multi-Paradigm EEG', 'eess.SP'),
    ('P50', 'PSDNorm Temporal Normalization Deep Learning Sleep Staging', 'eess.SP'),
    ('P76', 'Channel Adaptation EEG Foundation Models Benchmark', 'eess.SP'),
    ('P77', 'EEG-Deformer Dense Convolutional Transformer Brain-Computer Interface', 'eess.SP'),
    ('P78', 'NeuroPhysNet FitzHugh-Nagumo Physics-Informed Neural Network EEG', 'eess.SP'),
    ('P70', 'Res2SPDNet Multi-Granularity SPD Matrix Residual Learning Signal Classification', 'cs.LG'),
    ('P97', 'Outlier Detection Riemannian Manifold Functional Data', 'stat.ME'),
]

for paper_id, title, pclass in wave2_ss:
    print(f"\n{paper_id}: {title[:60]}...")
    results = ss_search(title, sleep_before=12)

    best = None
    best_sim = 0
    for r in results:
        sim = jaccard(title, r.get('title', ''))
        if sim > best_sim:
            best_sim = sim
            best = r

    if best and best_sim > 0.45:
        print(f"  Best: '{best.get('title','')[:60]}' sim={best_sim:.3f}")
        ext_ids = best.get('externalIds', {})
        doi = ext_ids.get('DOI')
        arxiv_id = ext_ids.get('ArXiv')
        authors = best.get('authors', [])
        year = str(best.get('year', '2025'))
        ret_title = best.get('title', title)
        key = make_key(authors, year, ret_title)

        if doi and not doi.startswith('10.48550'):
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
            print(f"  -> SS found with DOI={doi} but can't get BibTeX")
            unresolved_details[paper_id] = f"SS: doi={doi}, no arXiv, BibTeX failed"
        else:
            print(f"  -> SS found but no DOI/arXiv IDs")
            unresolved_details[paper_id] = f"SS found '{ret_title[:50]}' but no citable ID"
    else:
        print(f"  -> NOT FOUND on SS (best sim={best_sim:.3f})")
        unresolved_details[paper_id] = "Not found on SS"

print("\n" + "=" * 60)
print("WAVE 2: arXiv alternate searches for unresolved Cat1 papers")
print("=" * 60)

# Unresolved from Cat1: P54, P67, P75, P85, P91, P93, P94
# Try shorter/alternate queries
arxiv_retries = [
    ('P54', 'Cross-Subject Generalization EEG Decoding Survey Deep Learning', 'eess.SP', 0.45),
    ('P67', 'FedSPDnet Federated Learning SPDnet geometry', 'cs.LG', 0.45),
    ('P75', 'SPDLearn Geometric Deep Learning library neural decoding trivialization', 'cs.LG', 0.40),
    ('P85', 'Riemannian adversarial attacks SPD matrices', 'cs.LG', 0.45),
    ('P91', 'Riemannian diffusion models physics-informed neural networks', 'cs.LG', 0.40),
    ('P93', 'intrinsic local polynomial regression Riemannian manifolds', 'math.ST', 0.40),
    ('P94', 'batch normalization neural networks complex domains', 'cs.LG', 0.40),
]

for paper_id, query, pclass, threshold in arxiv_retries:
    print(f"\n{paper_id}: {query[:60]}...")
    results = search_arxiv(query, sleep_before=0.5)

    if not results:
        print(f"  -> No results")
        unresolved_details[paper_id] = "Not found on arXiv (alternate query)"
        continue

    best = max(results, key=lambda r: r['similarity'])
    print(f"  Best: '{best['title'][:60]}' (sim={best['similarity']:.3f})")

    if best['similarity'] > threshold:
        authors = best['authors']
        year = best['year']
        title_found = best['title']
        arxiv_id = best['arxiv_id']
        key = make_key(authors, year, title_found)
        bib = make_misc_bibtex(key, title_found, authors, year, arxiv_id, pclass)
        print(f"  -> RESOLVED: key={key}, arXiv={arxiv_id}")
        resolved[paper_id] = {'key': key, 'bib': bib, 'title': title_found}
    else:
        print(f"  -> REJECTED (sim too low)")
        unresolved_details[paper_id] = f"arXiv best sim={best['similarity']:.3f}"

print("\n" + "=" * 60)
print("WAVE 2: HAL alternate searches")
print("=" * 60)

def hal_search(query, rows=5):
    encoded = urllib.parse.quote(query)
    url = f"https://api.archives-ouvertes.fr/search/?q={encoded}&fl=halId_s,title_s,authFirstName_s,authLastName_s,producedDate_s,doiId_s&wt=json&rows={rows}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('response', {}).get('docs', [])
    except Exception as e:
        print(f"  HAL ERROR: {e}")
        return []

hal_retries = [
    ('P34', 'FB-RCSP-RA filter bank regularized CSP Riemannian alignment EEG', 'FB-RCSP-RA A Filter-Bank Regularized CSP Framework Per-Band Riemannian Alignment Cross-Subject Motor Imagery', 'eess.SP'),
    ('P66', 'ARMAGNAC batch normalization SPDNet', 'ARMAGNAC A New Parametric Batch Normalization Layer SPDNet', 'cs.LG'),
]

for paper_id, query, full_title, pclass in hal_retries:
    print(f"\n{paper_id}: {query[:60]}...")
    docs = hal_search(query)
    time.sleep(1)

    best = None
    best_sim = 0
    for doc in docs:
        hal_titles = doc.get('title_s', [])
        hal_title = hal_titles[0] if hal_titles else ''
        sim = jaccard(full_title, hal_title)
        if sim > best_sim:
            best_sim = sim
            best = doc
            best['_matched_title'] = hal_title

    if best and best_sim > 0.3:
        hal_id = best.get('halId_s', '')
        hal_title = best.get('_matched_title', full_title)
        doi = best.get('doiId_s', '')
        first_names = best.get('authFirstName_s', [])
        last_names = best.get('authLastName_s', [])
        produced_date = best.get('producedDate_s', '')
        year = produced_date[:4] if produced_date else '2026'

        authors_list = []
        for fn, ln in zip(first_names, last_names):
            authors_list.append(f"{fn} {ln}")
        author_str = ' and '.join(authors_list) if authors_list else 'Unknown Author'
        key = make_key([authors_list[0]] if authors_list else ['Unknown'], year, hal_title)

        print(f"  HAL found: '{hal_title[:60]}' sim={best_sim:.3f}, halId={hal_id}")

        if doi:
            bib = fetch_doi_bibtex(doi)
            time.sleep(1)
            if bib and len(bib) > 50:
                print(f"  -> RESOLVED via DOI")
                resolved[paper_id] = {'key': key, 'bib': bib, 'title': hal_title}
                continue

        bib = f"""@misc{{{key},
  title  = {{{hal_title}}},
  author = {{{author_str}}},
  year   = {{{year}}},
  url    = {{https://hal.science/{hal_id}}},
  note   = {{HAL preprint {hal_id}}}
}}"""
        print(f"  -> RESOLVED via HAL: key={key}")
        resolved[paper_id] = {'key': key, 'bib': bib, 'title': hal_title}
    else:
        print(f"  -> NOT FOUND on HAL (best sim={best_sim:.3f})")
        unresolved_details[paper_id] = "Not found on HAL"

    # Also try arXiv for these
    print(f"  Trying arXiv...")
    arxiv_results = search_arxiv(full_title.split()[:6].__class__.__mro__[0] and ' '.join(full_title.split()[:6]))
    # Simpler: just search with paper_id-specific terms
    if paper_id == 'P34':
        arxiv_results = search_arxiv('FB-RCSP-RA filter bank regularized CSP Riemannian alignment')
    elif paper_id == 'P66':
        arxiv_results = search_arxiv('ARMAGNAC batch normalization SPDNet parametric')

    if arxiv_results:
        best_ax = max(arxiv_results, key=lambda r: r['similarity'])
        print(f"  arXiv best: '{best_ax['title'][:60]}' sim={best_ax['similarity']:.3f}")
        if best_ax['similarity'] > 0.35 and paper_id not in resolved:
            authors = best_ax['authors']
            year = best_ax['year']
            title_found = best_ax['title']
            arxiv_id = best_ax['arxiv_id']
            key = make_key(authors, year, title_found)
            bib = make_misc_bibtex(key, title_found, authors, year, arxiv_id, pclass)
            print(f"  -> RESOLVED via arXiv: key={key}")
            resolved[paper_id] = {'key': key, 'bib': bib, 'title': title_found}

print("\n" + "=" * 60)
print(f"WAVE 2 RESOLVED: {len(resolved)}")
for pid, v in resolved.items():
    print(f"  {pid}: {v['key']}")
print(f"\nSTILL UNRESOLVED: {list(unresolved_details.keys())}")
for pid, reason in unresolved_details.items():
    print(f"  {pid}: {reason}")

output = {'resolved': resolved, 'unresolved': unresolved_details}
with open('/tmp/wave2_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to /tmp/wave2_results.json")
