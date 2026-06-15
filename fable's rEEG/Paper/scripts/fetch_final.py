#!/usr/bin/env python3
"""Final wave: targeted searches with crossref + very long SS waits + arXiv exact IDs."""
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
        if w not in skip and len(w) > 2:
            firstword = re.sub(r'[^a-z0-9]', '', w)
            break
    return f"{surname}{year}{firstword}"

def ss_search(title, sleep_before=30):
    time.sleep(sleep_before)
    encoded = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&fields=title,externalIds,year,venue,authors&limit=3"
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

def search_arxiv(title, sleep_before=0.8):
    time.sleep(sleep_before)
    encoded = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=ti:{encoded}&max_results=5&sortBy=relevance"
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
        print(f"  DOI BibTeX ERROR: {e}")
        return None

def crossref_search(query, rows=3, sleep_before=1):
    time.sleep(sleep_before)
    encoded = urllib.parse.quote(query)
    url = f"https://api.crossref.org/works?query.bibliographic={encoded}&rows={rows}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MyApp/1.0 (research@example.com)'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('message', {}).get('items', [])
    except Exception as e:
        print(f"  CrossRef ERROR: {e}")
        return []

resolved = {}
unresolved_details = {}

print("=" * 60)
print("FINAL WAVE: Targeted arXiv + SS + CrossRef")
print("=" * 60)

# For P54 - try keyword-only arXiv
print("\nP54: Cross-Subject EEG survey deep learning...")
results = search_arxiv('cross subject EEG decoding generalization survey deep learning 2026')
for r in results:
    sim = jaccard('Cross-Subject Generalization for EEG Decoding A Survey of Deep Learning Methods', r['title'])
    if sim > 0.40:
        print(f"  Match: '{r['title'][:60]}' sim={sim:.3f}")
        key = make_key(r['authors'], r['year'], r['title'])
        bib = make_misc_bibtex(key, r['title'], r['authors'], r['year'], r['arxiv_id'], 'eess.SP')
        resolved['P54'] = {'key': key, 'bib': bib, 'title': r['title']}
        break
if 'P54' not in resolved:
    # Try SS
    results = ss_search('Cross-Subject Generalization EEG Decoding Survey', sleep_before=30)
    for r in results:
        sim = jaccard('Cross-Subject Generalization for EEG Decoding A Survey of Deep Learning Methods', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.40:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            doi = ext_ids.get('DOI')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
                resolved['P54'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P54 via SS arXiv")
                break
            elif doi and not doi.startswith('10.48550'):
                bib = fetch_doi_bibtex(doi)
                if bib and len(bib) > 50:
                    resolved['P54'] = {'key': key, 'bib': bib, 'title': ret_title}
                    print(f"  -> RESOLVED P54 via DOI")
                    break
if 'P54' not in resolved:
    unresolved_details['P54'] = "Not found on arXiv or SS"

# P67 FedSPDnet
print("\nP67: FedSPDnet...")
results = search_arxiv('FedSPDnet federated learning geometry SPD manifold')
for r in results:
    sim = jaccard('FedSPDnet Geometry-Aware Federated Deep Learning with SPDnet', r['title'])
    if sim > 0.35:
        print(f"  arXiv: '{r['title'][:60]}' sim={sim:.3f}")
        key = make_key(r['authors'], r['year'], r['title'])
        bib = make_misc_bibtex(key, r['title'], r['authors'], r['year'], r['arxiv_id'], 'cs.LG')
        resolved['P67'] = {'key': key, 'bib': bib, 'title': r['title']}
        break

if 'P67' not in resolved:
    results = ss_search('FedSPDnet Geometry-Aware Federated Learning SPDnet', sleep_before=30)
    for r in results:
        sim = jaccard('FedSPDnet Geometry-Aware Federated Deep Learning with SPDnet', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.40:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'cs.LG')
                resolved['P67'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P67 via SS arXiv: {arxiv_id}")
                break
if 'P67' not in resolved:
    unresolved_details['P67'] = "Not found"

# P85 Riemannian adversarial attacks SPD
print("\nP85: Riemannian adversarial attacks SPD...")
results = search_arxiv('Riemannian adversarial attacks symmetric positive definite')
for r in results:
    sim = jaccard('Riemannian Adversarial Attacks on Symmetric Positive Definite Matrices', r['title'])
    print(f"  arXiv: '{r['title'][:60]}' sim={sim:.3f}")
    if sim > 0.45:
        key = make_key(r['authors'], r['year'], r['title'])
        bib = make_misc_bibtex(key, r['title'], r['authors'], r['year'], r['arxiv_id'], 'cs.LG')
        resolved['P85'] = {'key': key, 'bib': bib, 'title': r['title']}
        break
if 'P85' not in resolved:
    results = ss_search('Riemannian Adversarial Attacks Symmetric Positive Definite Matrices', sleep_before=30)
    for r in results:
        sim = jaccard('Riemannian Adversarial Attacks on Symmetric Positive Definite Matrices', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.40:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'cs.LG')
                resolved['P85'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P85 via SS arXiv: {arxiv_id}")
                break
if 'P85' not in resolved:
    unresolved_details['P85'] = "Not found"

# P91 Riemannian diffusion physics-informed
print("\nP91: Riemannian diffusion models physics-informed...")
results = search_arxiv('Riemannian diffusion models manifolds physics informed neural')
for r in results:
    sim = jaccard('Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks', r['title'])
    print(f"  arXiv: '{r['title'][:60]}' sim={sim:.3f}")
    if sim > 0.40:
        key = make_key(r['authors'], r['year'], r['title'])
        bib = make_misc_bibtex(key, r['title'], r['authors'], r['year'], r['arxiv_id'], 'cs.LG')
        resolved['P91'] = {'key': key, 'bib': bib, 'title': r['title']}
        break
if 'P91' not in resolved:
    results = ss_search('Riemannian Diffusion Models General Manifolds Physics-Informed Neural Networks', sleep_before=30)
    for r in results:
        sim = jaccard('Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.40:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'cs.LG')
                resolved['P91'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P91 via SS arXiv: {arxiv_id}")
                break
if 'P91' not in resolved:
    unresolved_details['P91'] = "Not found"

# P94 Batch normalization complex domains
print("\nP94: Batch normalization complex domains...")
results = search_arxiv('batch normalization complex domains neural networks')
for r in results:
    sim = jaccard('Batch Normalization for Neural Networks on Complex Domains', r['title'])
    print(f"  arXiv: '{r['title'][:60]}' sim={sim:.3f}")
    if sim > 0.45:
        key = make_key(r['authors'], r['year'], r['title'])
        bib = make_misc_bibtex(key, r['title'], r['authors'], r['year'], r['arxiv_id'], 'cs.LG')
        resolved['P94'] = {'key': key, 'bib': bib, 'title': r['title']}
        break
if 'P94' not in resolved:
    results = ss_search('Batch Normalization Neural Networks Complex Domains', sleep_before=30)
    for r in results:
        sim = jaccard('Batch Normalization for Neural Networks on Complex Domains', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.45:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            doi = ext_ids.get('DOI')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'cs.LG')
                resolved['P94'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P94 via SS arXiv: {arxiv_id}")
                break
            elif doi and not doi.startswith('10.48550'):
                bib = fetch_doi_bibtex(doi)
                if bib and len(bib) > 50:
                    resolved['P94'] = {'key': key, 'bib': bib, 'title': ret_title}
                    print(f"  -> RESOLVED P94 via DOI")
                    break
if 'P94' not in resolved:
    unresolved_details['P94'] = "Not found"

# Cat4: P70, P76, P77, P78, P97 - try CrossRef
print("\nP70: Res2SPDNet via CrossRef...")
items = crossref_search("Res2SPDNet Multi-Granularity SPD Matrix Residual Learning Signal Classification")
for item in items:
    cr_title = ' '.join(item.get('title', ['']))
    sim = jaccard('Res2SPDNet Multi-Granularity SPD Matrix Residual Learning for Signal Classification', cr_title)
    print(f"  CrossRef: '{cr_title[:60]}' sim={sim:.3f}")
    if sim > 0.50:
        doi = item.get('DOI')
        if doi:
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                resolved['P70'] = {'key': 'res2spdnet2026', 'bib': bib, 'title': cr_title}
                print(f"  -> RESOLVED P70 via CrossRef")
                break

if 'P70' not in resolved:
    # SS
    results = ss_search('Res2SPDNet Multi-Granularity SPD Matrix Residual Learning Signal Classification', sleep_before=30)
    for r in results:
        sim = jaccard('Res2SPDNet Multi-Granularity SPD Matrix Residual Learning for Signal Classification', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.50:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            doi = ext_ids.get('DOI')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'cs.LG')
                resolved['P70'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P70 via SS arXiv: {arxiv_id}")
                break
            elif doi and not doi.startswith('10.48550'):
                bib = fetch_doi_bibtex(doi)
                if bib and len(bib) > 50:
                    resolved['P70'] = {'key': key, 'bib': bib, 'title': ret_title}
                    print(f"  -> RESOLVED P70 via DOI")
                    break
if 'P70' not in resolved:
    unresolved_details['P70'] = "Not found"

print("\nP76: Channel Adaptation EEG Foundation Models...")
items = crossref_search("Channel Adaptation EEG Foundation Models Systematic Benchmark")
for item in items:
    cr_title = ' '.join(item.get('title', ['']))
    sim = jaccard('Channel Adaptation for EEG Foundation Models A Systematic Benchmark', cr_title)
    print(f"  CrossRef: '{cr_title[:60]}' sim={sim:.3f}")
    if sim > 0.45:
        doi = item.get('DOI')
        if doi:
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                resolved['P76'] = {'key': 'chan2026channel', 'bib': bib, 'title': cr_title}
                break
if 'P76' not in resolved:
    results = ss_search('Channel Adaptation EEG Foundation Models Benchmark', sleep_before=30)
    for r in results:
        sim = jaccard('Channel Adaptation for EEG Foundation Models A Systematic Benchmark', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.45:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            doi = ext_ids.get('DOI')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
                resolved['P76'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P76: {arxiv_id}")
                break
if 'P76' not in resolved:
    unresolved_details['P76'] = "Not found"

print("\nP77: EEG-Deformer...")
results = search_arxiv('EEG-Deformer dense convolutional transformer brain-computer interface')
for r in results:
    sim = jaccard('EEG-Deformer A Dense Convolutional Transformer for Brain-Computer Interfaces', r['title'])
    print(f"  arXiv: '{r['title'][:60]}' sim={sim:.3f}")
    if sim > 0.45:
        key = make_key(r['authors'], r['year'], r['title'])
        bib = make_misc_bibtex(key, r['title'], r['authors'], r['year'], r['arxiv_id'], 'eess.SP')
        resolved['P77'] = {'key': key, 'bib': bib, 'title': r['title']}
        break
if 'P77' not in resolved:
    results = ss_search('EEG-Deformer Dense Convolutional Transformer Brain-Computer Interface', sleep_before=30)
    for r in results:
        sim = jaccard('EEG-Deformer A Dense Convolutional Transformer for Brain-Computer Interfaces', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.45:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            doi = ext_ids.get('DOI')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
                resolved['P77'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P77: {arxiv_id}")
                break
            elif doi and not doi.startswith('10.48550'):
                bib = fetch_doi_bibtex(doi)
                if bib and len(bib) > 50:
                    resolved['P77'] = {'key': key, 'bib': bib, 'title': ret_title}
                    print(f"  -> RESOLVED P77 via DOI")
                    break
if 'P77' not in resolved:
    unresolved_details['P77'] = "Not found"

print("\nP78: NeuroPhysNet FitzHugh-Nagumo...")
results = search_arxiv('NeuroPhysNet FitzHugh-Nagumo physics-informed EEG neural network')
for r in results:
    sim = jaccard('NeuroPhysNet A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis', r['title'])
    print(f"  arXiv: '{r['title'][:60]}' sim={sim:.3f}")
    if sim > 0.40:
        key = make_key(r['authors'], r['year'], r['title'])
        bib = make_misc_bibtex(key, r['title'], r['authors'], r['year'], r['arxiv_id'], 'eess.SP')
        resolved['P78'] = {'key': key, 'bib': bib, 'title': r['title']}
        break
if 'P78' not in resolved:
    results = ss_search('NeuroPhysNet FitzHugh-Nagumo Physics-Informed Neural Network EEG', sleep_before=30)
    for r in results:
        sim = jaccard('NeuroPhysNet A FitzHugh-Nagumo-Based Physics-Informed Neural Network for EEG Analysis', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.40:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            doi = ext_ids.get('DOI')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
                resolved['P78'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P78: {arxiv_id}")
                break
if 'P78' not in resolved:
    unresolved_details['P78'] = "Not found"

print("\nP97: Outlier Detection Riemannian manifold functional data...")
items = crossref_search("Outlier Detection Riemannian Manifold Functional Data Applied Intelligence")
for item in items:
    cr_title = ' '.join(item.get('title', ['']))
    sim = jaccard('Outlier Detection for Riemannian Manifold-Valued Functional Data', cr_title)
    print(f"  CrossRef: '{cr_title[:60]}' sim={sim:.3f}")
    if sim > 0.55:
        doi = item.get('DOI')
        if doi:
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                resolved['P97'] = {'key': 'outlier2026', 'bib': bib, 'title': cr_title}
                break
if 'P97' not in resolved:
    results = ss_search('Outlier Detection for Riemannian Manifold-Valued Functional Data', sleep_before=30)
    for r in results:
        sim = jaccard('Outlier Detection for Riemannian Manifold-Valued Functional Data', r.get('title',''))
        print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
        if sim > 0.55:
            ext_ids = r.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv')
            doi = ext_ids.get('DOI')
            authors = r.get('authors', [])
            year = str(r.get('year', '2026'))
            ret_title = r.get('title', '')
            key = make_key(authors, year, ret_title)
            if arxiv_id:
                bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'stat.ME')
                resolved['P97'] = {'key': key, 'bib': bib, 'title': ret_title}
                print(f"  -> RESOLVED P97: {arxiv_id}")
                break
            elif doi and not doi.startswith('10.48550'):
                bib = fetch_doi_bibtex(doi)
                if bib and len(bib) > 50:
                    resolved['P97'] = {'key': key, 'bib': bib, 'title': ret_title}
                    break
if 'P97' not in resolved:
    unresolved_details['P97'] = "Not found"

# NeurIPS 2025 / ICLR 2026 still not found: P73, P3, P46, P50, P96
print("\nP73: Gyrovector spaces attention NeurIPS 2025...")
results = ss_search('Gyrovector Spaces Attention Framework Matrix Manifolds NeurIPS 2025', sleep_before=30)
for r in results:
    sim = jaccard('Towards a General Attention Framework on Gyrovector Spaces for Matrix Manifolds', r.get('title',''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.45:
        ext_ids = r.get('externalIds', {})
        arxiv_id = ext_ids.get('ArXiv')
        doi = ext_ids.get('DOI')
        authors = r.get('authors', [])
        year = str(r.get('year', '2025'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'cs.LG')
            resolved['P73'] = {'key': key, 'bib': bib, 'title': ret_title}
            print(f"  -> RESOLVED P73: {arxiv_id}")
            break
if 'P73' not in resolved:
    unresolved_details['P73'] = "Not found on SS or arXiv"

print("\nP3: Riemannian High-Order Pooling brain foundation...")
results = ss_search('Riemannian High-Order Pooling Brain Foundation Models ICLR', sleep_before=30)
for r in results:
    sim = jaccard('Riemannian High-Order Pooling for Brain Foundation Models', r.get('title',''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.50:
        ext_ids = r.get('externalIds', {})
        arxiv_id = ext_ids.get('ArXiv')
        doi = ext_ids.get('DOI')
        authors = r.get('authors', [])
        year = str(r.get('year', '2026'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
            resolved['P3'] = {'key': key, 'bib': bib, 'title': ret_title}
            print(f"  -> RESOLVED P3: {arxiv_id}")
            break
        elif doi and not doi.startswith('10.48550'):
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                resolved['P3'] = {'key': key, 'bib': bib, 'title': ret_title}
                break
if 'P3' not in resolved:
    unresolved_details['P3'] = "Not found"

print("\nP50: PSDNorm sleep staging ICLR...")
results = ss_search('PSDNorm Temporal Normalization Sleep Staging ICLR 2026', sleep_before=30)
for r in results:
    sim = jaccard('PSDNorm Temporal Normalization for Deep Learning in Sleep Staging', r.get('title',''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.50:
        ext_ids = r.get('externalIds', {})
        arxiv_id = ext_ids.get('ArXiv')
        doi = ext_ids.get('DOI')
        authors = r.get('authors', [])
        year = str(r.get('year', '2026'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
            resolved['P50'] = {'key': key, 'bib': bib, 'title': ret_title}
            print(f"  -> RESOLVED P50: {arxiv_id}")
            break
if 'P50' not in resolved:
    unresolved_details['P50'] = "Not found on SS or arXiv"

print("\nP46: HybridRDG ASD EEG...")
results = ss_search('HybridRDG Zero-Shot ASD EEG Riemannian Domain Generalization', sleep_before=30)
for r in results:
    sim = jaccard('HybridRDG Zero-Shot ASD Biomarker Detection Multi-Paradigm EEG Hybrid Deep-Riemannian Domain Generalization', r.get('title',''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.35:
        ext_ids = r.get('externalIds', {})
        arxiv_id = ext_ids.get('ArXiv')
        doi = ext_ids.get('DOI')
        authors = r.get('authors', [])
        year = str(r.get('year', '2026'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
            resolved['P46'] = {'key': key, 'bib': bib, 'title': ret_title}
            print(f"  -> RESOLVED P46: {arxiv_id}")
            break
if 'P46' not in resolved:
    unresolved_details['P46'] = "Not found"

print("\nP96: NEED EEG NeurIPS 2025...")
results = ss_search('NEED cross-subject EEG video image reconstruction NeurIPS 2025', sleep_before=30)
for r in results:
    sim = jaccard('NEED Cross-Subject Cross-Task Generalization Video Image Reconstruction EEG', r.get('title',''))
    print(f"  SS: '{r.get('title','')[:60]}' sim={sim:.3f}")
    if sim > 0.45:
        ext_ids = r.get('externalIds', {})
        arxiv_id = ext_ids.get('ArXiv')
        doi = ext_ids.get('DOI')
        authors = r.get('authors', [])
        year = str(r.get('year', '2025'))
        ret_title = r.get('title', '')
        key = make_key(authors, year, ret_title)
        if arxiv_id:
            bib = make_misc_bibtex(key, ret_title, authors, year, arxiv_id, 'eess.SP')
            resolved['P96'] = {'key': key, 'bib': bib, 'title': ret_title}
            print(f"  -> RESOLVED P96: {arxiv_id}")
            break
        elif doi and not doi.startswith('10.48550'):
            bib = fetch_doi_bibtex(doi)
            if bib and len(bib) > 50:
                resolved['P96'] = {'key': key, 'bib': bib, 'title': ret_title}
                break
if 'P96' not in resolved:
    unresolved_details['P96'] = "Found on SS but no arXiv/DOI ID; needs manual lookup"

print("\n" + "=" * 60)
print(f"FINAL WAVE RESOLVED: {len(resolved)}")
for pid, v in resolved.items():
    print(f"  {pid}: {v['key']}")
print(f"\nFINAL UNRESOLVED: {list(unresolved_details.keys())}")
for pid, reason in unresolved_details.items():
    print(f"  {pid}: {reason}")

output = {'resolved': resolved, 'unresolved': unresolved_details}
with open('/tmp/final_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved to /tmp/final_results.json")
