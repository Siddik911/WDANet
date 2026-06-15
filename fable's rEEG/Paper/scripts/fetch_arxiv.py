#!/usr/bin/env python3
"""Fetch BibTeX for arXiv preprints via arXiv API."""
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time
import json
import re

def jaccard(a, b):
    """Word-level Jaccard similarity."""
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def make_key(authors, year, title):
    """firstauthorYEARfirstword"""
    # Get first author surname
    if not authors:
        return f"unknown{year}key"
    first_author = authors[0]
    # Handle "Last, First" or "First Last"
    if ',' in first_author:
        surname = first_author.split(',')[0].strip()
    else:
        parts = first_author.strip().split()
        surname = parts[-1] if parts else 'unknown'
    surname = surname.lower()
    surname = re.sub(r'[^a-z]', '', surname)

    # Get first non-trivial word
    skip = {'a', 'an', 'the', 'on', 'of', 'for', 'in', 'to', 'and', 'with', 'via', 'by', 'from'}
    words = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().split()
    firstword = 'paper'
    for w in words:
        if w not in skip and len(w) > 1:
            firstword = re.sub(r'[^a-z0-9]', '', w)
            break

    return f"{surname}{year}{firstword}"

def search_arxiv(title):
    """Search arXiv for a paper by title. Returns list of results."""
    encoded = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=ti:{encoded}&max_results=3&sortBy=relevance"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_data = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  ERROR fetching arXiv for '{title[:50]}': {e}")
        return []

    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    try:
        root = ET.fromstring(xml_data)
    except Exception as e:
        print(f"  ERROR parsing XML for '{title[:50]}': {e}")
        return []

    results = []
    for entry in root.findall('atom:entry', ns):
        arxiv_id_full = entry.find('atom:id', ns)
        ret_title = entry.find('atom:title', ns)
        published = entry.find('atom:published', ns)
        authors = entry.findall('atom:author', ns)

        if arxiv_id_full is None or ret_title is None:
            continue

        arxiv_url = arxiv_id_full.text.strip()
        # Extract ID: https://arxiv.org/abs/2503.12345v1 -> 2503.12345
        arxiv_id = re.sub(r'v\d+$', '', arxiv_url.split('abs/')[-1])

        title_clean = ' '.join(ret_title.text.strip().split())
        year = published.text[:4] if published is not None else '2026'

        author_names = []
        for a in authors:
            name_el = a.find('atom:name', ns)
            if name_el is not None:
                author_names.append(name_el.text.strip())

        sim = jaccard(title, title_clean)
        results.append({
            'arxiv_id': arxiv_id,
            'arxiv_url': arxiv_url,
            'title': title_clean,
            'year': year,
            'authors': author_names,
            'similarity': sim
        })

    return results

def make_bibtex(paper_id, result, primary_class='eess.SP'):
    """Generate @misc BibTeX from arXiv result."""
    authors = result['authors']
    year = result['year']
    title = result['title']
    arxiv_id = result['arxiv_id']

    key = make_key(authors, year, title)

    # Format authors
    author_str = ' and '.join(authors)

    bib = f"""@misc{{{key},
  title        = {{{title}}},
  author       = {{{author_str}}},
  year         = {{{year}}},
  eprint       = {{{arxiv_id}}},
  archivePrefix= {{arXiv}},
  primaryClass = {{{primary_class}}},
  url          = {{https://arxiv.org/abs/{arxiv_id}}},
  note         = {{arXiv preprint}}
}}"""
    return key, bib

# Papers to search
papers = [
    ('P13', 'RepSPD: Enhancing SPD Manifold Representation in EEGs via Dynamic Graphs', 'eess.SP'),
    ('P14', 'Riemannian Geometry-Preserving Variational Autoencoder for MI-BCI Data Augmentation', 'eess.SP'),
    ('P15', 'Riemannian Geometry Meets fMRI: The Advantages of Modeling Correlation Manifolds and Eigenvector Subspaces', 'eess.SP'),
    ('P26', 'Routing on the Stiefel Manifold: When Does Adaptive Subspace Selection Help for EEG Decoding', 'eess.SP'),
    ('P51', 'NeuroBRIDGE: Behavior-Conditioned Koopman Dynamics with Riemannian Alignment for Early Substance Use Initiation Prediction from Longitudinal Functional Connectome', 'eess.SP'),
    ('P52', 'Real-Time Decoding of Movement Onset and Offset for Brain-Controlled Rehabilitation Exoskeleton', 'eess.SP'),
    ('P54', 'Cross-Subject Generalization for EEG Decoding: A Survey of Deep Learning Methods', 'eess.SP'),
    ('P67', 'FedSPDnet: Geometry-Aware Federated Deep Learning with SPDnet', 'cs.LG'),
    ('P68', 'Riemannian Networks over Full-Rank Correlation Matrices', 'cs.LG'),
    ('P69', 'SPD Matrix Learning for Neuroimaging Analysis: Perspectives Methods and Challenges', 'cs.LG'),
    ('P72', 'A Unified SPD Token Transformer Framework for EEG Classification Systematic Comparison of Geometric Embeddings', 'eess.SP'),
    ('P75', 'SPDLearn: A Geometric Deep Learning Python Library for Neural Decoding Through Trivialization', 'cs.LG'),
    ('P80', 'Latte: Hyperbolic Lorentz Attention for Joint-Subject EEG Classification', 'eess.SP'),
    ('P82', 'EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts', 'eess.SP'),
    ('P84', 'Beyond Rigid Geometries: The Spline-Pullback Metric for Universal Diffeomorphic Autoencoders', 'cs.LG'),
    ('P85', 'Riemannian Adversarial Attacks on Symmetric Positive Definite Matrices', 'cs.LG'),
    ('P86', 'Riemannian Block SPD Coupling Manifold and Its Application to Optimal Transport', 'cs.LG'),
    ('P88', 'Riemannian Optimization over Symmetric Positive Definite Matrices with the Alpha-Procrustes Metric', 'cs.LG'),
    ('P91', 'Riemannian Diffusion Models on General Manifolds via Physics-Informed Neural Networks', 'cs.LG'),
    ('P92', 'A Riemannian Quasi-Newton Algorithm for Optimization with Euclidean Bounds', 'cs.LG'),
    ('P93', 'Multivariate Intrinsic Local Polynomial Regression on Isometric Riemannian Manifolds', 'math.ST'),
    ('P94', 'Batch Normalization for Neural Networks on Complex Domains', 'cs.LG'),
]

resolved = {}
unresolved = []

print("=" * 60)
print("CATEGORY 1: arXiv searches")
print("=" * 60)

for paper_id, title, pclass in papers:
    print(f"\n{paper_id}: {title[:60]}...")
    results = search_arxiv(title)

    if not results:
        print(f"  -> No results from API")
        unresolved.append(paper_id)
        time.sleep(0.5)
        continue

    best = max(results, key=lambda r: r['similarity'])
    print(f"  Best match: '{best['title'][:60]}' (sim={best['similarity']:.3f})")

    if best['similarity'] > 0.55:
        key, bib = make_bibtex(paper_id, best, pclass)
        print(f"  -> RESOLVED: key={key}, arXiv={best['arxiv_id']}")
        resolved[paper_id] = {'key': key, 'bib': bib, 'title': best['title']}
    else:
        print(f"  -> REJECTED (sim too low)")
        unresolved.append(paper_id)

    time.sleep(0.5)

print("\n" + "=" * 60)
print(f"RESOLVED: {len(resolved)}/{len(papers)}")
print(f"UNRESOLVED: {unresolved}")

# Save results
output = {
    'resolved': resolved,
    'unresolved': unresolved
}
with open('/tmp/arxiv_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved to /tmp/arxiv_results.json")
