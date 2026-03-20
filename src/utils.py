# src/utils.py
# Shared utility functions for corpus analysis project

from datasets import load_dataset


def load_arxiv_corpus(field="cs.LG", min_abstract_length=50):
    """
    Load and filter the arXiv abstracts dataset.
    
    Args:
        field: arXiv category string e.g. 'cs.LG', 'cs.CL', 'cs.AI'
        min_abstract_length: minimum character length to keep
    
    Returns:
        filtered HuggingFace dataset
    """
    dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
    
    filtered = dataset.filter(lambda x: field in x['categories'])
    filtered = filtered.filter(
        lambda x: x['abstract'] is not None and len(x['abstract']) > min_abstract_length
    )
    return filtered


def extract_year(arxiv_id):
    """
    Infer publication year from arXiv paper ID.
    Handles both old-style (cs/YYMMNNN) and new-style (YYMM.NNNNN) IDs.
    """
    if arxiv_id.startswith("cs/"):
        arxiv_id = arxiv_id.split("/")[1]
    yy = int(arxiv_id[:2])
    return 1900 + yy if yy >= 91 else 2000 + yy


def get_docs_and_years(filtered_dataset):
    """
    Extract abstracts and inferred years from a filtered dataset.
    
    Returns:
        docs: list of abstract strings
        years: list of integers
    """
    years = [extract_year(x) for x in filtered_dataset['id']]
    docs = filtered_dataset['abstract']
    return list(docs), years
