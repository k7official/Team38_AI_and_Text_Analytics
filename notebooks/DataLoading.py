#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:00:45 2026

@author: zehra
"""

import re

from datasets import load_dataset, Features, Value, Sequence

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# 1. LOAD DATASET

dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
print("Initial dataset size:", len(dataset))

# 2. ID STANDARDIZATION
"""
Normalise an arXiv identifier to the canonical ``YYMM.NNNNN`` format.

Handles both legacy subject-prefixed IDs and modern numeric IDs, and 
strips trailing version suffixes (e.g. ``hep-th``).
 
Parameters
----------
arxiv_id : str or None
Raw arXiv identifier as stored in the dataset.
 
Returns
-------
str or None
Standardised ID string, or ``None`` if the input is empty or does
not match either recognised format.
"""

_NEW_ID  = re.compile(r'^\d{4}\.\d{4,5}$')
_OLD_ID  = re.compile(r'^[a-z\-]+/\d{7}$')
_VERSION = re.compile(r'v\d+$')
 
 
def standardize_id(arxiv_id):
    if not arxiv_id:
        return None
    arxiv_id = arxiv_id.strip()
    arxiv_id = _VERSION.sub('', arxiv_id)
    if _NEW_ID.match(arxiv_id):
        return arxiv_id
    if _OLD_ID.match(arxiv_id):
        _, num = arxiv_id.split("/")
        return f"{num[:4]}.{num[4:]}"
    return None



# 3. YEAR EXTRACTION
"""
Derive the 4-digit publication year from an arXiv identifier.
 
The year is encoded in the first two digits of the numeric part of the ID.
The century is resolved by the following rule:
 
* ``yy >= 91``  ->  ``1900 + yy``  (covers 1991 - 1999)
* ``yy <  91``  ->  ``2000 + yy``  (covers 2000 - present)
 
Returns a plain Python ``int`` (not numpy) to avoid Apache Arrow type
conflicts when writing to a HuggingFace ``Dataset``.
 
Parameters
----------
arxiv_id : str or None
Raw or standardised arXiv identifier.
 
Returns
-------
int or None
4-digit year, or ``None`` if the ID is missing or unrecognised.
"""

def extract_year(arxiv_id):
    if not arxiv_id:
        return None
    try:
        arxiv_id = _VERSION.sub('', arxiv_id.strip())
        if "/" in arxiv_id:
            yy = int(arxiv_id.split("/")[1][:2])
        elif _NEW_ID.match(arxiv_id):
            yy = int(arxiv_id[:2])
        else:
            return None
        return int(1900 + yy) if yy >= 91 else int(2000 + yy)   # explicit int
    except Exception:
        return None
    
    
    
    
# 4. TEXT CLEANING
"""
Aggressively normalise abstract text to plain ASCII lowercase.
Operations applied (in order):
 
1. Strip leading / trailing whitespace.
2. Replace non-ASCII characters (e.g. accented letters, math symbols) with a space.
3. Replace control characters (``\\n``, ``\\r``, ``\\t``, DEL) with a space.
4. Remove all punctuation, keeping only word characters and spaces.
5. Collapse whitespace runs to a single space.
6. Convert to lowercase.
 
Parameters
----------
text : str or None
Raw abstract string from the dataset.
 
Returns
-------
str or None
Cleaned abstract string, or ``None`` if the input is empty.
 
Notes
-----
This function is intended to be composed with :func:`tokenize_lemmatize`,
which performs stopword removal and lemmatization on the output.
""" 

_WHITESPACE = re.compile(r'\s+')
_NON_ASCII  = re.compile(r'[^\x00-\x7F]+')
_CONTROL    = re.compile(r'[\n\r\t\x7f]')
_ALL_PUNCT  = re.compile(r'[^\w\s]')
 
 
def clean_title(title):
    if not title:
        return None
    return _WHITESPACE.sub(" ", title.strip()).lower()


def clean_abstract(text):
    if not text:
        return None
    text = text.strip()
    text = _NON_ASCII.sub(" ", text)
    text = _CONTROL.sub(" ", text)
    text = _ALL_PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).lower()

# 5. CATEGORY EXTRACTION
"""Return the primary arXiv category if it belongs to the target set.
 
arXiv papers carry one or more category tags; the **first** tag is always
the primary (submitter-assigned) category.  Only papers whose primary
category is in ``TARGET_CATEGORIES`` are retained; all others return
``None`` and are removed during the final filter step.
 
Parameters
----------
categories : str, list, or None
Raw category field from the dataset.  When stored as a string, multiple categories are space-separated (e.g. ``"cs.lg stat.ml"``).
 
Returns
-------
str or None
Lowercase primary category (e.g. ``"cs.lg"``), or ``None`` if the
primary category is not in ``TARGET_CATEGORIES`` or the field is empty.
 
Notes
-----
``TARGET_CATEGORIES = {"cs.lg", "cs.cl", "cs.ai"}`` 
``cs.cl`` is Natural Language Processing.
``cs.ai`` is Artificial Intelligence.
``cs.ml`` is Machine Learning.
"""

TARGET_CATEGORIES = {"cs.lg", "cs.cl", "cs.ai"}
 
def extract_primary_category(categories):
    if not categories:
        return None
    if isinstance(categories, str):
        categories = categories.split()
    # Only the first entry is the true primary category on arXiv
    primary = categories[0].lower()
    return primary if primary in TARGET_CATEGORIES else None


# 6. PERIOD CREATION (5-YEAR)
"""
Map a publication year to a 5-year time-window label.
 
The window boundaries are computed by flooring the year to the nearest
multiple of 5::
 
start = (year // 5) * 5
label = f"{start}-{start + 4}"
 
Parameters
----------
year : int or None
4-digit publication year.
 
Returns
-------
str or None
Period label string, or ``None`` if *year* is ``None``.
"""

def assign_period(year):
    if year is None:
        return None
    start = (year // 5) * 5
    return f"{start}-{start + 4}"


# 7. TOKENIZATION + LEMMATIZATION
"""
Return filtered tokens shared by both lemmatization.
 
Splits *text* on whitespace and discards tokens that are non-alphabetic,
shorter than 3 characters, or present in ``ALL_STOPWORDS``.
 
Parameters
----------
text : str or None
     Pre-cleaned (lowercase, no punctuation) abstract string.
 
Returns
-------
list of str or None
    Filtered token list, or ``None`` if *text* is empty or all tokens
    are filtered out.
"""

lemmatizer = WordNetLemmatizer()
 
CUSTOM_STOPWORDS = {
    "paper", "show", "shown", "shows",
    "study", "work", "also", "used", "use", "using", "based", "data", "new", "two", "one", "may",
    "well", "different", "present", "problem", "consider",
    "however", "thus", "therefore", "find", "found", "give",
    "given", "provide", "large", "high", "many", "first",
    "set", "number", "form", "type", "way", "case", "order", "ha",
}
 
ALL_STOPWORDS = set(stopwords.words("english")) | CUSTOM_STOPWORDS
 
 
def _base_tokens(text):
    if not text:
        return None
    tokens = [
        t for t in text.split()
        if t.isalpha() and len(t) > 2 and t not in ALL_STOPWORDS
    ]
    return tokens if tokens else None
 

"""
Tokenise and lemmatise a cleaned abstract using WordNet.

Applies :func:`_base_tokens` filtering, then reduces each token to its
dictionary base form via ``WordNetLemmatizer``.

Parameters
----------
text : str or None
    Abstract string passed through :func:`clean_abstract`.

Returns
-------
list of str
    List of lemmatised tokens. Returns an empty list if the input is
    empty or all tokens were filtered out.
"""
 
def tokenize_lemmatize(text):
    tokens = _base_tokens(text)
    if tokens is None:
        return []
    return [lemmatizer.lemmatize(t) for t in tokens]


# 8a. BATCH CLEANING FUNCTION
"""
Apply the full cleaning pipeline to a batch of raw dataset records.
 
This function is designed to be passed to ``Dataset.map(..., batched=True)``.
It orchestrates all individual cleaning steps — ID standardisation, year
extraction, title/abstract cleaning, category extraction, and period
assignment — and returns only the six output columns required by the final
schema.
 
Parameters
----------
batch : dict[str, list]
      A dictionary of column name → list of values, as provided by the
      HuggingFace ``datasets`` batched map interface.  Expected keys:
      ``"id"``, ``"title"``, ``"abstract"``, ``"categories"``.
 
Returns
-------
dict[str, list]
      Dictionary with keys ``"id"``, ``"year"``, ``"title"``,
      ``"abstract"``, ``"category"``, ``"period"``, each mapping to a
      list of the same length as the input batch.  Any value may be
      ``None`` where the corresponding record could not be parsed or does
      not belong to a target category.
 
Notes
-----
``year`` is computed once and reused for both the ``"year"`` and
``"period"`` outputs to avoid redundant computation.
"""

def clean_batch(batch):
    ids           = batch["id"]
    years         = [extract_year(x) for x in ids]
    cleaned_texts = [clean_abstract(x) for x in batch["abstract"]]
 
    return {
        "id":               [standardize_id(x) for x in ids],
        "year":             years,
        "title":            [clean_title(x) for x in batch["title"]],
        "abstract":         [tokenize_lemmatize(t) for t in cleaned_texts],
        "category":         [extract_primary_category(x) for x in batch["categories"]],
        "period":           [assign_period(y) for y in years],
    }
# 8b. APPLY CLEANING PIPELINE

OUTPUT_FEATURES = Features({
    "id":       Value("string"),
    "year":     Value("int32"),
    "title":    Value("string"),
    "abstract": Sequence(Value("string")),
    "category": Value("string"),
    "period":   Value("string"),
})
 
dataset = dataset.map(
    clean_batch,
    batched=True,
    batch_size=1000,
    desc="Cleaning dataset",
    remove_columns=dataset.column_names,   # drop original schema
    features=OUTPUT_FEATURES,              # enforce output schema
    load_from_cache_file=False,            
)


# 9. FINAL FILTERING
"""
Records are removed if:

1. Abstract is missing
2. Abstract contains fewer than "50 words"
3. Year is missing

Threshold:
```
Minimum abstract length = 50 words
``` 
"""

MIN_ABSTRACT_WORDS = 50

dataset = dataset.filter(
    lambda x: (
        x["abstract"] is not None
        and len(x["abstract"]) >= MIN_ABSTRACT_WORDS  # length of token list
        and x["year"] is not None
        and x["category"] is not None
    ),
    desc="Filtering abstracts",
)


# 10. KEEP ONLY FINAL COLUMNS

dataset = dataset.remove_columns(
    [col
     for col in dataset.column_names
        if col not in
        [
            "id",
            "year",
            "title",
            "abstract",
            "category",
            "period"
        ]])


# 11. FINAL OUTPUT CHECK

print("\nFinal dataset structure:")
print(dataset)

print("\nFinal dataset size:")
print(len(dataset))

print("\nSample row:")
print(dataset[0])


# 11. SAVING THE CLEAN DATA
"""
Save the cleaned dataset to a Parquet file.

This command writes the processed dataset to disk in Parquet format,
which is a columnar, compressed storage format optimized for fast
read and write operations. Saving the dataset allows it to be reused
in other notebooks or stages of the data pipeline without repeating
the cleaning process.

Output:
    cleaned_dataset.parquet : Serialized dataset stored locally.
"""

dataset.to_parquet("cleaned_dataset.parquet")






