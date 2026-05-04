# Comparative Corpus Analysis Project

Comparative corpus analysis of AI/ML/NLP research abstracts using the 
arXiv dataset, examining shifts in research focus, limitations framing, 
and terminology diffusion over time.

## Environment Setup

This project uses a shared conda environment. To set it up:
```bash
conda env create -f environment.yml
conda activate text_analytics
pip install bertopic umap-learn
python -m ipykernel install --user --name=text_analytics
```

## Data

The dataset is loaded automatically from HuggingFace inside the notebooks.
No manual download is required. Do not commit any data files to this repo.

## Project Structure
```
├── data/               # Local only — gitignored
├── notebooks/          # Analysis notebooks
├── src/utils.py        # Shared helper functions
├── outputs/            # Figures and result files
├── environment.yml     # Conda environment
└── README.md
```


# AI Research Evolution Through Comparative Corpus Analysis

## Overview

This project investigates how research in Artificial Intelligence (AI), Machine Learning (ML), and Natural Language Processing (NLP) evolved over time through large-scale analysis of scientific abstracts. Using multiple text representations and analytical methods, the project examines changes in research focus, terminology, semantic structure, and limitations framing across the AI literature.

The work was completed as a group comparative corpus analysis project and combines lexical, topical, embedding-based, and geometric approaches to analyse temporal drift in AI research discourse.

The project emphasises two methodological axes:

- **Text Representation**
  - Frequency
  - TF-IDF
  - Topic models (LDA, NMF)
  - Word embeddings (Word2Vec)
  - Sentence embeddings (Sentence-BERT)

- **Comparison Method**
  - Keyword trend analysis
  - Topic proportion analysis
  - Classification
  - Semantic drift analysis
  - PCA and UMAP trajectory analysis
  - Centroid distance analysis

The goal is not only to identify how AI research changed, but also to evaluate how different analytical methods affect the conclusions that can be drawn.

---

# Research Objectives

The project addresses the following core research question:

> How has the focus, framing, terminology, and semantic structure of AI research evolved over time, and how do different text analytics methods affect the insights that can be drawn?

Specific objectives include:

- Identifying shifts in research focus and methodology
- Analysing changes in limitations framing and ethical discourse
- Measuring terminology diffusion and semantic drift
- Visualising temporal movement through semantic space
- Comparing robustness and interpretability across methods

---

# Dataset

The corpus consists of scientific abstracts from AI-related arXiv categories:

- `cs.AI`
- `cs.LG`
- `cs.CL`

The dataset includes metadata such as:
- title
- abstract
- year
- authors
- category

Data source:
- HuggingFace scientific abstracts dataset
- Text Analytics coursework repository utilities

---

# Project Structure

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_thread1_focus_shift_lexical.ipynb
│   ├── 03_thread1_focus_shift_topics.ipynb
│   ├── 04_thread2_limitations.ipynb
│   ├── 05_thread3_terminology.ipynb
│   └── 06_pca_temporal_drift.ipynb
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
│
├── report/
│   └── final_report.pdf
│
└── README.md
```

---

# Methodology

## 1. Preprocessing

Common preprocessing pipeline:
- lowercasing
- punctuation removal
- tokenisation
- stopword removal
- lemmatisation
- filtering short/empty abstracts

Additional preprocessing varied depending on representation.

---

# Analytical Threads

## Thread 1 — Research Focus Shift

### Lexical Analysis
- Keyword frequency tracking
- TF-IDF weighting
- Temporal trend analysis

### Topic Modelling
- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorisation (NMF)

Purpose:
- Detect thematic evolution
- Compare topic coherence and interpretability

---

## Thread 2 — Limitations Framing

Analyses how researchers discuss:
- limitations
- risks
- ethical concerns
- uncertainty

Methods:
- lexical trend analysis
- TF-IDF classification
- temporal framing comparison

Key finding:
- strong rise in ethics/fairness-related framing post-2017

---

## Thread 3 — Terminology Diffusion

Tracks how terminology:
- emerges
- spreads
- changes semantically

Methods:
- TF-IDF
- Word2Vec semantic neighbourhood analysis

Key finding:
- conceptual change often precedes widespread vocabulary adoption

---

## Thread 4 — Representation, PCA & Temporal Drift

Analyses temporal movement through semantic space using:
- TF-IDF vectors
- Sentence-BERT embeddings

Methods:
- PCA
- UMAP
- centroid trajectory analysis

Key findings:
- disciplinary structure dominates early PCA components

---

# Key Findings

## 1. Neural Transition (2014–2016)
All major methods identify 2014–2016 as the strongest period of structural movement in the corpus.

## 2. Expansion Rather Than Replacement
UMAP and clustering analysis suggest that new paradigms expanded existing research areas rather than replacing them entirely.

## 3. Ethics and Social Impact Shift
Later embedding components reveal a strong temporal movement toward fairness, explainability, and societal impact research.

## 4. Representation Sensitivity
- TF-IDF detects vocabulary-level change
- Embeddings detect conceptual change earlier

## 5. Cross-Method Robustness
Agreement across lexical, topical, and embedding methods strengthens confidence in the observed trends.

---

# Technologies Used

## Python Libraries
- pandas
- numpy
- scikit-learn
- gensim
- sentence-transformers
- umap-learn
- matplotlib
- seaborn
- scipy

## Models
- TF-IDF
- Word2Vec
- all-MiniLM-L6-v2 Sentence-BERT

---

# Running the Project

## Environment Setup

This project uses a shared conda environment. To set it up:
```bash
conda env create -f environment.yml
conda activate text_analytics
pip install bertopic umap-learn
python -m ipykernel install --user --name=text_analytics
```

Recommended execution order:

1. `01_eda_preprocessing.ipynb`
2. `02_thread1_focus_shift_lexical.ipynb`
3. `03_thread1_focus_shift_topics.ipynb`
4. `04_thread2_limitations.ipynb`
5. `05_thread3_terminology.ipynb`
6. `06_pca_temporal_drift.ipynb`

---

# Limitations

- arXiv categories do not fully represent all AI research
- embeddings reduce interpretability compared to lexical methods
- PCA prioritises dominant variance rather than temporal relevance
- temporal signals can lag behind landmark publications

---

# Conclusion

This project demonstrates that AI research evolution is multi-layered:
- lexical,
- semantic,
- thematic,
- and structural changes occur simultaneously but not synchronously.

No single method fully captures the evolution of the field. Robust conclusions emerge through convergence across multiple representations and analytical approaches.

---

# License

This project was developed for academic coursework purposes.
