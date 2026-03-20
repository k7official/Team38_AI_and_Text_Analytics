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
