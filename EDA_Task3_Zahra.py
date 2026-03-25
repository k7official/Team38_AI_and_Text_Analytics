"""
@author: Zahra A.

"""
from datasets import load_dataset
import pandas as pd

# Load the arXiv abstracts dataset
dataset = load_dataset("gfissore/arxiv-abstracts-2021", split="train")
print(dataset)

# Define the fields explored
fields = ["cs.AI", "cs.LG", "cs.CL"]  # AI, Machine Learning, NLP


# Filter by these categories
def filter_category(example):
    # Keep the paper if any of the desired fields are in its categories
    return any(field in example['categories'] for field in fields)

filtered = dataset.filter(filter_category)

# Remove missing or too-short abstracts
def valid_abstract(example):
    return example['abstract'] is not None and len(example['abstract']) > 50

filtered = filtered.filter(valid_abstract)

# Convert to pandas and keep only necessary columns
df = filtered.to_pandas()
df = df[['id', 'abstract', 'categories']]  

print("Shape after filtering:", df.shape)
print(df.head())

# Extract year from arXiv ID
def extract_year(arxiv_id):
    if arxiv_id.startswith("cs/"):
        arxiv_id = arxiv_id.split("/")[1]
    yy = int(arxiv_id[:2])
    return 1900 + yy if yy >= 91 else 2000 + yy

# Add year column to DataFrame
df['year'] = df['id'].apply(extract_year)

# Smart sampling
sample_size = 100000  # total rows that I want
if len(df) > sample_size:
    df = df.groupby('year', group_keys=False).apply(
        lambda x: x.sample(frac=min(1, sample_size/len(df)), random_state=42)) #early years and recent years are still represented.
#x.sample - randomly picks some papers from that year
# Prepare abstracts for analysis
docs = df['abstract']

# Quick check; df now contains ~100k rows, randomly sampled but still spread across all years.
print("Shape after sampling (if applied):", df.shape)
print(df['year'].value_counts().sort_index()) 
print(df.head())


# Number of papers per year
papers_per_year = df['year'].value_counts().sort_index()
print(papers_per_year)


import matplotlib.pyplot as plt
papers_per_year.plot(kind = 'bar', figsize = (15,5), title = "Number of Papers per Year")
plt.xlabel("Year")
plt.ylabel("Number of papers")
plt.show()


#Distribution of papers across the categories
print(df['categories'].value_counts())

# Count by FIRST category 

# Take the first category in the list for each paper
df['main_category'] = df['categories'].apply(lambda x: x[0])
print(df['main_category'].value_counts())

#got exactly same results. does this indicate that all papers are single-category? 


# Count how many papers have more than one category
multi_cat_count = df[df['categories'].apply(len) > 1].shape[0]
total_papers = df.shape[0]

print(f"Papers with multiple categories: {multi_cat_count}")
print(f"Total papers: {total_papers}")

# It is confirmed that all papers are within one category.

#Average abstract length over time
df['length'] = df['abstract'].apply(lambda x: len(x.split()))
avg_length = df.groupby('year')['length'].mean()
print(avg_length)

#Convert the abstract to lowercases
df['abstract_lower'] = df['abstract'].str.lower()
print(df.head())

#Papers per year by category

paper_per_year_cat =df.groupby(['year', 'main_category']).size().unstack(fill_value = 0) #unstack converts it to a table
paper_per_year_cat.plot(figsize = (12,5), title = "Papers per Year by Category")
plt.show()

#Track change over topics
#Start with Deep Learning

df['deep_learning'] = df['abstract_lower'].str.contains('deep learning')
df.groupby('year')['deep_learning'].mean().plot(title = 'Deep Learning Trend over the Years')
plt.show()


#Then with Neural Networks
df['neural_network'] = df['abstract_lower'].str.contains('neural network')
df.groupby('year')['neural_network'].mean().plot(title = 'Neural Network Trend over the Years')
plt.show()

#Deep Learning vs Neural Networks by Proportion
df['deep_learning'] = df['abstract_lower'].str.contains("deep learning", na = False)
df.groupby('year')['deep_learning'].mean().plot(label = 'Deep Learning')
df.groupby('year')['neural_network'].mean().plot(label = 'Neural Network') #Out of all papers in a year, what fraction mention neural networks?

plt.legend()
plt.title("Terminology Shift")
plt.show()

# Deep Learning vs Neural Network by Count
#This is necessary because proportion depends strongly on total papers
df.groupby('year')['neural_network'].sum().plot(title="Count NN")
plt.show()

df.groupby('year')['deep_learning'].sum().plot(title="Count DL")
plt.show()

#Year over Year Growth for these two categories
nn_count = df.groupby('year')['neural_network'].sum()
dl_count = df.groupby('year')['deep_learning'].sum()

nn_growth = nn_count.pct_change() * 100
dl_growth = dl_count.pct_change() * 100

plt.plot(nn_growth, label='NN % growth')
plt.plot(dl_growth, label='DL % growth')
plt.legend()
plt.title("Year-over-Year Growth Rate")
plt.show()


#bins that match the dataset
bins = [1993, 1998, 2003, 2008, 2013, 2018, 2022]
labels = ["1993-97", "1998-02","2003-07","2008-12", "2013-17", "2018-21"]

df['time_period'] = pd.cut(df['year'], bins=bins, labels=labels, right=False)

print(df[['year', 'time_period']].head())
print(df['time_period'].value_counts().sort_index())


import re
# text cleaning function
def simple_preprocess(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#Apply preprocessing to all abstracts and create a cleaned text column
df['clean_text'] = df['abstract'].apply(simple_preprocess)


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#Function to remove common English stopwords and very short words
def remove_stopwords(text):
    tokens = text.split()
    clean_tokens = []

    for t in tokens:
        if t not in ENGLISH_STOP_WORDS and len(t) > 2:
            clean_tokens.append(t)

    return " ".join(clean_tokens)

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert cleaned abstracts into TF-IDF vectors
# max_features=5000 keeps the 5000 most important unigrams/bigrams
# ngram_range=(1,2) includes both single words and two-word phrases
# stop_words='english' removes common English function words

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english') #Keep only the top 5000 most important words/bigrams in the entire dataset.

X_tfidf = tfidf.fit_transform(df['clean_text']) #transform = turn text into numbers


from sklearn.decomposition import PCA

# Reduce high-dimensional TF-IDF vectors to 2 dimensions for visualisation
pca = PCA(n_components=2)
X_pca_tfidf = pca.fit_transform(X_tfidf.toarray())

df['pca1_tfidf'] = X_pca_tfidf[:, 0]
df['pca2_tfidf'] = X_pca_tfidf[:, 1]


# First plotting all periods together
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
for period in df['time_period'].unique():
    subset = df[df['time_period'] == period]
    plt.scatter(subset['pca1_tfidf'], subset['pca2_tfidf'], alpha=0.3, label=period)

plt.legend()
plt.title("PCA of AI, ML and NLP Research (TF-IDF)")
plt.show()

#Plotting 3 different periods by Early, Transition, and Modern
groups = {
    "Early (1993–2007)": ["1993-97", "1998-02", "2003-07"],
    "Transition (2008–12)": ["2008-12"],
    "Modern (2013–21)": ["2013-17", "2018-21"]
}

for title, periods in groups.items():
    subset = df[df['time_period'].isin(periods)]
    
    plt.figure(figsize=(6,4))
    for p in periods:
        sub = subset[subset['time_period'] == p]
        plt.scatter(sub['pca1_tfidf'], sub['pca2_tfidf'], alpha=0.4, label=p)
    
    plt.title(f"PCA — {title}")
    plt.legend()
    plt.show()


df['clean_text'].to_csv("clean_texts.csv", index=False)


import numpy as np

# Load precomputed embedding vectors from file
embeddings = np.load("/Users/zehra/Dropbox/Mac/Downloads/embeddings.npy")

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

df['pca1_embed'] = X_pca[:, 0]
df['pca2_embed'] = X_pca[:, 1]

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df,
    x='pca1_embed',
    y='pca2_embed',
    hue='time_period',
    palette='tab10',
    alpha=0.6,
    s=20
)

plt.title("Embedding PCA by Time Period")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Time Period")
plt.show()

#Keyword grouping per category + evaluation framework

classical_ai = ["expert system", "knowledge representation", "reasoning", "planning", "ontology", "rule based"]
ml_general = ["machine learning", "supervised", "unsupervised", "reinforcement learning", "feature", "classifier"]
deep_learning = ["deep learning", "neural network", "convolutional", "recurrent neural", "transformer", "attention"]
nlp_terms = ["language model", "machine translation", "text","word embedding", "bert", "token", "corpus"]
evaluation = ["experiment", "results", "performance", "accuracy","benchmark", "dataset", "evaluation"]

keyword_groups = {
    "classical_ai": classical_ai,
    "ml_general": ml_general,
    "deep_learning": deep_learning,
    "nlp_terms": nlp_terms,
    "evaluation": evaluation
}

# For each keyword group, create a column showing
# whether an abstract contains any of the terms in that group

for group, terms in keyword_groups.items():
    pattern = r'\b(' + '|'.join(terms) + r')\b'
    df[group] = df['abstract_lower'].str.contains(pattern, na=False)
    
# Calculate yearly proportion of abstracts containing each keyword group
group_trends = df.groupby('year')[list(keyword_groups.keys())].mean()

#Plotting keywords
group_trends.plot(figsize=(12,6))
plt.title("Keyword Trends Over Time")
plt.ylabel("Proportion of abstracts")
plt.xlabel("Year")
plt.show()

