"""
Created on Tue Mar 31 20:50:27 2026

@author: Zahra A.
"""

from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="/Users/zehra/cleaned_dataset.parquet",
    split="train"
)

print(dataset)
print(dataset[0])


""" 
Create a string version of each abstract by joining tokens.

The original 'abstract' column is a list of words, which is suitable for
vector-based methods but not for phrase detection. This step converts it
into a single text string, allowing keyword and phrase-based analysis.

Returns
-------
Dataset with an added 'abstract_text' column.

"""

dataset = dataset.map(lambda x: {
    "abstract_text": " ".join(x["abstract"])})


df = dataset.to_pandas()
print(df.head())
print(df.columns)


"""
Define keyword sets for different types of limitations framing.

Four categories are used:
- Failure: direct statements of model failure (e.g. 'fail', 'failure')
- Capability: expressions of constraints or limitations (e.g. 'limited', 'unable')
- Hedging: indirect or uncertain language (e.g. 'may', 'could')
- Ethics/societal: references to broader impacts (e.g. 'bias', 'fairness')

These categories support lexical tracking of how limitations are expressed over time.

"""

failure_terms = ["fail", "fails", "failed", "failure"]

capability_terms = ["limit", "limited", "limitation", "limitations",
                    "cannot", "unable", "insufficient"]

hedging_terms = ["may", "might", "could", "possibly", "likely",
                 "suggest", "appear", "seem"]

ethics_terms = ["bias", "fairness", "ethical", "privacy",
                "risk", "harm", "safety", "societal"]

"""
Check whether any term from a given list appears in the text.

Returns 1 if at least one term is present, otherwise 0.
Used to create binary indicators for limitation categories.

"""

def contains_term(text, terms):
    return int(any(term in text for term in terms))

df["failure_present"] = df["abstract_text"].apply(lambda x: contains_term(x, failure_terms))
df["capability_present"] = df["abstract_text"].apply(lambda x: contains_term(x, capability_terms))
df["hedging_present"] = df["abstract_text"].apply(lambda x: contains_term(x, hedging_terms))
df["ethics_present"] = df["abstract_text"].apply(lambda x: contains_term(x, ethics_terms))

"""
Aggregate limitation indicators by time period.

Computes the proportion of abstracts in each period that contain
each type of limitation (failure, capability, hedging, ethical), enabling
comparison of trends over time.
"""

trend = df.groupby("period")[[
    "failure_present",
    "capability_present",
    "hedging_present",
    "ethics_present"
]].mean()

"""
Plot trends in limitations framing over time.

Visualises how the proportion of abstracts mentioning different
types of limitations changes across time periods.
"""

import matplotlib.pyplot as plt

trend.plot(marker="o")
plt.title("Limitations Framing Over Time")
plt.xticks(rotation=30)
plt.show()


#Frequency - "how strongly is it expressed?" 
def count_terms(text, terms):
    return sum(text.count(term) for term in terms)

#Main comparison - failure vs ethics
df["failure_count"] = df["abstract_text"].apply(lambda x: count_terms(x, failure_terms))
df["ethics_count"] = df["abstract_text"].apply(lambda x: count_terms(x, ethics_terms))


freq_trend = df.groupby("period")[["failure_count", "ethics_count"]].mean()
freq_trend.plot(marker = "o")
plt.title("Frequency of Limitations Over Time")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()

""" 
A tone score to check whether limitations are expressed more explicitly or cautiously.
- +1 --> explicit failure language 
- -1 --> hedging language

"""

df["tone_score"] = df["failure_present"] - df["hedging_present"] #checking whether the paper is more direct or cautious

df.groupby("period")["tone_score"].mean().plot(marker="o")
plt.title("Tone Shift: Explicit vs Hedging")
plt.xticks(rotation=45) 
plt.show()


df["ethics_and_failure"] = ( #whether the paper contains both ethic and failure words
    (df["ethics_present"] == 1) &
    (df["failure_present"] == 1)
).astype(int) #checking whether ethics 


"""
Aggregreate and plot the co-occurence over the time frame.
The mean per period --> The proportion of abstracts that contain both ethical concerns and explicit limitations.
"""
df.groupby("period")["ethics_and_failure"].mean().plot(marker="o")
plt.title("Ethics + Explicit Limitations")
plt.xticks(rotation=45) 
plt.show()


#from sentence_transformers import SentenceTransformer

"""
Generate semantic embeddings for each abstract using a lightweight
Sentence-BERT model.

A smaller model is used here to keep runtime manageable while still
capturing semantic similarity between abstracts.
"""

#model = SentenceTransformer("all-MiniLM-L6-v2")

#embeddings = model.encode(
#    df["abstract_text"].tolist(),
#    convert_to_numpy=True,
#   show_progress_bar=True,
#    batch_size=32)

#print("Embedding shape:", embeddings.shape)

#import numpy as np

"""
Save embeddings to disk so they don’t need to be recomputed.
"""

#np.save("/Users/zehra/abstract_embeddings.npy", embeddings)

#print("Embeddings saved successfully.")

import numpy as np
from numpy.linalg import norm

"""
Load the above precomputed embeddings.

"""

load_embeddings = np.load("/Users/zehra/abstract_embeddings.npy")
print("Loaded embeddings:", load_embeddings.shape)

"""
Compute one semantic centroid for each period.

Each centroid will be the average embedding vector for all abstracts in each period. 

"""

periods = sorted(df["period"].dropna().unique())

centroids = {}

for p in periods:
    mask = df["period"] == p
    centroids[p] = load_embeddings[mask.values].mean(axis = 0)


"""
Calculate the semantic shift between consecutive periods.

As a metric, Euclidean distance is used between centroids. This is done to estimate how much the semantic representation changed from one period to the next.

Higher values --> more/stronger changes

"""

distances = []
labels = []

for i in range(len(periods) - 1):
    p1 = periods[i]
    p2 = periods[i+1]
    
    dist = norm(centroids[p2] - centroids[p1])

    distances.append(dist)
    labels.append(f"{p1} -- {p2}")
    
print("Semantic distances between consecutive periods:")

for label, dist in zip(labels, distances):
    print(label, round(dist, 4))
    
    

"""
Plot semantic shift over time.

"""

plt.figure(figsize = (9,5))
plt.plot(labels, distances, marker = "o")
plt.title("Semantic Shift Between Periods (Embeddings)")
plt.xlabel("Period Transition")
plt.ylabel("Centroid Distance")
plt.xticks(rotation = 30)
plt.tight_layout()
plt.show()

"""
Filter the dataset to limitation-related abstracts only.

This makes the semantic analysis more directly relevant to
limitations framing rather than overall research trends.
"""
limitation_mask = (
    (df["failure_present"] == 1) |
    (df["capability_present"] == 1) |
    (df["ethics_present"] == 1)
)

df_lim = df[limitation_mask].copy()
embeddings_lim = load_embeddings[limitation_mask.values]

print("Filtered limitation-related abstracts:", len(df_lim))
print("Filtered embeddings shape:", embeddings_lim.shape)


"""
Compute one semantic centroid for each period.

Each centroid is the average embedding vector for all
limitation-related abstracts in that period.
"""
periods_lim = sorted(df_lim["period"].dropna().unique())

centroids_lim = {}

for p in periods_lim:
    mask_p = df_lim["period"] == p
    centroids_lim[p] = embeddings_lim[mask_p.values].mean(axis=0)


"""
Calculate semantic shift between consecutive periods.

Euclidean distance between centroids is used to estimate how much
limitations discourse changes from one period to the next.

Higher values indicate stronger semantic change.
"""
distances_lim = []
labels_lim = []

for i in range(len(periods_lim) - 1):
    p1 = periods_lim[i]
    p2 = periods_lim[i + 1]

    dist = norm(centroids_lim[p2] - centroids_lim[p1])

    distances_lim.append(dist)
    labels_lim.append(f"{p1} -- {p2}")

print("Semantic distances between consecutive periods (limitations only):")
for label, dist in zip(labels_lim, distances_lim):
    print(label, round(dist, 4))


"""
Plot semantic shift in limitations framing over time.
"""
plt.figure(figsize=(9, 5))
plt.plot(labels_lim, distances_lim, marker="o")
plt.title("Semantic Shift in Limitations Framing (Embeddings)")
plt.xlabel("Period Transition")
plt.ylabel("Centroid Distance")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

 
# Logistic Regression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


#create labels (early vs late)

df_clf = df.copy()
df_clf = df_clf[df_clf["year"].notnull()]

"""
Create binary labels for below period classification.

0 --> early papers (before 2010)
1 --> late papers (2015 and later)

"""

df_clf["label"] = df_clf["year"].apply(
    lambda x: 0 if x < 2010 else (1 if x >= 2015 else None))

df_clf = df_clf.dropna(subset = ["label"])

#Vectorize

vectorizer = TfidfVectorizer(max_features = 5000)
X = vectorizer.fit_transform(df_clf["abstract_text"])
y = df_clf["label"]

# Train the model

"""
Train the logistic regression.
The model learns the most predictive words in each period.

"""

model = LogisticRegression(max_iter = 1000)
model.fit(X,y)

# Extract features


"""
Extract and rank the top words.

Positive coeffs --> late papers
Negative coeffs --> early papers

"""

feature_names = vectorizer.get_feature_names_out()
coeffs = model.coef_[0]

top_late = sorted(zip(coeffs, feature_names), reverse = True) [:20]
top_early = sorted(zip(coeffs, feature_names))[:20]

"""
Print top words for each period.
"""


print("Top LATE words:")
for c,w in top_late:
    print(w, round(c,3))
    
print("\nTop EARLY words:")
for c,w in top_early:
    print(w, round(c,3))


"""
Create a limitations-focused subset for classification.

Work only with abstracts that contain at least one limitation-related context.
This reduces the influence of broad topic shifts and focuses more on how limitations have been framed over time.

"""

df_lim = df[(df["failure_present"] == 1) |
            (df["capability_present"] == 1) |
            (df["hedging_present"] == 1) |
            (df["ethics_present"] == 1)].copy()

print("Number of limitation-related papers:", len(df_lim))


"""
Create binary period labels for this classification.

0 = earlier papers
1 = later papers

The period is mainly focused on pre-deep learning and post-deep learning eras.

"""

df_lim["label"] = df_lim["year"].apply(lambda x: 0 if x < 2010 else (1 if x >= 2015 else None))

df_lim = df_lim.dropna(subset=["label"]).copy()
print(df_lim["label"].value_counts())


"""

Convert these selected abstracts into TF-IDF features.

max_features --> the vocabulary size the study will look at to reduce noise.

"""
vectorizer_lim = TfidfVectorizer(max_features = 3000)
X_lim = vectorizer_lim.fit_transform(df_lim["abstract_text"])
y_lim = df_lim["label"]


"""
Train a logistic regression classifier on the limitation-focused subset.

"""
model_lim = LogisticRegression(max_iter = 1000)
model_lim.fit(X_lim, y_lim)


"""
Extract the top words from earlier and later papers.
Positive coefficients = later papers
Negative coefficients = earlier papers

"""

feature_names_lim = vectorizer_lim.get_feature_names_out()
coeffs_lim = model_lim.coef_[0]

top_late_lim = sorted(zip(coeffs_lim, feature_names_lim), reverse=True)[:20]
top_early_lim = sorted(zip(coeffs_lim, feature_names_lim))[:20]

print("Top words for LATE limitation-related papers:")
for coef, word in top_late_lim:
    print(word, round(coef, 3))

print("\nTop words for EARLY limitation-related papers:")
for coef, word in top_early_lim:
    print(word, round(coef, 3))