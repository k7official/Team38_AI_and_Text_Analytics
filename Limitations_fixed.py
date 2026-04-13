"""
Created on Sat Apr  4 14:22:36 2026

@author: Zahra A.
"""

from datasets import load_dataset
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


# LOAD THE CLEANED DATA

dataset = load_dataset("parquet", data_files = "/Users/zehra/cleaned_dataset.parquet", split = "train")


#Create Abstract Text
"""
Join abstract tokens into a string for each paper.
""" 

dataset = dataset.map(lambda x: {"abstract_text": " ".join(x["abstract"])})

df = dataset.to_pandas()
print(df.head())
print(df.columns)

# Turn the text to lowecase string

"""
Clean the texts --> fill missing values, convert everything to a lowercase
"""
df["abstract_text"] = df["abstract_text"].fillna("").astype(str).str.lower()


# DEFINE LIMITATION CATEGORIES

"""
Create keyword lists for different types of limitation categories.
"""

failure_terms = ["fail", "fails", "failed", "failure", "failing",
    "weakness", "weaknesses", "error", "errors",
    "poor", "degrade", "degrades", "degraded",
    "unstable", "fragile", "difficult", "challenging"]

capability_terms = [
    "limit", "limits", "limited", "limitation", "limitations",
    "cannot", "unable", "insufficient", "constraint", "constraints",
    "restricted", "restriction", "restricts",
    "inapplicable", "impractical", "inefficient", "bottleneck"]

hedging_terms = [
    "may", "might", "could", "possibly", "potentially", "likely",
    "suggest", "suggests", "suggested",
    "appear", "appears", "appeared",
    "seem", "seems", "seemed"]

ethics_terms = [
    "bias", "biased", "fairness", "ethical", "ethics",
    "privacy", "risk", "risks", "harm", "harms",
    "safety", "societal", "responsible", "accountability",
    "transparent", "transparency", "discrimination", "security"]

# Extra analysis: method introduction language
method_intro_terms = [
    "novel", "new", "propose", "proposes", "proposed",
    "introduce", "introduces", "introduced",
    "present", "presents", "presented",
    "framework", "approach", "method"]


# FUNCTIONS

def count_term_matches(text, terms):
    """
    Count exact word-boundary matches for a list of terms.
    
    """
    count = 0
    
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        count += len(re.findall(pattern, text))
        
    return count


def contains_term(text, terms):
    """
    Returns
    1 if any exact term appears, else 0.

    """

    return int(count_term_matches(text, terms) > 0)

#CREATE INDICATORS

df["failure_present"] = df["abstract_text"].apply(lambda x: contains_term(x, failure_terms))
df["capability_present"] = df["abstract_text"].apply(lambda x: contains_term(x, capability_terms))
df["hedging_present"] = df["abstract_text"].apply(lambda x: contains_term(x, hedging_terms))
df["ethics_present"] = df["abstract_text"].apply(lambda x: contains_term(x, ethics_terms))
df["method_intro_present"] = df["abstract_text"].apply(lambda x: contains_term(x, method_intro_terms))

# TECHNICAL LIMITATIONS CATEGORY -- failure or capability related limits

df["technical_present"] = ((df["failure_present"] == 1) | (df["capability_present"] ==1)).astype(int)

#CREATE THE COUNT PER KEYWORD CATEGORY -- the number of times the limitation related term appeared

df["failure_count"] = df["abstract_text"].apply(lambda x: count_term_matches(x, failure_terms))
df["capability_count"] = df["abstract_text"].apply(lambda x: count_term_matches(x, capability_terms))
df["hedging_count"] = df["abstract_text"].apply(lambda x: count_term_matches(x, hedging_terms))
df["ethics_count"] = df["abstract_text"].apply(lambda x: count_term_matches(x, ethics_terms))


# LEXICAL TREND ANALYSIS OVER TIME
"""
Find the proportion of abstracts containing each limitation category per period.
"""
trend = df.groupby("period")[["failure_present", "capability_present", "hedging_present", "ethics_present"]].mean()

# VISUALIZE THE TREND

trend.plot(marker ="o", figsize =(9,5))
plt.title("Limitations Framing Over Time")
plt.ylabel("Proportion of abstracts")
plt.xticks(rotation = 30)
plt.tight_layout()
plt.show()

# TECHNICAL VS ETHICAL

"""
Find the proportion of abstracts containing TECHNICAL AND ETHICAL LIMITATION categories per period.
"""
ethical_turn = df.groupby("period") [["technical_present", "ethics_present"]].mean()

# VISUALIZE IT
ethical_turn.plot(marker ="o", figsize = (9,5))
plt.title("Technical vs Ethical Limitations Over Time")
plt.ylabel("Proportion of abstracts")
plt.xticks(rotation = 30)
plt.tight_layout()
plt.show()


# TONE ANALYSIS

"""
Track explicit failure language and hedging language over time.
"""

tone_trend = df.groupby("period")[["failure_present", "hedging_present"]].mean()

tone_trend.plot(marker="o", figsize=(9, 5))
plt.title("Tone Shift: Explicit Failure vs Hedging")
plt.ylabel("Proportion of abstracts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ETHICS + EXPLICIT FAILURE CO-OCCURENCE

"""
Abstracts that include both ethical and failure related limitations
"""

df["ethics_and_failure"] = ((df["ethics_present"] == 1) & (df["failure_present"] == 1)).astype(int)

df.groupby("period")["ethics_and_failure"].mean().plot(marker = "o", figsize = (9,5))
plt.title("Co-occurrence: Ethics + Explicit Failure")
plt.ylabel("Proportion of abstracts")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.show()


# NEW METHODS VS LIMITATION 

df["any_limitation_present"] = (
    (df["failure_present"] == 1) |
    (df["capability_present"] == 1) |
    (df["hedging_present"] == 1) |
    (df["ethics_present"] == 1)
).astype(int)

"""
Compare limitations language in new method introducing vs any limitation(failure, capability, hedging, and ethics) papers.
"""
method_vs_lim = df.groupby("method_intro_present")["any_limitation_present"].mean()
print("\nProportion with any limitations language:")
print(method_vs_lim)


#GROUP BY PERIOD AND METHOD_INTRO_PRESENT

"""
Compare per period limitations language in new method introducing vs any limitation(failure, capability, hedging, and ethics) papers.

"""

method_period_trend = df.groupby(["period", "method_intro_present"])["any_limitation_present"].mean().unstack()

method_period_trend.plot(marker="o", figsize=(9, 5))
plt.title("Limitations Language: Method-Introducing vs Other Abstracts")
plt.ylabel("Proportion of abstracts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# CREATE A LIMITATION SUBSET

"""
Filter to get the abstracts that contains limitation-related language.

"""

df_lim = df[(df["failure_present"] == 1) |
            (df["capability_present"] == 1) |
            (df["hedging_present"] == 1) |
            (df["ethics_present"] == 1)].copy()

print("Number of limitation-related papers:", len(df_lim))


# PERIOD CLASSIFICATION
"""
Focus on the deep learning period by dividing into two distinct periods.

"""

df_clf = df_lim[
    ((df_lim["year"] >= 2010) & (df_lim["year"] <= 2013)) |
    ((df_lim["year"] >= 2018) & (df_lim["year"] <= 2020))
].copy()

df_clf["label"] = df_clf["year"].apply(
    lambda x: 0 if 2010 <= x <= 2013 else 1
)

#TF-IDF VECTORIZATION

"""
Create TF-IDF features using words and two-word phrases.

"""

from sklearn.pipeline import make_pipeline

y = df_clf["label"]

"""
A pipeline: TF - IDF + Logistic Regression

(I) Converts text into TF - IDF features, (II) Trains a logistic regression classifier

"""

pipeline = make_pipeline(
    TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=3), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))

"""
Evaluate the classifier with 5-fold cross-validation.
Training and testing data on different splits, providing a better estimate of how late vs early papers can be distinguished by their languages.

"""

cv_scores = cross_val_score(
    pipeline,
    df_clf["abstract_text"],
    y,
    cv=5,
    scoring="accuracy")

print("\nCross-validated accuracy scores:", cv_scores)
print("Mean CV accuracy:", round(cv_scores.mean(), 3))

# TRAIN/TEST SPLIT for FEATURE ANALYSIS

"""
Split the dataset -- training vs test sets.

"""

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df_clf["abstract_text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y)

"""
Create a TF-IDF vectorizer.
- max_features -- limit vocabulary size to most important terms
- ngram_range --include both single words and two-word phrases
- min_df -- ignore very rare terms (noise reduction)

"""

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=3)

"""
Fit the vectorizer on training data and transform it into TF-IDF features.
"""

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

"""
Train the logistic regression model and predict test labels.
"""
model = LogisticRegression(max_iter = 1000, class_weight = "balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

"""
Evaluate the moderl performance.

- Classification report: precision, recall, and F1 - score.
- Confusion matrix: TP, TN, FP, FN

"""
print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


# EXTRACT TOP FEATURES

"""
Get the most important words and phrases learned.
Positive coeffs --> later papers, negative coeffs --> earlier papers.

"""


feature_names = vectorizer.get_feature_names_out()
coeffs = model.coef_[0]

top_late = sorted(zip(coeffs, feature_names), reverse=True)[:20]
top_early = sorted(zip(coeffs, feature_names))[:20]

print("\nTop EARLY features (2010-2013):")
for coef, word in top_early:
    print(word, round(coef, 3))
    
print("\nTop LATE features (2018-2020):")
for coef, word in top_late:
    print(word, round(coef, 3))

# FEATURE IMPORTANCE - CREATE A BAR CHART

"""
Extract the top 15 most discriminative words for each period.

"""

top_n = 15
late_words = [w for _, w in top_late[:top_n]]
late_coefs = [c for c, _ in top_late[:top_n]]
early_words = [w for _, w in top_early[:top_n]]
early_coefs = [abs(c) for c, _ in top_early[:top_n]]

"""
Create a side-by-side horizontal bar chart to compare the most 
discriminative words for late (2018-2020) and early (2010-2013) periods.

"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.barh(late_words, late_coefs, color="steelblue")
ax1.set_title("Top Late Features (2018–2020)")
ax1.set_xlabel("Coefficient")
ax1.invert_yaxis()

ax2.barh(early_words, early_coefs, color="coral")
ax2.set_title("Top Early Features (2010–2013)")
ax2.set_xlabel("|Coefficient|")
ax2.invert_yaxis()

plt.suptitle("Most Discriminative Features: Early vs Late Periods")
plt.tight_layout()
plt.show()

# OVERLAP ANALYSIS FOR LEXICAL - CLASSIFIER

"""
Check how many of the classifier's top features match the manual lexical vocabulary

"""


top_late_words = {word for _, word in top_late}
top_early_words = {word for _, word in top_early}
lexical_vocab = set(failure_terms + capability_terms + 
                    ethics_terms + hedging_terms)

late_overlap = top_late_words & lexical_vocab
early_overlap = top_early_words & lexical_vocab

print("\nClassifier vs Lexical Vocabulary Overlap:")
print(f"Late features overlapping with lexical vocab: {late_overlap}")
print(f"Early features overlapping with lexical vocab: {early_overlap}")
print(f"Late overlap count: {len(late_overlap)}/20")
print(f"Early overlap count: {len(early_overlap)}/20")


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


# EMBEDDINGS

"""
Load precomputed sentence embeddings for each abstract.

"""

from numpy.linalg import norm

load_embeddings = np.load("/Users/zehra/abstract_embeddings.npy")
print("Loaded embeddings:", load_embeddings.shape)


"""
Filter the dataset to only include abstracts that mention limitations.
This is done to avoid general research trends and focus on limitations.

"""
limitation_mask = ((df["failure_present"] == 1) |
    (df["capability_present"] == 1) |
    (df["hedging_present"] == 1) |
    (df["ethics_present"] == 1))

df_emb = df[limitation_mask].copy()
embeddings_lim = load_embeddings[limitation_mask.values]

print("Filtered limitation-related abstracts:", len(df_emb))
print("Filtered embeddings shape:", embeddings_lim.shape)

"""
Get the unique time periods in sorted order for further analysis.
"""
periods_emb = sorted(df_emb["period"].dropna().unique())

"""
Compute a centroid (average embedding) for each period.

Each centroid represents the overall semantic position
of limitations discourse in that time period.
"""
centroids_emb = {}

for p in periods_emb:
    
    mask_p = df_emb["period"] == p
    centroids_emb[p] = embeddings_lim[mask_p.values].mean(axis = 0)

"""
Calculate semantic shift between consecutive periods.

"""

distances_emb = []
labels_emb = []

for i in range(len(periods_emb) - 1):
    p1 = periods_emb[i]
    p2 = periods_emb[i + 1]

    dist = norm(centroids_emb[p2] - centroids_emb[p1])

    distances_emb.append(dist)
    labels_emb.append(f"{p1} -- {p2}")

print("Semantic distances between consecutive periods (limitations only):")
for label, dist in zip(labels_emb, distances_emb):
    print(label, round(dist, 4))


"""
Plot semantic shift for periods.

"""
plt.figure(figsize=(9, 5))
plt.plot(labels_emb, distances_emb, marker="o")
plt.title("Semantic Shift in Limitations Framing (Embeddings)")
plt.xlabel("Period Transition")
plt.ylabel("Centroid Distance")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

"""
Reduce high-dimensional embeddings to 2D with PCA.

"""

from sklearn.decomposition import PCA

pca = PCA(n_components= 2)
embeddings_2d = pca.fit_transform(embeddings_lim)

"""
PC1 and PC2 --> main directions of variation in the data.

"""
df_emb["pc1"] = embeddings_2d[:,0]
df_emb["pc2"] = embeddings_2d[:, 1]

"""
Compute average PCA coordinates for each period.
"""
trajectory = df_emb.groupby("period")[["pc1", "pc2"]].mean().reset_index()
print(trajectory)


"""
Plot the trajectory of limitations framing over time.

The path shows how the semantic position of abstracts moves across periods.

"""

plt.figure(figsize=(8, 6))
plt.plot(trajectory["pc1"], trajectory["pc2"], marker="o")

for _, row in trajectory.iterrows():
    plt.text(row["pc1"], row["pc2"], str(row["period"]))

plt.title("PCA Trajectory of Limitations Framing Over Time")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()




