import json, numpy as np
from pathlib import Path
import pandas as pd 
import gensim

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec
import spacy
from spacy.tokens import Token, Span, Doc

nlp = spacy.load("en_core_web_sm")


HERE = Path(__file__).resolve().parent     
DATA_PATH = HERE / "SPC.json"               
df = pd.read_json(DATA_PATH, lines=True)
x=[]
y=[]
for idx, row in df.iterrows():
    sentence = f"{row['before']} {row['first']} {row['second']} {row['after']}"
    sentence = " ".join(sentence.split())  
    if row['consensus']=='neither':
        i=0
    else:
        i=1
    x.append(sentence)
    y.append(i)
    


tokenized_sentences = []
for sentence in x:
    # lowercasing
    tokens = word_tokenize(sentence.lower())
    # remove punctuation tokens
    tokens = [t for t in tokens if t not in string.punctuation]
    tokenized_sentences.append(tokens)



# Suppose you already have your tokenized sentences
# Example:
# tokenized_sentences = [["i", "love", "python"], ["word2vec", "is", "cool"]]

model= Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,   # dimension of word vectors
    window=5,          # context window size
    min_count=1,       # include all words
    sg=1,              # 1 = skip-gram; 0 = CBOW
    epochs=20
)


def word2vec(tokens):
    return np.vstack([model.wv[w] for w in tokens if w in model.wv])

sentence_matrices = [word2vec(tokens) for tokens in tokenized_sentences]

def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def extract_parts(sentence):
    """Extract subject, verb, object, adjunct tokens (HEAD words, your 'manner')."""
    doc = nlp(sentence)
    subjects = [t.text for t in doc if t.dep_ in ("nsubj", "csubj", "nsubjpass")]
    verbs    = [t.text for t in doc if t.dep_ == "ROOT"]
    objects  = [t.text for t in doc if t.dep_ in ("dobj", "attr", "oprd", "pobj")]
    adjuncts = [t.text for t in doc if t.dep_ in ("prep", "advmod")]
    return subjects, verbs, objects, adjuncts

sentence_vectors = []   # list of dicts aligned with x (your sentences)

for sentence in x:
    subj, verb, obj, adj = extract_parts(sentence)
    sent_dict = {
        "subject": subj,
        "verb":    verb,
        "object":  obj,
        "adjunct": adj,
    }
    sentence_vectors.append(sent_dict)


import numpy as np
from itertools import combinations

# --- Helper: get word vector if in vocab (lowercased) ---
def get_vec(word, model):
    w = word.lower()
    if w in model.wv.key_to_index:
        return model.wv[w]
    return None

# --- Compute average pairwise cosine for a list of tokens ---
# --- Cross-class average cosine between two token sets ---
def cross_avg_cosine(tokens_a, tokens_b, model) -> float:
    va = [get_vec(t, model) for t in tokens_a]
    vb = [get_vec(t, model) for t in tokens_b]
    va = [v for v in va if v is not None]
    vb = [v for v in vb if v is not None]
    if len(va) == 0 or len(vb) == 0:
        return 0.0

    def unit(v):
        n = np.linalg.norm(v)
        return v / (n if n != 0 else 1.0)

    ua = [unit(v) for v in va]
    ub = [unit(v) for v in vb]

    vals = []
    for a in ua:
        for b in ub:
            vals.append(float(a @ b))  # cosine between a and b
    return float(np.mean(vals))



# --- For each sentence, compute the 4-entry similarity vector ---
# Uses your earlier `extract_parts(sentence)` returning: (subjects, verbs, objects, adjuncts)
similarity_rows = []   # list of dicts per sentence
similarity_matrix = [] # optional: Nx4 numpy array
# --- For each sentence, compute the 6-entry cross-class similarity vector ---
# Uses your earlier `extract_parts(sentence)` returning: (subjects, verbs, objects, adjuncts)
similarity_rows = []   # list of dicts per sentence
similarity_matrix = [] # Nx6 numpy-ready list

for sentence in x:
    subj_tokens, verb_tokens, obj_tokens, adj_tokens = extract_parts(sentence)

    s_v = cross_avg_cosine(subj_tokens, verb_tokens, model)
    s_o = cross_avg_cosine(subj_tokens, obj_tokens,  model)
    s_a = cross_avg_cosine(subj_tokens, adj_tokens,  model)
    v_o = cross_avg_cosine(verb_tokens, obj_tokens,  model)
    v_a = cross_avg_cosine(verb_tokens, adj_tokens,  model)
    o_a = cross_avg_cosine(obj_tokens,  adj_tokens,  model)

    row = {
        "SxV": s_v,
        "SxO": s_o,
        "SxA": s_a,
        "VxO": v_o,
        "VxA": v_a,
        "OxA": o_a
    }
    similarity_rows.append(row)
    similarity_matrix.append([s_v, s_o, s_a, v_o, v_a, o_a])


similarity_matrix = np.array(similarity_matrix, dtype=np.float32)  # shape (num_sentences, 4)

# Optional: DataFrame for easy viewing/saving
df_similarity = pd.DataFrame(similarity_rows)
X = np.asarray(similarity_matrix, dtype=np.float32)
y = np.asarray(y, dtype=np.int32)

# 5-fold stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Pipeline (scaling helps even though features are in [0,1], but it's quick & safe)
pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("logit", LogisticRegression(
        penalty="l2",
        solver="liblinear",   # robust for small feature sets
        max_iter=1000,
        class_weight=None     # set to 'balanced' if classes are imbalanced
    ))
])

# Cross-validated metrics
acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
roc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
f1_scores  = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

print(f"Accuracy  (5-fold): {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
print(f"ROC-AUC   (5-fold): {roc_scores.mean():.3f} ± {roc_scores.std():.3f}")
print(f"F1-score  (5-fold): {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

# Get cross-validated predictions for inspection/confusion matrix
y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

print("\nConfusion matrix (CV):")
print(confusion_matrix(y, y_pred))
print("\nClassification report (CV):")
print(classification_report(y, y_pred, digits=3))
