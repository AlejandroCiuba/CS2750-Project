#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 21:04:58 2025

@author: raul
"""

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

vec = model.wv["always"]

similarity = model.wv.similarity("always", "forever")

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
# =========================
# 0) Imports & data
# =========================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report

# ---- Your features/labels ----
# cross_matrix: shape (N, 6) with [SxV, SxO, SxA, VxO, VxA, OxA]
# y: list/array of 0/1 labels aligned with rows of cross_matrix
X_np = np.asarray(X, dtype=np.float32)   # <--- use your computed 6-dim features here
y_np = np.asarray(y, dtype=np.int32)

# =========================
# 1) Model definition
# =========================
class SENNLogit(nn.Module):
    """
    f(x) = sum_k theta_k(x) * x_k + b, with theta(x) = MLP(x)
    p(y=1|x) = sigmoid(f(x))
    """
    def __init__(self, K=6, hidden=16, nonneg=True, normalize_theta=False):
        super().__init__()
        self.K = K
        self.theta_net = nn.Sequential(
            nn.Linear(K, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K)
        )
        self.b = nn.Parameter(torch.zeros(1))
        self.nonneg = nonneg
        self.normalize_theta = normalize_theta

    def theta(self, z):
        th = self.theta_net(z)            # (B, K)
        if self.nonneg:
            th = F.softplus(th)           # ensure θ >= 0 (optional but interpretable)
        if self.normalize_theta:
            th = th / (th.sum(dim=1, keepdim=True) + 1e-8)  # attention-like normalization
        return th

    def forward(self, z):
        th = self.theta(z)                # (B, K)
        logit = (th * z).sum(dim=1, keepdim=True) + self.b
        return logit, th                  # return both: prediction + explanations

def stability_regularizer(z, logit, theta):
    # Encourage grad wrt inputs to match theta(x) (SENN faithfulness)
    grad = torch.autograd.grad(outputs=logit.sum(), inputs=z,
                               create_graph=True, retain_graph=True)[0]  # (B, K)
    return F.mse_loss(grad, theta)

# =========================
# 2) One-fold train/eval helper
# =========================
def fit_one_fold(X_tr, y_tr, X_te, epochs=200, lr=1e-3, weight_decay=1e-4,
                 lambda_stab=1e-2, lambda_l1=1e-3, hidden=200, seed=42):
    torch.manual_seed(seed)

    model = SENNLogit(K=X_tr.shape[1], hidden=hidden, nonneg=True, normalize_theta=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr.reshape(-1, 1), dtype=torch.float32)
    Xe = torch.tensor(X_te, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        Xt.requires_grad_(True)
        logit, theta = model(Xt)
        bce  = F.binary_cross_entropy_with_logits(logit, yt)
        stab = stability_regularizer(Xt, logit, theta)
        l1   = theta.abs().mean()
        loss = bce + lambda_stab * stab + lambda_l1 * l1
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logit_te, theta_te = model(Xe)
        prob_te = torch.sigmoid(logit_te).squeeze().cpu().numpy()  # OOF probabilities
        theta_te = theta_te.cpu().numpy()                          # OOF per-sample θ
    return prob_te, theta_te, model

# =========================
# 3) 5-fold cross validation (OOF predictions + metrics)
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob  = np.zeros(len(X_np), dtype=np.float32)
oof_theta = np.zeros((len(X_np), X_np.shape[1]), dtype=np.float32)

for fold_idx, (tr, te) in enumerate(cv.split(X_np, y_np), start=1):
    p_te, th_te, _ = fit_one_fold(X_np[tr], y_np[tr], X_np[te],
                                  epochs=200, lr=1e-3, weight_decay=1e-4,
                                  lambda_stab=1e-2, lambda_l1=1e-3,
                                  hidden=16, seed=42+fold_idx)
    oof_prob[te]  = p_te
    oof_theta[te] = th_te
    print(f"Fold {fold_idx} done.")

# Metrics on OOF preds (fair estimate)
y_pred = (oof_prob >= 0.5).astype(int)
print("\n=== OOF Metrics (5-fold) ===")
print("Accuracy:",  accuracy_score(y_np, y_pred))
print("ROC-AUC: ",  roc_auc_score(y_np, oof_prob))
print("F1:      ",  f1_score(y_np, y_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_np, y_pred))
print("\nClassification report:")
print(classification_report(y_np, y_pred, digits=3))

# (Optional) inspect explanations for first few samples
print("\nSample θ(x) rows (first 3):")
print(oof_theta[:3])

# =========================
# 4) Fit final model on ALL data (for deployment)
# =========================
_, _, final_model = fit_one_fold(X_np, y_np, X_np, epochs=200)  # trains on all; ignore returned OOF
with torch.no_grad():
    final_logit, final_theta = final_model(torch.tensor(X_np, dtype=torch.float32))
    final_prob = torch.sigmoid(final_logit).squeeze().cpu().numpy()

# Save if you want (PyTorch way):
# torch.save(final_model.state_dict(), "sennlogit_final.pt")
