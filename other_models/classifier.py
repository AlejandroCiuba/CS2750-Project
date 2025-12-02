import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import spacy
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------


BASE_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
BASE_EPOCHS = 3      # for DistilBERT
SENN_EPOCHS = 40     # for SENN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# -------------------------------------------------------------
# 0. Load data (SPC.json)
# -------------------------------------------------------------
df = pd.read_json(f'SPC.json', lines=True)
sentences = []
labels = []
for _, row in df.iterrows():
    sent = f"{row['before']} {row['first']} {row['second']} {row['after']}"
    sent = " ".join(sent.split())
    label = 0 if row["consensus"] == "neither" else 1
    sentences.append(sent)
    labels.append(label)
    consensus_labels = df["consensus"].values  # e.g. "neither", "first", "second", maybe "both"


sentences = np.array(sentences)
labels = np.array(labels, dtype=np.int64)

print(f"Loaded {len(sentences)} sentences.")
print("Class balance:", np.bincount(labels))
# -------------------------------------------------------------
# 1. Train base DistilBERT classifier
# -------------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL_NAME)

class PleonasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    sentences, labels, test_size=0.1, random_state=42, stratify=labels
)

train_ds = PleonasmDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_ds   = PleonasmDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

base_model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, num_labels=2
).to(DEVICE)

optimizer = AdamW(base_model.parameters(), lr=2e-5)
total_steps = len(train_loader) * BASE_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def train_base_model():
    base_model.train()
    for epoch in range(BASE_EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = base_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Base] Epoch {epoch+1}/{BASE_EPOCHS} - loss: {avg_loss:.4f}")
        eval_base_model()

def eval_base_model():
    base_model.eval()
    preds = []
    true  = []
    with torch.no_grad():
        for batch in val_loader:
            labels_b = batch["labels"].numpy()
            true.extend(labels_b)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = base_model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            pred_labels = (probs > 0.5).long().cpu().numpy()
            preds.extend(pred_labels)
    print("Base validation report:")
    print(classification_report(true, preds, digits=3))

print("Training base DistilBERT classifier...")
train_base_model()
print("Done training base model.")

# -------------------------------------------------------------
# 2. spaCy parsing and role extraction
# -------------------------------------------------------------
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

INTENSIFIERS = {
    "very", "extremely", "totally", "utterly", "completely",
    "absolutely", "really", "quite", "so", "highly", "fully"
}
NEGATION_LEMMAS = {"not", "no", "never"}
TIME_LEMMAS = {"yesterday", "today", "tomorrow", "now", "recently", "currently", "tonight"}
DIRECTION_ADVERBS = {
    "back", "forward", "ahead", "around", "away", "together",
    "down", "up", "out", "in", "off", "over", "across"
}

PLEONASTIC_SUBJECT_LEMMAS = {"it", "there"}

def in_root_clause(tok, root):
    """Check if tok is in the clause dominated by root."""
    cur = tok
    while cur.head != cur:
        if cur == root:
            return True
        cur = cur.head
    return False

def root_index(doc):
    for i, t in enumerate(doc):
        if t.dep_ == "ROOT":
            return i
    return 0

def extract_role_token_sets(doc):
    """
    Return:
      - roles: list of sets of token indices for 16 roles
      - core_roles: dict with 'subj','verb','obj','adjunct' sets

    Roles (16):
      0) Subject NP
      1) Object NP
      2) VerbPhrase (Option A: ROOT verb + aux/auxpass/advmod/neg)
      3) AdjunctPP          (prep)
      4) AdjunctAdvP        (advmod)
      5) AdjectivePhrases   (amod, acomp)
      6) NounModifiers      (det, poss, nummod)
      7) AuxVerbs           (aux, auxpass)
      8) RelativeClause     (relcl)
      9) ConjunctionPhrases (conj subtree)
     10) Pronouns
     11) Intensifiers

     12) VerbParticles        (dep_ == "prt" in root clause)
     13) DirectionalAdverbs   (ADV with lemma in DIRECTION_ADVERBS)
     14) PleonasticSubjects   ("it"/"there" as expl/nsubj in root clause)
     15) ClauseFinalAdjunct   (rightmost advmod/prep subtree in root clause)
    """

    # ---- 16 slots instead of 12 ----
    roles = [set() for _ in range(16)]
    ridx = root_index(doc)
    root = doc[ridx]

    def subtree_indices(tok):
        return {t.i for t in tok.subtree}

    subj_tokens    = set()
    obj_tokens     = set()
    adjunct_tokens = set()
    verb_tokens    = set()  # Option A

    # --- Pass 1: gather subjects, objects, adjuncts (root clause) ---
    for t in doc:
        # subjects in root clause
        if t.dep_ in ("nsubj", "csubj", "nsubjpass") and in_root_clause(t, root):
            subj_tokens.update(subtree_indices(t))

        # objects in root clause
        if t.dep_ in ("dobj", "attr", "oprd", "pobj") and in_root_clause(t, root):
            obj_tokens.update(subtree_indices(t))

        # generic adjuncts (prep/advmod) in root clause
        if t.dep_ in ("prep", "advmod") and in_root_clause(t, root):
            adjunct_tokens.update(subtree_indices(t))

    core_roles = {
        "subj":    subj_tokens,
        "verb":    set(),
        "obj":     obj_tokens,
        "adjunct": adjunct_tokens,
    }

    # 0) Subject NP
    roles[0].update(subj_tokens)

    # 1) Object NP
    roles[1].update(obj_tokens)

    # 2) Verb phrase (Option A: ROOT + aux/auxpass/advmod/neg)
    verb_tokens.add(root.i)
    for t in doc:
        if t.head == root and t.dep_ in ("aux", "auxpass", "advmod", "neg"):
            verb_tokens.add(t.i)
    roles[2].update(verb_tokens)
    core_roles["verb"] = verb_tokens

    # 3) Adjunct PP (prep subtrees in root clause)
    for t in doc:
        if t.dep_ == "prep" and in_root_clause(t, root):
            roles[3].update(subtree_indices(t))

    # 4) Adjunct AdvP (advmod subtrees in root clause)
    for t in doc:
        if t.dep_ == "advmod" and in_root_clause(t, root):
            roles[4].update(subtree_indices(t))

    # 5) Adjective phrases (ADJ with amod/acomp)
    for t in doc:
        if t.pos_ == "ADJ" and t.dep_ in ("amod", "acomp"):
            roles[5].update(subtree_indices(t))

    # 6) Noun modifiers (det, nummod, poss)
    for t in doc:
        if t.dep_ in ("det", "nummod", "poss"):
            roles[6].add(t.i)

    # 7) Auxiliary verbs (aux, auxpass) in root clause
    for t in doc:
        if t.dep_ in ("aux", "auxpass") and in_root_clause(t, root):
            roles[7].add(t.i)

    # 8) Relative clause (relcl subtrees)
    for t in doc:
        if t.dep_ == "relcl":
            roles[8].update(subtree_indices(t))

    # 9) Conjunction phrases (conj subtrees)
    for t in doc:
        if t.dep_ == "conj":
            roles[9].update(subtree_indices(t))

    # 10) Pronouns
    for t in doc:
        if t.pos_ == "PRON":
            roles[10].add(t.i)

    # 11) Intensifiers (by lemma)
    for t in doc:
        if t.lemma_.lower() in INTENSIFIERS:
            roles[11].add(t.i)

    # 12) VerbParticles (phrasal verb particles) in root clause
    for t in doc:
        if t.dep_ == "prt" and in_root_clause(t, root):
            roles[12].add(t.i)

    # 13) DirectionalAdverbs (again, back, forward, up, down, etc.)
    for t in doc:
        if t.pos_ == "ADV" and t.lemma_.lower() in DIRECTION_ADVERBS:
            roles[13].add(t.i)

    # 14) PleonasticSubjects ("it"/"there" as expl or nsubj in root clause)
    for t in doc:
        lemma = t.lemma_.lower()
        if in_root_clause(t, root):
            if t.dep_ == "expl":
                roles[14].add(t.i)
            elif lemma in PLEONASTIC_SUBJECT_LEMMAS and t.dep_ in ("nsubj", "expl"):
                roles[14].add(t.i)

    # 15) ClauseFinalAdjunct (rightmost advmod/prep subtree in root clause)
    last_adj_idx = None
    for t in doc:
        if t.dep_ in ("advmod", "prep") and in_root_clause(t, root):
            if last_adj_idx is None or t.i > last_adj_idx:
                last_adj_idx = t.i
    if last_adj_idx is not None:
        roles[15].update(subtree_indices(doc[last_adj_idx]))

    return roles, core_roles


print("Parsing sentences with spaCy...")
docs = list(nlp.pipe([str(s) for s in sentences], batch_size=64))

all_roles = []
for doc in docs:
    role_sets, _ = extract_role_token_sets(doc)
    all_roles.append(role_sets)

K = len(all_roles[0])  # number of roles/concepts
print(f"Using {K} masking-based concepts (roles).")

# -------------------------------------------------------------
# 3. Helper: base model prediction (probabilities) + sentence reps
# -------------------------------------------------------------
def base_predict_proba(texts, batch_size=32):
    """Return np.array of shape (N,) with P(y=1|x) from base_model."""
    base_model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                list(batch_texts),
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            outputs = base_model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_all.append(probs.cpu().numpy())
    return np.concatenate(probs_all, axis=0)

def base_sentence_reps(texts, batch_size=32):
    """Return np.array (N, H) with hidden CLS representations from base_model."""
    base_model.eval()
    reps_all = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(
                list(batch_texts),
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            outputs = base_model.distilbert(**{k: enc[k] for k in ["input_ids","attention_mask"]})
            # CLS token representation: first token of last_hidden_state
            cls_rep = outputs.last_hidden_state[:, 0, :]  # (B, H)
            reps_all.append(cls_rep.cpu().numpy())
    return np.concatenate(reps_all, axis=0)

# -------------------------------------------------------------
# 4. Build masking-based concepts
# -------------------------------------------------------------
def mask_sentence_by_role(doc, idx_set):
    """
    Replace tokens in idx_set by [MASK].
    Return reconstructed string.
    """
    if not idx_set:
        return doc.text
    tokens = []
    for i, t in enumerate(doc):
        if i in idx_set:
            tokens.append("[MASK]")
        else:
            tokens.append(t.text)
    sent = " ".join(tokens)
    return " ".join(sent.split())

print("Computing base probabilities on full sentences...")
s_full = base_predict_proba(sentences)  # shape (N,)

print("Building masked sentences and computing concept values...")
concept_matrix = np.zeros((len(sentences), K), dtype=np.float32)

# For each role k, mask that role in every sentence and compute s_masked
for k in range(K):
    print(f"  Role {k+1}/{K}...")
    masked_texts = []
    for doc, role_sets in zip(docs, all_roles):
        idx_set = role_sets[k]
        masked_texts.append(mask_sentence_by_role(doc, idx_set))
    s_masked = base_predict_proba(masked_texts)
    c_k = s_full - s_masked  # importance of that role
    concept_matrix[:, k] = c_k

print("Concept matrix shape:", concept_matrix.shape)

# Sentence representations for θ-network
print("Computing sentence representations for θ-network...")
sent_reps = base_sentence_reps(sentences)  # (N, H)
print("Sentence reps shape:", sent_reps.shape)

# -------------------------------------------------------------
# 5. SENN model on masking-based concepts
# -------------------------------------------------------------
class SENNDirectConcept(nn.Module):
    """
    Simpler SENN-style model:
      - concepts c(x) are used directly (no concept_net, no h ≠ c)
      - θ(x) is still learned from the sentence representation
      - prediction: f(x) = sum_k c_k(x) * θ_k(x) + b
    """
    def __init__(self, concept_dim, sent_dim,
                 theta_hidden=64,
                 nonneg_theta=True, normalize_theta=False):
        super().__init__()
        self.K = concept_dim
        self.nonneg_theta = nonneg_theta
        self.normalize_theta = normalize_theta

        # only theta_net, no concept_net
        self.theta_net = nn.Sequential(
            nn.Linear(sent_dim, theta_hidden),
            nn.ReLU(),
            nn.Linear(theta_hidden, concept_dim),
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def theta(self, sent_vec):
        th = self.theta_net(sent_vec)      # (B, K)
        if self.nonneg_theta:
            th = F.softplus(th)            # enforce θ_k >= 0
        if self.normalize_theta:
            th = th / (th.sum(dim=1, keepdim=True) + 1e-8)
        return th

    def forward(self, cvec, sent_vec):
        """
        cvec: (B, K) raw concept values (probability differences)
        sent_vec: (B, H) DistilBERT CLS embeddings

        returns:
          logit: (B, 1)
          h: here we just return cvec itself so that the rest of your pipeline
             (which expects (logit, h, theta)) still works
          theta: (B, K)
        """
        th = self.theta(sent_vec)          # (B, K)
        contrib = cvec * th                # (B, K)
        logit = contrib.sum(dim=1, keepdim=True) + self.bias
        # For compatibility with your existing code, we return cvec as "h"
        return logit, cvec, th


def stability_regularizer(cvec, logit, theta):
    grad = torch.autograd.grad(
        outputs=logit.sum(),
        inputs=cvec,
        create_graph=True,
        retain_graph=True,
    )[0]
    return F.mse_loss(grad, theta)

lambda_stab = 1e-3
lambda_l1   = 1e-4

def fit_one_fold_direct(Xc_tr, Xs_tr, y_tr, Xc_te, Xs_te,
                        epochs=40, lr=1e-3, weight_decay=1e-4,
                        theta_hidden=64, seed=123):

    torch.manual_seed(seed)
    np.random.seed(seed)

    concept_dim = Xc_tr.shape[1]
    sent_dim    = Xs_tr.shape[1]

    model = SENNDirectConcept(concept_dim, sent_dim,
                              theta_hidden=theta_hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xc_tr_t = torch.tensor(Xc_tr, dtype=torch.float32).to(DEVICE)
    Xs_tr_t = torch.tensor(Xs_tr, dtype=torch.float32).to(DEVICE)
    y_tr_t  = torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32).to(DEVICE)

    Xc_te_t = torch.tensor(Xc_te, dtype=torch.float32).to(DEVICE)
    Xs_te_t = torch.tensor(Xs_te, dtype=torch.float32).to(DEVICE)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        Xc_tr_t.requires_grad_(True)

        logit, h, theta = model(Xc_tr_t, Xs_tr_t)   # here h == cvec
        bce  = F.binary_cross_entropy_with_logits(logit, y_tr_t)
        stab = stability_regularizer(Xc_tr_t, logit, theta)
        l1   = theta.abs().mean()
        loss = bce + lambda_stab * stab + lambda_l1 * l1

        loss.backward()
        opt.step()

    # predict on this fold's test set
    model.eval()
    with torch.no_grad():
        logit_te, h_te, theta_te = model(Xc_te_t, Xs_te_t)
        prob_te  = torch.sigmoid(logit_te).cpu().numpy().squeeze()
        theta_te = theta_te.cpu().numpy()
        h_te     = h_te.cpu().numpy()   # here h_te == Xc_te

    return prob_te, theta_te, h_te, model

X_c = concept_matrix
X_s = sent_reps
y   = labels
N   = len(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

oof_prob  = np.zeros(N, dtype=np.float32)
oof_theta = np.zeros((N, X_c.shape[1]), dtype=np.float32)
oof_h     = np.zeros((N, X_c.shape[1]), dtype=np.float32)

print("Training DIRECT-CONCEPT SENN with 5-fold cross-validation...")
for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X_c, y), start=1):
    print(f"\n=== Fold {fold_idx} ===")
    Xc_tr, Xc_te = X_c[tr_idx], X_c[te_idx]
    Xs_tr, Xs_te = X_s[tr_idx], X_s[te_idx]
    y_tr,  y_te  = y[tr_idx],  y[te_idx]

    prob_te, theta_te, h_te, _ = fit_one_fold_direct(
        Xc_tr, Xs_tr, y_tr, Xc_te, Xs_te,
        epochs=SENN_EPOCHS, lr=1e-3, weight_decay=1e-4,
        theta_hidden=64,
        seed=123 + fold_idx
    )

    oof_prob[te_idx]  = prob_te
    oof_theta[te_idx] = theta_te
    oof_h[te_idx]     = h_te   # note: this equals Xc_te

print("\n=== OOF Metrics for DIRECT-CONCEPT SENN (5-fold CV) ===")
y_pred_oof = (oof_prob >= 0.5).astype(int)
print(classification_report(y, y_pred_oof, digits=3))
print("Accuracy:", accuracy_score(y, y_pred_oof))
print("ROC-AUC:", roc_auc_score(y, oof_prob))
print("Confusion matrix:")
print(confusion_matrix(y, y_pred_oof))


# -------------------------------------------------------------
# 6. Evaluate SENN on full dataset
# -------------------------------------------------------------


role_names = [
    "SubjectNP",           # 0
    "ObjectNP",            # 1
    "VerbPhrase",          # 2
    "AdjunctPP",           # 3
    "AdjunctAdvP",         # 4
    "AdjectivePhrases",    # 5
    "NounModifiers",       # 6
    "AuxVerbs",            # 7
    "RelativeClause",      # 8
    "ConjunctionPhrases",  # 9
    "Pronouns",            #10
    "Intensifiers",        #11
    "VerbParticles",       #12  <-- NEW
    "DirectionalAdverbs",  #13  <-- NEW
    "PleonasticSubjects",  #14  <-- NEW
    "ClauseFinalAdjunct"   #15  <-- NEW
]

for i, name in enumerate(role_names):
    print(f"{i}: {name}")

# Optionally save SENN
#torch.save(senn.state_dict(), HERE / "senn_masking_concepts.pt")
#print("\nSaved SENN model as senn_masking_concepts.pt")
