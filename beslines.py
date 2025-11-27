import json, numpy as np
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2


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
    


tfidf_tri = TfidfVectorizer(
    analyzer="word",
    ngram_range=(3, 3),
    lowercase=True,
    min_df=2,          
    max_df=0.95,       
    sublinear_tf=True  
)
    
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
selector = SelectPercentile(score_func=chi2, percentile=90)



nb_pipe = Pipeline([
    ("tfidf", tfidf_tri),
    ("chi2", selector),
    ("clf", MultinomialNB(alpha=0.1))  
])

nb_scores = cross_val_score(nb_pipe, x, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Naive Bayes (TF-IDF trigrams) 5-fold accuracy: {nb_scores.mean():.4f} ± {nb_scores.std():.4f}")






maxent_pipe = Pipeline([
    ("tfidf", tfidf_tri),
    ("chi2", selector),
    ("clf", LogisticRegression(
        penalty="l2",           
        C=1.0,                 
        solver="liblinear",     
        max_iter=500,
        n_jobs=-1 if hasattr(LogisticRegression, "n_jobs") else None
    ))
])


maxent_scores = cross_val_score(maxent_pipe, x, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"MaxEnt (LogReg, TF-IDF trigrams) 5-fold accuracy: {maxent_scores.mean():.4f} ± {maxent_scores.std():.4f}")





