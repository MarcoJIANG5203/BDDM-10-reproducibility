import math
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .utils import tie_key


def batch_accept(ids, pool, score, capacity, copy_keys=None):
    pool,score=np.asarray(pool,bool),np.asarray(score,float)
    k=min(math.floor(capacity*len(pool)),int(pool.sum()))
    copy_keys=np.arange(len(pool)) if copy_keys is None else copy_keys
    order=sorted(np.flatnonzero(pool),key=lambda i:(score[i],tie_key(ids[i]),int(copy_keys[i])))
    accepted=np.zeros(len(pool),bool);accepted[order[:k]]=True
    return accepted


def fit_threshold(ids, pool, score, capacity):
    selected=batch_accept(ids,pool,score,capacity)
    if selected.sum()==np.sum(pool):
        return dict(threshold=float('inf'),tie_hash='~',selection_acceptance=float(selected.mean()))
    if not selected.any():
        return dict(threshold=-float('inf'),tie_hash='',selection_acceptance=0.)
    last=max(np.flatnonzero(selected),key=lambda i:(score[i],tie_key(ids[i])))
    return dict(threshold=float(score[last]),tie_hash=tie_key(ids[last]),selection_acceptance=float(selected.mean()))


def frozen_accept(ids,pool,score,rule):
    score=np.asarray(score)
    return np.asarray(pool,bool)&((score<rule['threshold'])|((score==rule['threshold'])&np.array([tie_key(i)<=rule['tie_hash'] for i in ids])))


def fit_risk(features, errors):
    errors=np.asarray(errors,int)
    counts=np.bincount(errors,minlength=2)
    if min(counts)<20:
        return None, f'unavailable: {counts[1]} errors, {counts[0]} correct; need >=20 each'
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
    model.fit(features,errors)
    return model,'available'
