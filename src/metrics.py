import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from .selection import batch_accept


def ratio(a,b):return float(a/b) if b else float('nan')


def evaluate(y,p,pred,sets,accepted):
    y=np.asarray(y,int);p=np.asarray(p);pred=np.asarray(pred);sets=np.asarray(sets,bool);a=np.asarray(accepted,bool)
    positives=y==1;neg=a&(pred==0);errors=a&(pred!=y)
    return dict(n=len(y),positives=int(positives.sum()),base_rate=float(y.mean()),accepted=int(a.sum()),accepted_errors=int(errors.sum()),accepted_negative=int(neg.sum()),accepted_false_negative=int((neg&positives).sum()),acceptance=float(a.mean()),selective_risk=ratio(errors.sum(),a.sum()),dangerous_miss=ratio((neg&positives).sum(),positives.sum()),negative_residual=ratio((neg&positives).sum(),neg.sum()),coverage=float(sets[np.arange(len(y)),y].mean()),auroc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),auprc=float(average_precision_score(y,p)) if positives.any() else float('nan'))


def coverage_strata(y,sets,missing):
    y=np.asarray(y,int);sets=np.asarray(sets,bool);missing=np.asarray(missing)
    hit=sets[np.arange(len(y)),y]
    groups={'all':np.ones(len(y),bool),'y=0':y==0,'y=1':y==1,'missing_0_1/3':missing<=1/3,'missing_1/3_2/3':(missing>1/3)&(missing<=2/3),'missing_2/3_1':missing>2/3}
    return [dict(stratum=k,n=int(v.sum()),covered=int(hit[v].sum()),coverage=ratio(hit[v].sum(),v.sum())) for k,v in groups.items()]


def paired_bootstrap(ids,y,pred,pool,score_a,score_b,capacity,repetitions=1000,seed=2026,fixed_a=None,fixed_b=None):
    ids=np.asarray(ids);y=np.asarray(y);pred=np.asarray(pred);pool=np.asarray(pool)
    score_a=np.asarray(score_a);score_b=np.asarray(score_b)
    rng=np.random.default_rng(seed);diff=[];n=len(y)
    for _ in range(repetitions):
        ix=rng.integers(0,n,n)
        if fixed_a is None:
            a=batch_accept(ids[ix],pool[ix],score_a[ix],capacity,copy_keys=np.arange(n))
            b=batch_accept(ids[ix],pool[ix],score_b[ix],capacity,copy_keys=np.arange(n))
        else:a=np.asarray(fixed_a)[ix];b=np.asarray(fixed_b)[ix]
        if not a.any() or not b.any():continue
        err=pred[ix]!=y[ix];diff.append(float(err[a].mean()-err[b].mean()))
    return dict(ci_low=float(np.quantile(diff,.025)) if diff else float('nan'),ci_high=float(np.quantile(diff,.975)) if diff else float('nan'),valid=len(diff),invalid=repetitions-len(diff),resample_size=n,capacity_recomputed=fixed_a is None)
