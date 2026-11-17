"""Practical conditional Gaussian posterior sampling with missing-only clipping.

The fitted IterativeImputer supplies 10 rounds of BayesianRidge conditional
models. Vectorized conditional predictions retain independent per-patient RNG
streams. This is not a claim of a correct joint clinical posterior.
"""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from .data import FEATURES, PHYSIO
from .utils import stable_seed


class PosteriorImputer:
    def __init__(self, rounds=10, split_seed=2026):
        self.rounds=rounds;self.split_seed=split_seed

    def fit(self,x):
        x=x[FEATURES]
        self.age_=float(x.Age.median())
        mode=x.Gender.mode()
        if not np.isfinite(self.age_) or mode.empty:
            raise ValueError('Training Age/Gender cannot be entirely missing')
        self.gender_=float(mode.iloc[0])
        self.dropped_columns_=[c for c in PHYSIO if x[c].isna().all()]
        self.active_=[c for c in PHYSIO if c not in self.dropped_columns_]
        a=x[self.active_].to_numpy(float)
        self.chain_=None
        self.minimum_=self.maximum_=self.median_=np.empty(0,dtype=float)
        if self.active_:
            self.minimum_=np.nanmin(a,axis=0);self.maximum_=np.nanmax(a,axis=0)
            self.median_=np.nanmedian(a,axis=0)
            self.chain_=IterativeImputer(estimator=BayesianRidge(),max_iter=self.rounds,tol=0,sample_posterior=True,initial_strategy='median',random_state=self.split_seed,skip_complete=False)
            # Fixed 10 rounds are intentional; tol=0 disables early termination.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',category=ConvergenceWarning)
                self.chain_.fit(a)
        self.output_columns_=self.active_+['Age','Gender']+[c+'_missing' for c in FEATURES]
        return self

    def transform(self,x,ids,M=10,role='test',batch=0,posterior=True):
        if len(x)!=len(ids) or M<1:raise ValueError('IDs must align and M >= 1')
        order=np.argsort(np.asarray(ids,dtype=str),kind='stable')
        inverse=np.argsort(order)
        x=x.iloc[order];ids=np.asarray(ids)[order]
        raw=x[self.active_].to_numpy(float);missing=np.isnan(raw)
        indicators=x[FEATURES].isna().to_numpy(float)
        demographics=np.column_stack([x.Age.fillna(self.age_),x.Gender.fillna(self.gender_)])
        outputs=[];seed_rows=[]
        counts={c:{'missing_fills':0,'clipped':0} for c in PHYSIO}
        for draw in range(M):
            seeds=[stable_seed(self.split_seed,role,str(pid),batch,draw) for pid in ids]
            rngs=[np.random.default_rng(s) for s in seeds]
            seed_rows.extend(dict(patient_id=str(pid),role=role,batch=batch,draw=draw,seed=s) for pid,s in zip(ids,seeds))
            a=np.where(missing,self.median_,raw).copy()
            if posterior and self.chain_ is not None:
                for step in self.chain_.imputation_sequence_:
                    j=step.feat_idx;idx=np.flatnonzero(missing[:,j])
                    if len(idx)==0:continue
                    mean,std=step.estimator.predict(a[idx][:,step.neighbor_feat_idx],return_std=True)
                    a[idx,j]=mean+std*np.array([rngs[i].normal() for i in idx])
            # Clip final missing draws only; retain observed out-of-range values.
            for j,c in enumerate(self.active_):
                idx=missing[:,j];v=a[idx,j]
                counts[c]['missing_fills']+=int(idx.sum())
                counts[c]['clipped']+=int(((v<self.minimum_[j])|(v>self.maximum_[j])).sum())
                a[idx,j]=np.clip(v,self.minimum_[j],self.maximum_[j])
            outputs.append(np.column_stack([a,demographics,indicators]))
        for c in PHYSIO:
            n=counts[c]['missing_fills'];counts[c]['clip_fraction']=counts[c]['clipped']/n if n else None
            counts[c]['dropped']=c in self.dropped_columns_
        return np.asarray(outputs)[:,inverse],counts,seed_rows
