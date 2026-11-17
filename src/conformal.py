import math
import numpy as np


def calibrate(p, y, alpha=.1):
    p,y=np.asarray(p),np.asarray(y)
    if not 0<alpha<1 or len(y)==0:
        raise ValueError('Nonempty calibration and alpha in (0,1) required')
    scores=np.where(y==1,1-p,p)
    k=math.ceil((len(y)+1)*(1-alpha))
    return float('inf') if k>len(y) else float(np.partition(scores,k-1)[k-1])


def prediction_sets(p, q):
    p=np.asarray(p)
    return np.column_stack([p<=q,1-p<=q])


def singleton_predictions(sets):
    sets=np.asarray(sets,bool)
    pool=sets.sum(axis=1)==1
    return pool,np.where(pool,sets[:,1].astype(int),-1)
