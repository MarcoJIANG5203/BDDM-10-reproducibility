import hashlib
import numpy as np
import pandas as pd
from .data import PHYSIO
from .utils import stable_seed


def apply_scenario(x,ids,scenario,seed):
    if scenario not in ['S0','S1','S2','S3','S4']:
        raise ValueError('Unknown scenario')
    out=x.copy(); observed=x[PHYSIO].notna().to_numpy()
    mask=np.zeros(observed.shape,bool)
    if scenario=='S3':mask[:,8:]=observed[:,8:]
    elif scenario in ['S1','S2']:
        rate=.1 if scenario=='S1' else .3
        for i,pid in enumerate(ids):
            mask[i]=observed[i]&(np.random.default_rng(stable_seed(seed,'mask',scenario,str(pid))).random(34)<rate)
    out.loc[:,PHYSIO]=out[PHYSIO].mask(mask)
    stats=dict(original_observed=int(observed.sum()),hidden=int(mask.sum()),hidden_fraction=float(mask.sum()/observed.sum()) if observed.sum() else None,final_missing_fraction=float(out[PHYSIO].isna().to_numpy().mean()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),per_column={c:dict(original_observed=int(observed[:,j].sum()),hidden=int(mask[:,j].sum())) for j,c in enumerate(PHYSIO)})
    return out,pd.DataFrame(mask,index=x.index,columns=PHYSIO),stats
