"""Protocol orchestration. Test labels enter only the evaluation/export block."""
import itertools
import json
import logging
import platform
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from .data import FEATURES,PHYSIO,load_cohort
from .imputation import PosteriorImputer
from .models import fit_classifier,predict_draws,choose_classifier
from .conformal import calibrate,prediction_sets,singleton_predictions
from .selection import batch_accept,fit_threshold,frozen_accept,fit_risk
from .scenarios import apply_scenario
from .metrics import evaluate,coverage_strata,paired_bootstrap
from .utils import file_hash,stable_seed


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')


def timestamp():
    now=datetime.now(timezone.utc)
    return dict(utc=now.isoformat(),beijing=now.astimezone(timezone(timedelta(hours=8))).isoformat())


def default_config():
    return dict(protocol_version='v1',synthetic=False,features=FEATURES,data_path='data/processed/cohort.parquet',audit_path='data/processed/cohort_audit.json',split_pattern='splits/seed_{seed}.json',seeds=[2026,2027,2028,2029,2030],M=10,rounds=10,alpha=.1,capacities=[.4,.6,.8],bootstrap=1000,scenarios=['S0','S1','S2','S3','S4'],batches=5,output_root='results/clinical')


def create_demo(output,n_a=800,n_b=120):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(1402);n=n_a+n_b
    latent=rng.normal(size=(n,4));coeff=rng.normal(size=(4,34))
    values=latent@coeff+rng.normal(size=(n,34))
    x=pd.DataFrame(values,columns=PHYSIO)
    x=x.mask(rng.random((n,34))<.25)
    x['Age']=rng.uniform(20,90,n);x['Gender']=rng.integers(0,2,n)
    x.loc[rng.random(n)<.05,'Age']=np.nan
    x.loc[rng.random(n)<.05,'Gender']=np.nan
    # Deliberately noisy labels exercise both correctly classified and erroneous patients.
    x['y_true']=rng.binomial(1,.48+.08*np.tanh(latent[:,0]))
    x['site']=['A']*n_a+['B']*n_b
    x['patient_id']=[f'synthetic-{i:05d}' for i in range(n)]
    x.to_csv(output/'cohort.csv',index=False)
    for seed in [2026,2027,2028,2029,2030]:
        a=x.iloc[:n_a]
        train,rest=train_test_split(a.patient_id.tolist(),test_size=.4,stratify=a.y_true,random_state=seed)
        labels=x.set_index('patient_id').y_true
        selection,rest=train_test_split(rest,train_size=.25,stratify=labels.loc[rest],random_state=seed)
        calibration,test=train_test_split(rest,test_size=.5,stratify=labels.loc[rest],random_state=seed)
        write_json(output/f'seed_{seed}.json',dict(train=train,selection=selection,calibration=calibration,test=test,external=x.iloc[n_a:].patient_id.tolist()))
    write_json(output/'cohort_audit.json',dict(protocol_version='v1',synthetic=True,data_version='synthetic-v1-1402',feature_window_end=12,outcome_window='(12,24]',label_shift_hours=6,site_meanings={'A':'synthetic development hospital','B':'synthetic external hospital'},exclusion_counts={'input':n,'included':n,'excluded':0},warning='Synthetic inputs and labels; no original ICU temporal audit performed'))
    cfg=default_config();cfg.update(synthetic=True,data_path=str(output/'cohort.csv'),audit_path=str(output/'cohort_audit.json'),split_pattern=str(output/'seed_{seed}.json'),seeds=[2026],bootstrap=100,output_root='results/synthetic')
    return cfg


def _score_features(pred,missing):
    return np.column_stack([np.minimum(pred['p_mean'],1-pred['p_mean']),missing,pred['p_std']])


def _predict(imputer,model,x,ids,M,role,batch,posterior,logs,context):
    context={**context,'predictor':context.get('predictor','MI')}
    started=time.perf_counter()
    draws,clips,seeds=imputer.transform(x,ids,M=M,role=role,batch=batch,posterior=posterior)
    prediction=predict_draws(model,draws)
    logs['clipping'].append(dict(**context,role=role,batch=batch,columns=clips))
    record_seeds(logs['root'],context,seeds)
    logs['timing'].append(dict(**context,role=role,batch=batch,seconds=time.perf_counter()-started,n=len(x)))
    return prediction


def append_frame(root,name,frame):
    """Stream patient rows: bounded peak memory and partial evidence on failure."""
    path=root/f'{name}.csv'
    frame.to_csv(path,index=False,mode='a',header=not path.exists())


def record_seeds(root,context,rows):
    fixed={k:context.get(k) for k in ['split_seed','model','M','source_M','scenario','alpha']}
    fixed['predictor']=context.get('predictor','MI')
    append_frame(root,'imputation_seeds',pd.DataFrame([{**fixed,**row} for row in rows]))


def run(config,rf=False,m_sensitivity=False,alpha_sensitivity=False):
    cfg=dict(config)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    root=Path(cfg['output_root'])/run_id;root.mkdir(parents=True,exist_ok=False)
    (root/'models').mkdir();(root/'masks').mkdir()
    run_started=time.perf_counter()
    manifest=dict(run_id=run_id,synthetic=cfg.get('synthetic',False),run_status='running',start=timestamp(),python=sys.version,platform=platform.platform(),versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','PyYAML','joblib']},source_hashes={str(p):file_hash(p) for folder in ['src','scripts'] for p in Path(folder).glob('*.py')},warnings=[],supplements=dict(rf=rf,m_sensitivity=m_sensitivity,alpha_sensitivity=alpha_sensitivity))
    manifest['versions']['pyarrow']=version('pyarrow') if find_spec('pyarrow') else 'not-installed'
    (root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    manifest['config_sha256']=file_hash(root/'config.yaml')
    write_json(root/'manifest.json',manifest)
    log_handler=logging.FileHandler(root/'warnings.log',encoding='utf-8');logging.getLogger('py.warnings').addHandler(log_handler);logging.captureWarnings(True)
    tables={k:[] for k in ['predictions','pm','metrics','coverage','bootstrap','status','stability','thresholds','parameters']}
    logs=dict(clipping=[],timing=[],masks=[],root=root)
    try:
        manifest['data_sha256']=file_hash(cfg['data_path']);manifest['audit_sha256']=file_hash(cfg['audit_path'])
        manifest['split_sha256']={str(s):file_hash(cfg['split_pattern'].format(seed=s)) for s in cfg['seeds']}
        for split_seed in cfg['seeds']:
            frame,splits,audit=load_cohort(cfg,split_seed)
            manifest['data_version']=audit['data_version']
            roles={r:frame.loc[ids] for r,ids in splits.items()}
            xt=roles['train'][FEATURES];yt=roles['train'].y_true.to_numpy()
            xs=roles['selection'][FEATURES];ys=roles['selection'].y_true.to_numpy()
            xc=roles['calibration'][FEATURES];yc=roles['calibration'].y_true.to_numpy()
            imputer=PosteriorImputer(cfg['rounds'],split_seed).fit(xt)
            tasks=[('LR',cfg['M'],cfg['scenarios'])]
            if rf:tasks.append(('RF',10,['S0','S2']))
            if m_sensitivity and split_seed==2026:tasks.extend([('LR',m,['S0']) for m in [5,20]])
            for model_name,M,scenarios in tasks:
                context=dict(split_seed=split_seed,model=model_name,M=M,source_M=M)
                train,clip,seeds=imputer.transform(xt,splits['train'],M=M,role='train')
                sel,selclip,selseeds=imputer.transform(xs,splits['selection'],M=M,role='selection')
                logs['clipping'].extend([dict(**context,role='train',batch=0,columns=clip),dict(**context,role='selection',batch=0,columns=selclip)])
                record_seeds(root,context,seeds+selseeds)
                if model_name=='LR':model,C,evidence=choose_classifier(train,yt,sel,ys,split_seed)
                else:model=fit_classifier(train,yt,model='RF',seed=split_seed);C=1;evidence=[]
                median_train,median_clip,median_seeds=imputer.transform(xt,splits['train'],M=1,role='train',posterior=False)
                record_seeds(root,{**context,'M':1,'predictor':'median'},median_seeds)
                logs['clipping'].append(dict(**{**context,'M':1},role='train',batch=0,predictor='median',columns=median_clip))
                median_model=fit_classifier(median_train,yt,C=C,model=model_name,seed=split_seed)
                sp=predict_draws(model,sel)
                cp=_predict(imputer,model,xc,splits['calibration'],M,'calibration',0,True,logs,context)
                mc=_predict(imputer,median_model,xc,splits['calibration'],1,'calibration',0,False,logs,{**context,'M':1,'predictor':'median'})
                sf=_score_features(sp,xs[PHYSIO].isna().mean(axis=1).to_numpy())
                errors=(sp['p_mean']>=.5)!=ys
                risks={};reasons={}
                for method,cols in [('B5',2),('B6',3)]:risks[method],reasons[method]=fit_risk(sf[:,:cols],errors)
                tables['parameters'].append(dict(**context,C=C,C_selection=evidence,classifier_parameters=model.steps[-1][1].get_params(),median_classifier_parameters=median_model.steps[-1][1].get_params(),risk_classifier_parameters={'C':1,'max_iter':2000,'class_weight':None},training_all_missing=imputer.dropped_columns_,rounds=cfg['rounds'],risk_status=reasons,sampling='conditional Gaussian BayesianRidge chain, missing-only final clipping',seed_mapping='sha256 JSON [split_seed,role,patient_id,batch,draw] first 4 bytes big endian',selection_errors=int(errors.sum()),selection_correct=int((~errors).sum())))
                alphas=[cfg['alpha']]+([.05] if alpha_sensitivity and split_seed==2026 and model_name=='LR' and M==10 else [])
                for alpha in alphas:
                    q=calibrate(cp['p_mean'],yc,alpha);q0=calibrate(mc['p_mean'],yc,alpha)
                    selection_sets=prediction_sets(sp['p_mean'],q);selection_pool,_=singleton_predictions(selection_sets)
                    selection_scores={'B2':sf[:,0],'B3':sf[:,1],'B4':sf[:,2]}
                    for method,cols in [('B5',2),('B6',3)]:
                        if risks[method] is not None:selection_scores[method]=risks[method].predict_proba(sf[:,:cols])[:,1]
                    rules={(method,c):fit_threshold(splits['selection'],selection_pool,score,c) for method,score in selection_scores.items() for c in cfg['capacities']}
                    for (method,c),rule in rules.items():tables['thresholds'].append(dict(**context,alpha=alpha,method=method,target_acceptance=c,q=q,**rule,dependency='selection pool uses final calibration q; risk models use all selection patients'))
                    artifact=dict(imputer=imputer,model=model,median_model=median_model,risks=risks,thresholds=rules,q=q,q0=q0,C=C,config=cfg)
                    joblib.dump(artifact,root/'models'/f'seed{split_seed}_{model_name}_M{M}_alpha{alpha}.joblib')
                    for method in ['B0','B1','B2','B3','B4','B5','B6']:
                        tables['status'].append(dict(**{**context,'M':1 if method=='B0' else M},alpha=alpha,method=method,status='available' if method not in reasons or risks[method] is not None else 'unavailable',reason=reasons.get(method,'')))
                    for scenario in scenarios:
                        role='external' if scenario=='S4' else 'test'
                        part=roles[role];ids=part.patient_id.to_numpy()
                        x,mask,maskstat=apply_scenario(part[FEATURES],ids,scenario,split_seed)
                        maskpath=root/'masks'/f'seed{split_seed}_{scenario}.csv'
                        if not maskpath.exists():mask.assign(patient_id=ids).to_csv(maskpath,index=False)
                        logs['masks'].append(dict(split_seed=split_seed,scenario=scenario,**maskstat))
                        missing=x[PHYSIO].isna().mean(axis=1).to_numpy()
                        batch_count=cfg.get('batches',5) if scenario=='S0' and model_name=='LR' and M==10 and alpha==cfg['alpha'] else 1
                        decisions={}
                        for batch in range(batch_count):
                            p=_predict(imputer,model,x,ids,M,role,batch,True,logs,{**context, 'scenario':scenario,'alpha':alpha})
                            p0=_predict(imputer,median_model,x,ids,1,role,batch,False,logs,{**context,'M':1,'scenario':scenario,'alpha':alpha,'predictor':'median'})
                            sets=prediction_sets(p['p_mean'],q);pool,pred=singleton_predictions(sets)
                            sets0=prediction_sets(p0['p_mean'],q0);pool0,pred0=singleton_predictions(sets0)
                            f=_score_features(p,missing)
                            scores={'B2':f[:,0],'B3':f[:,1],'B4':f[:,2]}
                            for method,cols in [('B5',2),('B6',3)]:
                                if risks[method] is not None:scores[method]=risks[method].predict_proba(f[:,:cols])[:,1]
                            base=dict(protocol_version=cfg['protocol_version'],run_id=run_id,synthetic=cfg['synthetic'],**context,scenario=scenario,alpha=alpha,imputation_batch=batch)
                            # From here labels are read for export and evaluation only.
                            y=part.y_true.to_numpy()
                            for predictor,pp in [('MI',p),('median',p0)]:
                                for draw,prob in enumerate(pp['p_m']):
                                    pm_tags={**base,'M':1 if predictor=='median' else M}
                                    append_frame(root,'p_m',pd.DataFrame(dict(**pm_tags,predictor=predictor,patient_id=ids,draw=draw,p_m=prob)))
                            accepted_by={}
                            for method in ['B0','B1']+list(scores):
                                method_sets,method_pool,method_pred,pp=(sets0,pool0,pred0,p0) if method=='B0' else (sets,pool,pred,p)
                                capacities=[None] if method in ['B0','B1'] else cfg['capacities']
                                for capacity in capacities:
                                    modes=['batch'] if capacity is None else ['batch','frozen']
                                    for mode in modes:
                                        accepted=method_pool.copy() if capacity is None else batch_accept(ids,method_pool,scores[method],capacity) if mode=='batch' else frozen_accept(ids,method_pool,scores[method],rules[method,capacity])
                                        key=(method,capacity,mode);accepted_by[key]=accepted
                                        decisions.setdefault(key,[]).append(np.where(accepted,method_pred,-1))
                                        tags={**base,'M':1 if method=='B0' else M,'predictor':'median' if method=='B0' else 'MI','method':method,'target_acceptance':capacity,'selection_mode':mode}
                                        append_frame(root,'patient_predictions',pd.DataFrame(dict(**tags,site=part.site.to_numpy(),patient_id=ids,y_true=y,p_mean=pp['p_mean'],p_std=pp['p_std'],flip=pp['flip'],missing_fraction=missing,set_0=method_sets[:,0],set_1=method_sets[:,1],accepted=accepted,y_pred=method_pred,actual_acceptance=float(accepted.mean()),score=scores.get(method,np.full(len(ids),np.nan)),q=q0 if method=='B0' else q)))
                                        tables['metrics'].append(dict(**tags,**evaluate(y,pp['p_mean'],method_pred,method_sets,accepted),pool_size=int(method_pool.sum()),capacity_shortfall=bool(capacity is not None and method_pool.sum()<np.floor(capacity*len(ids)))))
                            for predictor,ps in [('MI',sets),('median',sets0)]:
                                tables['coverage'].extend(dict(**{**base,'M':1 if predictor=='median' else M},predictor=predictor,**row) for row in coverage_strata(y,ps,missing))
                            if split_seed==2026 and batch==0:
                                for a,b in [('B4','B2'),('B6','B5')]:
                                    if a not in scores or b not in scores:continue
                                    for c in cfg['capacities']:
                                        for mode in ['batch','frozen']:
                                            extra={} if mode=='batch' else dict(fixed_a=accepted_by[a,c,mode],fixed_b=accepted_by[b,c,mode])
                                            ci=paired_bootstrap(ids,y,pred,pool,scores[a],scores[b],c,cfg['bootstrap'],stable_seed(split_seed,scenario,a,b,c,mode),**extra)
                                            aa=accepted_by[a,c,mode];bb=accepted_by[b,c,mode]
                                            diff=float((pred[aa]!=y[aa]).mean()-(pred[bb]!=y[bb]).mean()) if aa.any() and bb.any() else float('nan')
                                            tables['bootstrap'].append(dict(**base,comparison=f'{a}-{b}',target_acceptance=c,selection_mode=mode,risk_difference=diff,**ci))
                        for (method,c,mode),states in decisions.items():
                            for a,b in itertools.combinations(range(len(states)),2):
                                tables['stability'].append(dict(**{**context,'M':1 if method=='B0' else M},alpha=alpha,scenario=scenario,method=method,target_acceptance=c,selection_mode=mode,batch_a=a,batch_b=b,disagreement=float(np.mean(states[a]!=states[b]))))
        filenames=dict(predictions='patient_predictions',pm='p_m',metrics='metrics',coverage='coverage_strata',bootstrap='bootstrap_differences',status='method_status',stability='stability',thresholds='thresholds',parameters='model_parameters')
        for key,filename in filenames.items():
            if key in ['predictions','pm']:continue
            elif key=='parameters':write_json(root/f'{filename}.json',tables[key])
            else:pd.DataFrame(tables[key]).to_csv(root/f'{filename}.csv',index=False)
        for key,rows in logs.items():
            if key!='root':write_json(root/f'{key}.json',rows)
        manifest['run_status']='completed'
    except BaseException as exc:
        manifest['run_status']='failed';manifest['error']=str(exc)
        (root/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        manifest['end']=timestamp()
        manifest['elapsed_seconds']=time.perf_counter()-run_started
        manifest['artifact_hashes']={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
        write_json(root/'manifest.json',manifest)
        logging.getLogger('py.warnings').removeHandler(log_handler);log_handler.close()
    return root
