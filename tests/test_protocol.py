from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import pytest


def module(name):
    assert importlib.util.find_spec('src.' + name) is not None, 'Required behavior module missing: ' + name
    return importlib.import_module('src.' + name)


def test_conformal_finite_sample_and_sets():
    c = module('conformal')
    assert np.isinf(c.calibrate([.1,.2],[0,1],.1))
    assert c.calibrate([.1,.2,.8,.9],[0,0,1,1],.4) == pytest.approx(.2)
    assert c.prediction_sets([.1,.5,.9],.2).tolist() == [[True,False],[False,False],[False,True]]
    assert c.prediction_sets([.5],.6).tolist() == [[True,True]]


def test_capacity_ties_frozen_and_labels_absent():
    s = module('selection')
    ids = np.array(['a','b','c','d','e'])
    pool = np.array([1,1,0,1,1],bool)
    score = np.zeros(5)
    a = s.batch_accept(ids,pool,score,.6)
    assert a.sum() == 3
    order = [4,1,3,0,2]
    b = s.batch_accept(ids[order],pool[order],score[order],.6)
    assert dict(zip(ids,a)) == dict(zip(ids[order],b))
    assert s.batch_accept(ids,pool,score,1).sum() == 4
    rule = s.fit_threshold(ids,pool,score,1)
    assert np.isinf(rule['threshold'])
    assert s.frozen_accept(ids,pool,score,rule).sum() == 4
    assert s.fit_risk(np.zeros((39,3)),np.r_[np.ones(19),np.zeros(20)])[0] is None


def test_metrics_denominators_and_bootstrap():
    m = module('metrics')
    y=np.array([1,1,0,0]); pred=np.array([0,1,1,0]); acc=np.array([1,0,0,1],bool)
    sets=np.array([[1,0],[1,1],[1,1],[1,0]],bool)
    result=m.evaluate(y,[.2,.7,.6,.1],pred,sets,acc)
    assert result['coverage']==.75
    assert result['dangerous_miss']==.5
    assert result['negative_residual']==.5
    assert np.isnan(m.evaluate([0],[.1],[0],[[1,0]],[False])['selective_risk'])
    b=m.paired_bootstrap(['a','b','c','d'],y,pred,np.ones(4,bool),[0,1,2,3],[3,2,1,0],.5,40,2)
    assert b['valid']+b['invalid']==40
    assert b['resample_size']==4 and b['capacity_recomputed']


def test_imputation_observed_reproducible_and_train_missing():
    d=module('data'); imp=module('imputation')
    rng=np.random.default_rng(2)
    x=pd.DataFrame(rng.normal(size=(25,36)),columns=d.FEATURES)
    x['Gender']=np.arange(25)%2; x['Age']=50
    x.iloc[::2,0]=np.nan; x.iloc[:,1]=np.nan
    im=imp.PosteriorImputer(10,2026).fit(x)
    z,diag,seeds=im.transform(x,['p'+str(i) for i in range(25)],M=3,role='test',batch=0)
    assert z.shape==(3,25,71)
    assert d.FEATURES[1] in im.dropped_columns_
    assert np.all(z[:,:,35+1]==1)
    assert np.all(z[:,1,0]==x.iloc[1,0])
    rev,_,_=im.transform(x.iloc[::-1],['p'+str(i) for i in range(24,-1,-1)],M=3,role='test',batch=0)
    assert np.array_equal(z,rev[:,::-1])
    again,_,_=im.transform(x,['p'+str(i) for i in range(25)],M=3,role='test',batch=1)
    assert not np.array_equal(z,again)
    out=x.iloc[:1].copy();out.iloc[0,0]=9999
    observed,_,_=im.transform(out,['outside'],M=2,role='test')
    assert np.all(observed[:,:,0]==9999)
    assert set(diag)==set(d.PHYSIO)
    with pytest.raises(ValueError):
        imp.PosteriorImputer().fit(x.assign(Age=np.nan))


def test_scenarios_only_observed_physiology():
    d=module('data');s=module('scenarios')
    x=pd.DataFrame(np.ones((100,36)),columns=d.FEATURES);x.iloc[0,0]=np.nan
    ids=['p'+str(i) for i in range(100)]
    z,mask,stats=s.apply_scenario(x,ids,'S3',2026)
    assert z[d.PHYSIO[8:]].isna().all().all()
    assert not z[['Age','Gender']].isna().any().any()
    assert mask.sum().sum()==2600
    a,ma,_=s.apply_scenario(x,ids,'S2',2026)
    b,mb,_=s.apply_scenario(x.iloc[::-1],ids[::-1],'S2',2026)
    assert np.array_equal(ma.to_numpy(),mb.to_numpy()[::-1])
    assert not ma.iloc[0,0]


def test_data_rejects_split_and_schema_leaks():
    d=module('data')
    x=pd.DataFrame(np.ones((10,36)),columns=d.FEATURES)
    x['patient_id']=[str(i) for i in range(10)];x['site']=['A']*8+['B']*2;x['y_true']=np.arange(10)%2
    splits=dict(train=['0','1'],selection=['2','3'],calibration=['4','5'],test=['6','7'],external=['8','9'])
    d.validate_cohort(x,splits,d.FEATURES)
    with pytest.raises(ValueError):d.validate_cohort(x,{**splits,'test':['5','7']},d.FEATURES)
    with pytest.raises(ValueError):d.validate_cohort(x,splits,d.FEATURES+['y_true'])
    with pytest.raises(ValueError):d.validate_cohort(x.assign(patient_id='same'),splits,d.FEATURES)


def test_shared_model_weights_and_probability_summary():
    model=module('models')
    draws=np.array([[[0.],[1.],[2.],[3.]],[[.1],[.9],[2.1],[2.9]]])
    fitted=model.fit_classifier(draws,np.array([0,0,1,1]),C=1)
    assert np.all(fitted.replica_weights_==.5)
    p=model.predict_draws(fitted,draws)
    assert np.allclose(p['p_std'],np.std(p['p_m'],axis=0,ddof=1))


def test_end_to_end_synthetic_artifacts(tmp_path):
    exp=module('experiment')
    cfg=exp.create_demo(tmp_path/'synthetic',n_a=800,n_b=100)
    cfg.update(bootstrap=5,seeds=[2026],scenarios=['S0','S1','S2','S3','S4'],output_root=str(tmp_path/'results'),batches=1)
    result=exp.run(cfg)
    import json
    assert json.loads((result/'manifest.json').read_text())['run_status']=='completed'
    preds=pd.read_csv(result/'patient_predictions.csv')
    assert set(preds.scenario)=={'S0','S1','S2','S3','S4'}
    status=pd.read_csv(result/'method_status.csv')
    assert set(status.method)=={'B0','B1','B2','B3','B4','B5','B6'}
    assert (status.loc[status.method.isin(['B5','B6']),'status']=='available').all()
    assert set(preds.selection_mode)=={'batch','frozen'}
    assert (preds.loc[preds.method=='B0','M']==1).all()
    assert (result/'p_m.csv').exists() and (result/'models').exists()
    pm=pd.read_csv(result/'p_m.csv')
    assert (pm.loc[pm.predictor=='median','M']==1).all()


def test_posterior_fit_and_median_initialization_are_explicit():
    from src.data import FEATURES
    from src.imputation import PosteriorImputer
    x=pd.DataFrame(np.random.default_rng(1).normal(size=(30,36)),columns=FEATURES)
    x['Gender']=np.arange(30)%2;x.iloc[::3,0]=np.nan
    im=PosteriorImputer().fit(x)
    assert im.chain_.sample_posterior is True
    assert im.chain_.initial_strategy=='median'
    assert len(im.chain_.imputation_sequence_)==340


def test_loader_rejects_blank_patient_id_before_string_conversion(tmp_path):
    from src.experiment import create_demo
    from src.data import load_cohort
    cfg=create_demo(tmp_path,n_a=100,n_b=20)
    x=pd.read_csv(cfg['data_path']);x.loc[0,'patient_id']=np.nan
    x.to_csv(cfg['data_path'],index=False)
    with pytest.raises(ValueError,match='missing'):
        load_cohort(cfg,2026)


def test_test_label_mutation_cannot_change_predictions_or_acceptance(tmp_path):
    from src.experiment import create_demo,run
    import json
    cfg=create_demo(tmp_path/'data',n_a=800,n_b=40)
    cfg.update(bootstrap=2,scenarios=['S0'],batches=1,output_root=str(tmp_path/'results'))
    first=run(cfg)
    frame=pd.read_csv(cfg['data_path'])
    split=json.loads(Path(cfg['split_pattern'].format(seed=2026)).read_text())
    selected=frame.patient_id.isin(split['test']+split['external'])
    frame.loc[selected,'y_true']=1-frame.loc[selected,'y_true']
    frame.to_csv(cfg['data_path'],index=False)
    second=run(cfg)
    a=pd.read_csv(first/'patient_predictions.csv');b=pd.read_csv(second/'patient_predictions.csv')
    for column in ['p_mean','p_std','set_0','set_1','accepted','score']:
        assert np.allclose(a[column],b[column],equal_nan=True)
    assert not np.array_equal(a.y_true,b.y_true)


def test_failed_run_keeps_manifest_and_streamed_predictions(tmp_path):
    from src.experiment import create_demo,run
    import json
    cfg=create_demo(tmp_path/'data',n_a=100,n_b=20)
    cfg.update(bootstrap=1,scenarios=['S0','BAD'],batches=1,output_root=str(tmp_path/'results'))
    with pytest.raises(ValueError,match='scenario'):run(cfg)
    root=next((tmp_path/'results').iterdir())
    assert json.loads((root/'manifest.json').read_text())['run_status']=='failed'
    assert len(pd.read_csv(root/'patient_predictions.csv'))>0


def test_parquet_loader_roundtrip(tmp_path):
    from src.experiment import create_demo
    from src.data import load_cohort
    cfg=create_demo(tmp_path/'data',n_a=100,n_b=20)
    frame=pd.read_csv(cfg['data_path']);frame.to_parquet(tmp_path/'cohort.parquet',index=False)
    cfg['data_path']=str(tmp_path/'cohort.parquet')
    loaded,splits,audit=load_cohort(cfg,2026)
    assert len(loaded)==120 and loaded.patient_id.is_unique
    assert loaded.y_true.sum()==frame.y_true.sum()


def test_non_singleton_prediction_is_explicitly_undefined():
    from src.conformal import singleton_predictions
    pool,pred=singleton_predictions([[0,0],[1,1],[1,0],[0,1]])
    assert pool.tolist()==[False,False,True,True]
    assert pred.tolist()==[-1,-1,0,1]


def test_audit_inclusion_count_matches_loaded_cohort(tmp_path):
    import json
    from src.experiment import create_demo
    from src.data import load_cohort
    cfg=create_demo(tmp_path/'data',n_a=100,n_b=20)
    path=Path(cfg['audit_path']);audit=json.loads(path.read_text())
    audit['exclusion_counts']['included']=1;path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='counts'):load_cohort(cfg,2026)


def test_all_physiology_missing_train_keeps_demographics_and_original_indicators():
    from src.data import FEATURES,PHYSIO
    from src.imputation import PosteriorImputer
    train=pd.DataFrame(np.nan,index=range(3),columns=FEATURES)
    train['Age']=[20.,40.,np.nan];train['Gender']=[1.,0.,1.]
    im=PosteriorImputer().fit(train)
    assert im.active_==[] and im.dropped_columns_==PHYSIO
    assert im.chain_ is None
    assert im.minimum_.shape==im.maximum_.shape==im.median_.shape==(0,)
    test=train.iloc[:2].copy()
    test['Age']=[67.,np.nan];test['Gender']=[0.,np.nan]
    test.loc[0,'HR']=99.  # A later observed value cannot reintroduce a dropped column.
    expected=np.r_[67.,0.,0.,np.ones(33),0.,0.]
    for role in ['train','selection','calibration','test','external']:
        z,diag,_=im.transform(test,['first','second'],M=3,role=role)
        assert z.shape==(3,2,38)
        assert im.output_columns_==['Age','Gender']+[c+'_missing' for c in FEATURES]
        assert np.array_equal(z[:,0,:],np.tile(expected,(3,1)))
        assert np.array_equal(z[:,1,:2],np.tile([30.,1.],(3,1)))
        assert np.all(z[:,1,2:]==1)
        assert all(d['dropped'] and d['missing_fills']==0 and d['clip_fraction'] is None for d in diag.values())
