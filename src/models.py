import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score


def fit_classifier(draws,y,C=1,model='LR',seed=2026):
    M,n,p=draws.shape
    estimator=LogisticRegression(C=C,max_iter=2000) if model=='LR' else RandomForestClassifier(n_estimators=300,min_samples_leaf=10,max_features='sqrt',class_weight=None,random_state=seed,n_jobs=-1)
    fitted=make_pipeline(StandardScaler(),estimator)
    weights=np.full(M*n,1/M)
    fitted.fit(draws.reshape(M*n,p),np.tile(y,M),**{f'{fitted.steps[-1][0]}__sample_weight':weights})
    fitted.replica_weights_=weights
    return fitted


def predict_draws(model,draws):
    M,n,p=draws.shape
    pm=model.predict_proba(draws.reshape(M*n,p))[:,1].reshape(M,n)
    return dict(p_m=pm,p_mean=pm.mean(axis=0),p_std=pm.std(axis=0,ddof=1) if M>1 else np.zeros(n),flip=(pm>=.5).any(axis=0)&(pm<.5).any(axis=0))


def choose_classifier(train,y_train,selection,y_selection,seed):
    scores=[];models=[]
    for C in [.1,1,10]:
        fitted=fit_classifier(train,y_train,C=C,seed=seed)
        score=average_precision_score(y_selection,predict_draws(fitted,selection)['p_mean'])
        scores.append(dict(C=C,selection_auprc=float(score)));models.append(fitted)
    index=max(range(3),key=lambda i:(scores[i]['selection_auprc'],-scores[i]['C']))
    return models[index],scores[index]['C'],scores
