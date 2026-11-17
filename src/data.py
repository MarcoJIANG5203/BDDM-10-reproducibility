"""Read-only A->B contract. Does not audit original hourly records or labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PHYSIO = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets']
FEATURES = PHYSIO + ['Age','Gender']
ROLES = ['train','selection','calibration','test','external']


def validate_cohort(frame, splits, features):
    if list(features) != FEATURES:
        raise ValueError('Features must match the ordered official whitelist of 36 inputs')
    required = FEATURES + ['patient_id','site','y_true']
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing required cohort columns')
    if frame.patient_id.isna().any() or frame.patient_id.astype(str).duplicated().any():
        raise ValueError('patient_id must be globally unique and nonmissing')
    if not frame.y_true.isin([0,1]).all():
        raise ValueError('y_true must be binary')
    numeric = frame[FEATURES].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if np.isinf(numeric).any():
        raise ValueError('Infinite features are unsupported')
    if not frame.Gender.dropna().isin([0,1]).all():
        raise ValueError('Gender must be binary or missing')
    if set(splits) != set(ROLES) or any(not splits[r] for r in ROLES):
        raise ValueError('All five fixed split roles must exist and be nonempty')
    ids = [str(i) for r in ROLES for i in splits[r]]
    if len(ids) != len(set(ids)) or set(ids) != set(frame.patient_id.astype(str)):
        raise ValueError('Split overlap, duplicate, omission or unknown patient')
    indexed = frame.assign(patient_id=frame.patient_id.astype(str)).set_index('patient_id')
    for role in ROLES:
        part = indexed.loc[list(map(str,splits[role]))]
        if not (part.site == ('B' if role=='external' else 'A')).all():
            raise ValueError('Hospital identity does not match split role')
        if role in ['train','selection'] and part.y_true.nunique()!=2:
            raise ValueError('train/selection require both classes')


def load_cohort(config, seed):
    path = Path(config['data_path'])
    try:
        frame = pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path, dtype={'patient_id':str})
    except ImportError as exc:
        raise ImportError('Parquet input requires pyarrow; install requirements or use CSV for demo') from exc
    if 'patient_id' not in frame or frame.patient_id.isna().any():
        raise ValueError('patient_id column is absent or contains missing IDs')
    frame['patient_id'] = frame.patient_id.astype(str)
    split_path = Path(config['split_pattern'].format(seed=seed))
    splits = json.loads(split_path.read_text(encoding='utf-8'))
    splits = {role:list(map(str,ids)) for role,ids in splits.items()}
    audit = json.loads(Path(config['audit_path']).read_text(encoding='utf-8'))
    expected = dict(protocol_version=config['protocol_version'],feature_window_end=12,outcome_window='(12,24]',label_shift_hours=6)
    if any(audit.get(k)!=v for k,v in expected.items()):
        raise ValueError('Audit does not match protocol/time-window declarations')
    if not audit.get('data_version') or not audit.get('site_meanings') or not audit.get('exclusion_counts'):
        raise ValueError('Audit requires data version, site meanings and inclusion/exclusion counts')
    if bool(audit.get('synthetic',False)) != bool(config.get('synthetic',False)):
        raise ValueError('Synthetic/clinical audit mismatch')
    if audit['site_meanings'].keys() != {'A','B'}:
        raise ValueError('Audit must describe both hospital identifiers')
    counts=audit['exclusion_counts']
    if any(not isinstance(counts.get(k),int) or counts[k]<0 for k in ['input','included','excluded']):
        raise ValueError('Audit counts require nonnegative integer input/included/excluded')
    if counts['included']!=len(frame) or counts['input']!=counts['included']+counts['excluded']:
        raise ValueError('Audit inclusion/exclusion counts inconsistent with cohort')
    validate_cohort(frame,splits,config['features'])
    return frame.set_index('patient_id',drop=False), splits, audit
