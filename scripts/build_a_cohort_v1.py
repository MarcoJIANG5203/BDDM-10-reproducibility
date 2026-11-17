"""A-owned raw extraction for the 2026-11-02 protocol v1 (not B's loader).

Raw files are immutable. Output data are clinical, not synthetic. Human C review
and the pre-freeze precision decision are separate from executable validation.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import zipfile

PHYSIO='HR O2Sat Temp SBP MAP DBP Resp EtCO2 BaseExcess HCO3 FiO2 pH PaCO2 SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total TroponinI Hct Hgb PTT WBC Fibrinogen Platelets'.split()
FEATURES=PHYSIO+['Age','Gender']
HEADER=PHYSIO+['Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS','SepsisLabel']
ROLES=['train','selection','calibration','test']
REASONS=['invalid_hourly_records','missing_initial_hours','insufficient_followup','first_observation_positive','onset_at_or_before_12','invalid_input_demographics','included']
VERSION='a-clinical-protocol-v1-20261103-full-r1'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='utf-8')

def write_csv(p,cols,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

def extract(text,site,name):
 r={'patient_id':site+'_'+Path(name).stem,'site':site,'source_filename':name,
    'observed_rows':0,'first_hour':None,'last_hour':None,'first_positive_hour':None,
    'onset_proxy':None,'first_observation_positive':None,'reason':'invalid_hourly_records','y_true':None}
 try:
  parsed=list(csv.reader(io.StringIO(text),delimiter='|'))
  if parsed[0]!=HEADER or any(len(x)!=41 for x in parsed[1:]):return r,None
  values=[[float(v) for v in x] for x in parsed[1:]]
  if not values or any(math.isinf(v) for row in values for v in row):return r,None
  times=[x[39] for x in values];labels=[x[40] for x in values]
  if any(not math.isfinite(t) or t<1 or int(t)!=t for t in times):return r,None
  if any(b!=a+1 for a,b in zip(times,times[1:])):return r,None
  if any(y not in (0,1) for y in labels) or any(a>b for a,b in zip(labels,labels[1:])):return r,None
 except (ValueError,IndexError):return r,None
 first=next((i for i,y in enumerate(labels) if y==1),None)
 onset=times[first]+6 if first is not None and first>0 else None
 r.update(observed_rows=len(values),first_hour=int(times[0]),last_hour=int(times[-1]),
          first_positive_hour=int(times[first]) if first is not None else None,
          onset_proxy=int(onset) if onset is not None else None,first_observation_positive=int(labels[0]==1))
 if times[0]!=1:r['reason']='missing_initial_hours';return r,None
 if times[-1]<24:r['reason']='insufficient_followup';return r,None
 if labels[0]==1:r['reason']='first_observation_positive';return r,None
 if onset is not None and onset<=12:r['reason']='onset_at_or_before_12';return r,None
 hist=[x for x in values if x[39]<=12]
 assert len(hist)==12
 last=[]
 for j in range(36):
  obs=[row[j] for row in hist if not math.isnan(row[j])]
  last.append(obs[-1] if obs else float('nan'))
 if not math.isnan(last[35]) and last[35] not in (0,1):r['reason']='invalid_input_demographics';return r,None
 y=int(onset is not None and 12<onset<=24)
 r.update(reason='included',y_true=y)
 row={'patient_id':r['patient_id'],'site':site,'y_true':y}
 row.update(zip(FEATURES,last))
 return r,row

def allocate(n):
 exact=[n*f for f in [.60,.10,.15,.15]]
 counts=[math.floor(x) for x in exact]
 order=sorted(range(4),key=lambda i:(-(exact[i]-counts[i]),i))
 for i in order[:n-sum(counts)]:counts[i]+=1
 return counts

def make_split(cohort,seed):
 result={k:[] for k in ROLES+['external']};rng=random.Random(seed)
 for y in [0,1]:
  ids=sorted(r['patient_id'] for r in cohort if r['site']=='A' and r['y_true']==y)
  rng.shuffle(ids);offset=0
  for role,n in zip(ROLES,allocate(len(ids))):
   result[role]+=ids[offset:offset+n];offset+=n
 for role in ROLES:result[role].sort()
 result['external']=sorted(r['patient_id'] for r in cohort if r['site']=='B')
 return result

def build(project,raw_dir,output):
 processed=output/'data/processed';processed.mkdir(parents=True,exist_ok=True)
 cohort=[];records=[];raw_hashes={}
 for site,expected in [('A',20336),('B',20000)]:
  archive=raw_dir/f'training_set{site}.zip';raw_hashes[archive.name]=sha(archive)
  with zipfile.ZipFile(archive) as z:
   names=sorted(z.namelist());assert len(names)==len(set(names))==expected
   for name in names:
    assert name.endswith('.psv')
    audit,row=extract(z.read(name).decode('utf-8-sig'),site,name);records.append(audit)
    if row is not None:cohort.append(row)
  print(f'Extracted raw site {site}',flush=True)
 # CSV is explicitly accepted by the handover; emit Parquet for B's default config too.
 write_csv(processed/'cohort.csv',['patient_id','site','y_true']+FEATURES,cohort)
 import pandas as pd
 frame=pd.DataFrame(cohort,columns=['patient_id','site','y_true']+FEATURES)
 frame['patient_id']=frame.patient_id.astype('string');frame.to_parquet(processed/'cohort.parquet',index=False)
 write_csv(processed/'record_audit.csv',list(records[0]),records)
 abnormal=[r for r in records if r['reason']!='included']
 write_csv(processed/'excluded_records.csv',list(records[0]),abnormal)
 stats=[];split_counts={};indexed={r['patient_id']:r for r in cohort}
 for seed in range(2026,2031):
  splits=make_split(cohort,seed);dump(output/f'splits/seed_{seed}.json',splits)
  split_counts[str(seed)]={}
  for role,ids in splits.items():
   pos=sum(indexed[i]['y_true'] for i in ids)
   split_counts[str(seed)][role]={'n':len(ids),'positive':pos,'negative':len(ids)-pos}
 for site in ['A','B']:
  part=[r for r in cohort if r['site']==site]
  for k in FEATURES:
   stats.append({'site':site,'feature':k,'n':len(part),'missing':sum(math.isnan(r[k]) for r in part),
                 'missing_fraction':sum(math.isnan(r[k]) for r in part)/len(part)})
 write_csv(processed/'feature_missingness.csv',list(stats[0]),stats)
 flow=collections.Counter(r['reason'] for r in records)
 sites={s:{'raw':sum(r['site']==s for r in records),'included':sum(r['site']==s for r in cohort),
           'positive':sum(r['y_true'] for r in cohort if r['site']==s),
           'exclusions':{reason:sum(r['site']==s and r['reason']==reason for r in records) for reason in REASONS[:-1]}}
        for s in ['A','B']}
 # Twenty actual source files, retained in the data bundle for independent C review.
 cases=[]
 for site in ['A','B']:
  with zipfile.ZipFile(raw_dir/f'training_set{site}.zip') as z:
   for y in [0,1]:
    pool=[r for r in records if r['reason']=='included' and r['site']==site and r['y_true']==y]
    pool.sort(key=lambda r:hashlib.sha256(('C-review|'+r['patient_id']).encode()).hexdigest())
    assert len(pool)>=5
    for r in pool[:5]:
     target=processed/'review_cases'/site/r['source_filename'];target.parent.mkdir(parents=True,exist_ok=True)
     target.write_bytes(z.read(r['source_filename']))
     cases.append({**r,'raw_evidence':target.relative_to(output).as_posix(),
                   'raw_sha256':sha(target),'C_review_status':'PENDING_HUMAN_REVIEW'})
 write_csv(processed/'review_10_positive_10_negative.csv',list(cases[0]),cases)
 source_paths=['T1_统一实验说明_v1.md','T1_技术交接约定.md','docs/A到B_数据交接说明.md','configs/protocol_v1.yaml','src/data.py']
 # Hash only artifacts generated by this extraction. Never hash a previous
 # cohort_audit.json into its replacement or absorb incidental README files.
 generated=[processed/name for name in ['cohort.csv','cohort.parquet','record_audit.csv',
      'excluded_records.csv','feature_missingness.csv','review_10_positive_10_negative.csv']]
 generated += [output/c['raw_evidence'] for c in cases]
 tracked={p.relative_to(output).as_posix():sha(p) for p in sorted(generated)}
 tracked.update({f'splits/seed_{s}.json':sha(output/f'splits/seed_{s}.json') for s in range(2026,2031)})
 audit={'protocol_version':'v1','data_version':VERSION,'synthetic':False,
  'feature_window_end':12,'outcome_window':'(12,24]','label_shift_hours':6,
  'site_meanings':{'A':'Beth Israel Deaconess Medical Center; PhysioNet 2019 training_setA',
                   'B':'Emory University Hospital; PhysioNet 2019 training_setB'},
  'exclusion_counts':{'input':len(records),'included':len(cohort),'excluded':len(records)-len(cohort),
                      'raw_patients':len(records),**{k:flow[k] for k in REASONS[:-1]}},
  'exclusion_order':REASONS,'exclusion_reason_counts_are_mutually_exclusive':True,
  'cohort_rule':'Require every ICU hour1--24. Exclude shifted-label early/left-censored onset. No age >=18 filter is specified by v1, so none is added. Missing Age/Gender is allowed.',
  'feature_order':FEATURES,'aggregation':'last observed value at actual ICULOS<=12; NaN if absent; no imputation',
  'split_algorithm':'Python random.Random(seed); sorted IDs; shuffle y=0 then y=1 within A; largest-remainder apportionment 60/10/15/15 with role-order ties; external B unchanged',
  'site_counts':sites,'split_counts':split_counts,
  'precision_gates':{'site_positive_minimum':200,'internal_test_positive_minimum':50,
     'site_pass':{s:sites[s]['positive']>=200 for s in sites},
     'internal_test_pass':{s:d['test']['positive']>=50 for s,d in split_counts.items()},
     'status':'NOT_MET_REQUIRES_PRE_FREEZE_DISCUSSION_NO_SILENT_ENDPOINT_CHANGE'},
  'human_review':{'A_owner_review':'PENDING','C_independent_10_positive_10_negative':'PENDING','E_precision_decision':'PENDING',
                  'note':'Automated raw-level checks are not human attestation or recipient acceptance.'},
  'model_fitting_performed':False,'clinical_ground_truth_independently_adjudicated':False,
  'raw_source':{'version':'1.0.0','year':2019,'doi':'10.13026/v64v-d857',
     'url':'https://physionet.org/content/challenge-2019/1.0.0/',
     'companion_article_year':2020,'companion_article_doi':'10.1097/CCM.0000000000004145'},
  'raw_archive_sha256':raw_hashes,'file_sha256':tracked,
  'code_sha256':sha(Path(__file__)),
  'source_contract_sha256':{p:sha(project/p) for p in source_paths},
  'python_version':platform.python_version(),
  'source_repository_commit':json.loads((project/'repository_provenance.json').read_text())['source_repository_commit'] if (project/'repository_provenance.json').is_file() else subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip(),
  'source_checkout_mode':'preserved_snapshot' if (project/'repository_provenance.json').is_file() else 'live_git',
  'analysis_unit_limitation':'one released subject record file; cross-stay person linkage is unavailable',
  'changes_from_previous_candidate':['Strict full hours1--24 replaces permissive delayed-start inclusion',
     '36 inputs; HospAdmTime removed','No undocumented adult-only eligibility filter',
     'A stratified60/10/15/15 with selection; seeds2026--2030 replace old split',
     'patient_id and y_true replace record_id and outcome_12_24; output paths match B contract'],
  'release_status':'LOCAL_PREPARED_NOT_PUBLISHED_NOT_ACCEPTED'}
 dump(processed/'cohort_audit.json',audit)
 print(json.dumps({'sites':sites,'splits':split_counts['2026'],'total':len(cohort)},indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--raw-dir',type=Path);parser.add_argument('--output-root',type=Path)
 args=parser.parse_args();project=args.project_root.resolve()
 build(project,(args.raw_dir or project/'physionet2019_original').resolve(),(args.output_root or project).resolve())
