"""Explicit synthetic boundary cases; never used as clinical observations."""
import math
import pytest
from scripts.build_a_cohort_v1 import HEADER,FEATURES,extract,make_split,allocate

def fixture(first=None,n=30,start=1,age=50,gender=1,future=999):
 lines=['|'.join(HEADER)]
 for t in range(start,start+n):
  values=['NaN']*41
  values[0]=str(70 if t<=12 else future)
  values[34]=str(age);values[35]=str(gender);values[38]='-20';values[39]=str(t)
  values[40]=str(int(first is not None and t>=first));lines.append('|'.join(values))
 return '\n'.join(lines)

@pytest.mark.parametrize('first,reason,y',[(None,'included',0),(6,'onset_at_or_before_12',None),(7,'included',1),(18,'included',1),(19,'included',0),(1,'first_observation_positive',None)])
def test_explicit_time_boundaries(first,reason,y):
 meta,row=extract(fixture(first=first),'A','p1.psv')
 assert meta['reason']==reason and meta['y_true']==y
 if first==1:assert meta['onset_proxy'] is None

def test_full_hour_requirement_and_last_hour():
 assert extract(fixture(start=2),'A','p1.psv')[0]['reason']=='missing_initial_hours'
 assert extract(fixture(n=23),'A','p1.psv')[0]['reason']=='insufficient_followup'
 assert extract(fixture(n=24),'A','p1.psv')[0]['reason']=='included'

def test_missing_age_gender_allowed_and_no_undocumented_age_filter():
 meta,row=extract(fixture(age='NaN',gender='NaN'),'A','p1.psv')
 assert meta['reason']=='included' and math.isnan(row['Age']) and math.isnan(row['Gender'])
 assert extract(fixture(age=17),'A','p1.psv')[0]['reason']=='included'
 assert extract(fixture(gender=2),'A','p1.psv')[0]['reason']=='invalid_input_demographics'

def test_no_future_fill_and_exact_whitelist():
 _,a=extract(fixture(future=888),'A','p1.psv');_,b=extract(fixture(future=-888),'A','p1.psv')
 assert list(a)==['patient_id','site','y_true']+FEATURES and len(FEATURES)==36
 assert 'HospAdmTime' not in a and a['HR']==b['HR']==70
 assert math.isnan(a['EtCO2']) and math.isnan(b['EtCO2'])

def test_reject_duplicate_gap_and_nonmonotonic_label():
 lines=fixture(first=7).splitlines();v=lines[4].split('|');v[39]='5';lines[4]='|'.join(v)
 assert extract('\n'.join(lines),'A','p1.psv')[0]['reason']=='invalid_hourly_records'
 lines=fixture(first=7).splitlines();v=lines[-1].split('|');v[40]='0';lines[-1]='|'.join(v)
 assert extract('\n'.join(lines),'A','p1.psv')[0]['reason']=='invalid_hourly_records'

@pytest.mark.parametrize('seed',range(2026,2031))
def test_five_seed_stratification(seed):
 rows=[dict(patient_id=f'A_{y}_{i:04}',site='A',y_true=y) for y in [0,1] for i in range(57)]
 rows += [dict(patient_id=f'B_{i}',site='B',y_true=i%2) for i in range(20)]
 split=make_split(rows,seed);assert split==make_split(list(reversed(rows)),seed)
 ids=[x for role in split.values() for x in role]
 assert len(ids)==len(set(ids))==len(rows)
 assert set(ids)=={r['patient_id'] for r in rows}
 assert all(x.startswith('B_') for x in split['external'])
 for role in ['train','selection']:
  assert {next(r['y_true'] for r in rows if r['patient_id']==x) for x in split[role]}=={0,1}
 assert [len(split[k]) for k in ['train','selection','calibration','test']]==[2*n for n in allocate(57)]
