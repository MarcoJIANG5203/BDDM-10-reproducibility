"""Rebuild v1 inputs locally from the two PhysioNet 2019 public training sets."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from build_a_cohort_v1 import FEATURES, REASONS, extract, make_split, write_csv


SOURCE = {
    'A': (20336, 'a6386e6faa958ca3d75a7e2380673c85c305124e5d9f325961eeb7fa5606aeab'),
    'B': (20000, '0dd8ad8426a5f9e469411fe9cb19be0686c1d18f782859e213b9047cdc455d32'),
}
EXPECTED = {
    'cohort.csv': '5e6c56564cdbf6bbfb6e91756b14bc7a6f7a8aa37e3f86ce723aca569da08ce2',
    'seed_2026.json': 'c706075eadfebd1f10ac59e67b4d3b3693bf95808dce420a527a02fe7e522768',
    'seed_2027.json': '7d92c0d71f1955f3e2f73b1966b8422382a4003f8e5d9810dd198c437c6616b8',
    'seed_2028.json': '5915bc32e0d2d8d951957055a75b5f0df663d91049aba6ee65fd6dec42a8b549',
    'seed_2029.json': 'ee6619288de5e3f7723729969630ec65e9bc4a6b0ce623cdede398679960a1db',
    'seed_2030.json': '901c2d4e0206285ec8744fb747fba334d2beb69808815fc05fbc7d0c2fd6bf5e',
}


def read_records(raw, site):
    name = f'training_set{site}'
    archive = raw / f'{name}.zip'
    if archive.is_file():
        with ZipFile(archive) as source:
            for item in sorted(source.namelist()):
                yield item, source.read(item)
        return
    for path in sorted((raw / name).glob('*.psv')):
        yield path.name, path.read_bytes()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def verify_generated(path):
    canonical = path.read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(canonical).hexdigest() != EXPECTED[path.name]:
        raise ValueError(f'Frozen v1 content mismatch for {path.name}')


def build(raw, output):
    cohort = []
    flow = Counter()
    site_counts = {}
    for site, (expected_count, expected_digest) in SOURCE.items():
        count = 0
        site_rows = []
        digest = hashlib.sha256()
        for name, payload in read_records(raw, site):
            if Path(name).name != name or not name.endswith('.psv'):
                raise ValueError('Unexpected training-set entry')
            digest.update(f'{name}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
            audit, row = extract(payload.decode('utf-8-sig'), site, name)
            count += 1
            flow[audit['reason']] += 1
            if row is not None:
                site_rows.append(row)
        if count != expected_count or digest.hexdigest() != expected_digest:
            raise ValueError(f'PhysioNet v1.0.0 input verification failed for set {site}')
        site_counts[site] = {'raw': count, 'included': len(site_rows),
                             'positive': sum(row['y_true'] for row in site_rows)}
        cohort.extend(site_rows)
        print(f'Set {site}: {site_counts[site]}', flush=True)

    processed = output / 'data' / 'processed'
    splits = output / 'splits'
    processed.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    columns = ['patient_id', 'site', 'y_true', *FEATURES]
    csv_path = processed / 'cohort.csv'
    write_csv(csv_path, columns, cohort)
    verify_generated(csv_path)
    frame = pd.DataFrame(cohort, columns=columns)
    frame['patient_id'] = frame.patient_id.astype('string')
    frame.to_parquet(processed / 'cohort.parquet', index=False)
    for seed in range(2026, 2031):
        path = splits / f'seed_{seed}.json'
        write_json(path, make_split(cohort, seed))
        verify_generated(path)
    total = sum(count for count, _ in SOURCE.values())
    write_json(processed / 'cohort_audit.json', {
        'protocol_version': 'v1', 'synthetic': False,
        'data_version': 'a-data-v1-20261104-r1',
        'feature_window_end': 12, 'outcome_window': '(12,24]',
        'label_shift_hours': 6,
        'site_meanings': {'A': 'PhysioNet training set A', 'B': 'PhysioNet training set B'},
        'site_counts': site_counts,
        'exclusion_counts': {'input': total, 'included': len(cohort),
                             'excluded': total - len(cohort),
                             **{key: flow[key] for key in REASONS[:-1]}},
        'audit_scope': 'Locally regenerated metadata. Cohort CSV and five fixed splits '
                       'verified against frozen v1 digests with normalized line endings.',
    })
    print('Frozen v1 cohort and all five patient splits verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('.local'))
    args = parser.parse_args()
    build(args.raw, args.output)
