"""Reproduce the primary S0/S4 analysis and compare with frozen aggregates."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.conformal import calibrate, prediction_sets, singleton_predictions
from src.data import FEATURES, PHYSIO, load_cohort
from src.experiment import _score_features
from src.imputation import PosteriorImputer
from src.metrics import evaluate, paired_bootstrap
from src.models import choose_classifier, predict_draws
from src.selection import batch_accept, fit_threshold, frozen_accept
from src.utils import stable_seed


SEED = 2026
CAPACITY = 0.60
METHODS = ('B2', 'B3', 'B4')
FILTER = 'split_seed == 2026 and model == "LR" and M == 10 and alpha == 0.10 and target_acceptance == 0.60'
TOLERANCE = {'rtol': 1e-12, 'atol': 1e-14}


def auroc_interval(y, p):
    rng = np.random.default_rng(SEED)
    estimates = []
    for _ in range(1000):
        sample = rng.integers(0, len(y), len(y))
        if np.unique(y[sample]).size == 2:
            estimates.append(roc_auc_score(y[sample], p[sample]))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def main():
    evidence = ROOT / 'evidence' / 'main'
    cfg = yaml.safe_load((ROOT / 'configs/main.yaml').read_text(encoding='utf8'))
    for key in ('data_path', 'audit_path', 'split_pattern'):
        cfg[key] = str(ROOT / cfg[key])
    frame, splits, _ = load_cohort(cfg, SEED)
    roles = {role: frame.loc[ids] for role, ids in splits.items()}
    train, selection, calibration = (roles[r] for r in ('train', 'selection', 'calibration'))
    imputer = PosteriorImputer(cfg['rounds'], SEED).fit(train[FEATURES])

    def transform(role):
        part = roles[role]
        return imputer.transform(part[FEATURES], splits[role], M=cfg['M'], role=role)[0]

    selection_draws = transform('selection')
    model, regularization, _ = choose_classifier(
        transform('train'), train.y_true.to_numpy(),
        selection_draws, selection.y_true.to_numpy(), SEED,
    )
    np.testing.assert_allclose(regularization, 0.1)
    cal = predict_draws(model, transform('calibration'))['p_mean']
    q = calibrate(cal, calibration.y_true.to_numpy(), cfg['alpha'])
    prediction = predict_draws(model, selection_draws)
    pool, _ = singleton_predictions(prediction_sets(prediction['p_mean'], q))
    features = _score_features(prediction, selection[PHYSIO].isna().mean(axis=1).to_numpy())
    tables = {
        name: pd.read_csv(evidence / f'{name}.csv', float_precision='round_trip').query(FILTER)
        for name in ('metrics', 'thresholds', 'bootstrap_differences')
    }
    rules = {}
    for i, method in enumerate(METHODS):
        rule = fit_threshold(splits['selection'], pool, features[:, i], CAPACITY)
        expected = tables['thresholds'].query('method == @method').iloc[0]
        np.testing.assert_allclose([q, rule['threshold'], rule['selection_acceptance']],
                                   expected[['q', 'threshold', 'selection_acceptance']].to_numpy(float),
                                   **TOLERANCE)
        rules[method] = rule

    paired_expected = json.loads((evidence / 'primary_dispositions.json').read_text(encoding='utf8'))
    intervals = {'S0': [0.579, 0.947], 'S4': [0.407, 0.596]}
    for scenario, role in (('S0', 'test'), ('S4', 'external')):
        part = roles[role]
        ids, y = part.patient_id.to_numpy(), part.y_true.to_numpy()
        prediction = predict_draws(model, transform(role))
        p = prediction['p_mean']
        pool, labels = singleton_predictions(sets := prediction_sets(p, q))
        scores = _score_features(prediction, part[PHYSIO].isna().mean(axis=1).to_numpy())
        actual_auc_interval = auroc_interval(y, p)
        np.testing.assert_equal(np.round(actual_auc_interval, 3), intervals[scenario])
        decisions = {}
        for i, method in enumerate(METHODS):
            for mode, accepted in (
                ('batch', batch_accept(ids, pool, scores[:, i], CAPACITY)),
                ('frozen', frozen_accept(ids, pool, scores[:, i], rules[method])),
            ):
                observed = evaluate(y, p, labels, sets, accepted)
                expected = tables['metrics'].query(
                    'scenario == @scenario and method == @method and selection_mode == @mode and imputation_batch == 0'
                ).iloc[0]
                np.testing.assert_allclose(list(observed.values()),
                                           expected[list(observed)].to_numpy(float), **TOLERANCE)
                decisions[method, mode] = accepted
        for mode in ('batch', 'frozen'):
            a, b = decisions['B2', mode], decisions['B4', mode]
            dispositions = {
                'both_missed': int(np.sum((y == 1) & a & b)),
                'only_confidence_missed': int(np.sum((y == 1) & a & ~b)),
                'only_disagreement_missed': int(np.sum((y == 1) & ~a & b)),
                'neither_missed': int(np.sum((y == 1) & ~a & ~b)),
            }
            if dispositions != paired_expected['results'][scenario][mode]:
                raise ValueError(f'Paired dispositions differ for {scenario} {mode}')
            frozen = {'fixed_a': b, 'fixed_b': a} if mode == 'frozen' else {}
            bootstrap = paired_bootstrap(
                ids, y, labels, pool, scores[:, 2], scores[:, 0], CAPACITY,
                cfg['bootstrap'], stable_seed(SEED, scenario, 'B4', 'B2', CAPACITY, mode), **frozen,
            )
            expected = tables['bootstrap_differences'].query(
                'scenario == @scenario and selection_mode == @mode and imputation_batch == 0'
            ).iloc[0]
            np.testing.assert_allclose([bootstrap['ci_low'], bootstrap['ci_high']],
                                       expected[['ci_low', 'ci_high']].to_numpy(float), **TOLERANCE)
        print(json.dumps({'scenario': scenario, 'positives': int(y.sum()),
                          'auroc': roc_auc_score(y, p), 'auroc_95_interval': actual_auc_interval,
                          'metrics_matched': 6, 'paired_intervals_matched': 2}), flush=True)


if __name__ == '__main__':
    main()
