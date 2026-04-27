#!/usr/bin/env python3
"""
LOCAL COMPETITION SCORER
Replicates the exact Codabench.

VALIDATION SET RULE:
- T validation: Samples where T label is MISSING from training file
- N validation: Samples where N label is MISSING from training file
- M validation: Samples where M label is MISSING from training file
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def load_validation_sets(train_path, github_metadata_path):
    """Identify which samples are in validation set for each task."""

    train = pd.read_csv(train_path)
    train['patient_id'] = train['patient_filename'].str.split('.').str[0]

    github = pd.read_csv(github_metadata_path)

    # Merge to see which labels are in training
    merged = github.merge(
        train[['patient_id', 't', 'n', 'm']],
        on='patient_id',
        how='left',
        suffixes=('', '_train'),
    )

    # Validation set = samples missing label from train AND have GitHub label
    val_sets = {
        't': merged[
            merged['t_train'].isna() & merged['t_actual_numeric'].notna()
        ].copy(),
        'n': merged[
            merged['n_train'].isna() & merged['n_actual_numeric'].notna()
        ].copy(),
        'm': merged[
            merged['m_train'].isna() & merged['m_actual_numeric'].notna()
        ].copy(),
    }

    return val_sets


def score_submission(
    predictions_path,
    train_path,
    github_metadata_path,
    verbose=True,
    discard_missing=False,
):

    # Load validation sets
    val_sets = load_validation_sets(train_path, github_metadata_path)

    # Load predictions
    predictions = pd.read_csv(predictions_path)
    predictions['patient_id'] = predictions['patient_filename'].str.split('.').str[0]

    results = {}
    all_y_true = []
    all_y_pred = []

    for stage, stage_name in [('t', 'T'), ('n', 'N'), ('m', 'M')]:
        val_set = val_sets[stage]

        val_with_pred = val_set.merge(
            predictions[['patient_id', stage]],
            on='patient_id',
            how='inner',
            suffixes=('', '_pred'),
        )

        pred_col = f'{stage}_pred'
        y_pred = np.nan_to_num(val_with_pred[pred_col].values, nan=-1).astype(int)
        y_true = val_with_pred[f'{stage}_actual_numeric'].astype(int).values

        if discard_missing:
            missing_mask = y_pred != -1
            y_pred = y_pred[missing_mask]
            y_true = y_true[missing_mask]

        correct = (y_pred == y_true).sum()
        total = len(y_pred)

        # Macro averaging to match platform scoring
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Per-class metrics for global per-class macro calculation
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        results[stage] = {
            'stage_name': stage_name,
            'evaluated': total,
            'correct': correct,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
        }

        # Accumulate for micro
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        if verbose:
            print(f'\n{stage_name} Stage:')
            print(f'  Samples evaluated: {total}')
            print(f'  Correct: {correct}/{total}')
            print(f'  Precision: {precision:.2f}')
            print(f'  Recall: {recall:.2f}')
            print(f'  F1: {f1:.2f}')

    # Micro: aggregate TP/FP/FN across all stages and classes
    micro_precision = precision_score(
        all_y_true, all_y_pred, average='micro', zero_division=0
    )
    micro_recall = recall_score(
        all_y_true, all_y_pred, average='micro', zero_division=0
    )
    micro_f1 = f1_score(all_y_true, all_y_pred, average='micro', zero_division=0)

    # Macro (per-stage): mean of per-stage macro F1s => (T_macro + N_macro + M_macro) / 3
    macro_per_stage_precision = np.mean([r['precision'] for r in results.values()])
    macro_per_stage_recall = np.mean([r['recall'] for r in results.values()])
    macro_per_stage_f1 = np.mean([r['f1'] for r in results.values()])

    # Macro (per-class): mean of all individual class F1s => (T1+T2+T3+T4+N0+N1+N2+N3+M0+M1) / 10
    all_class_precisions = []
    all_class_recalls = []
    all_class_f1s = []
    for stage_results in results.values():
        all_class_precisions.extend(stage_results['precision_per_class'])
        all_class_recalls.extend(stage_results['recall_per_class'])
        all_class_f1s.extend(stage_results['f1_per_class'])

    macro_per_class_precision = np.mean(all_class_precisions)
    macro_per_class_recall = np.mean(all_class_recalls)
    macro_per_class_f1 = np.mean(all_class_f1s)

    if verbose:
        print('\n' + '=' * 80)
        print('OVERALL RESULTS')
        print('=' * 80)
        print(
            f'\nTotal predictions evaluated: {sum(r["evaluated"] for r in results.values())}'
        )
        for s in ['t', 'n', 'm']:
            print(f'  {results[s]["stage_name"]}: {results[s]["evaluated"]}')
        print('\nMicro-average:')
        print(f'  Precision: {micro_precision:.2f}')
        print(f'  Recall: {micro_recall:.2f}')
        print(f'  F1: {micro_f1:.2f}')
        print('\nMacro-average (per-stage): Average of T/N/M stage macro scores')
        print(f'  Precision: {macro_per_stage_precision:.2f}')
        print(f'  Recall: {macro_per_stage_recall:.2f}')
        print(f'  F1: {macro_per_stage_f1:.2f}')
        print('\nMacro-average (per-class): Average of all individual class scores')
        print(f'  Precision: {macro_per_class_precision:.2f}')
        print(f'  Recall: {macro_per_class_recall:.2f}')
        print(f'  F1: {macro_per_class_f1:.2f}')

    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_per_stage_precision': macro_per_stage_precision,
        'macro_per_stage_recall': macro_per_stage_recall,
        'macro_per_stage_f1': macro_per_stage_f1,
        'macro_per_class_precision': macro_per_class_precision,
        'macro_per_class_recall': macro_per_class_recall,
        'macro_per_class_f1': macro_per_class_f1,
        'per_stage': results,
    }


if __name__ == '__main__':
    # Paths
    predictions_path = Path('data/submission/01-validation/predict/predictions_majority.csv')
    train_path = Path('data/submission/01-validation/challenge/tcga_tnm_train.csv')
    github_metadata_path = Path(
        'data/submission/01-validation/challenge/github_metadata.csv'
    )

    # Score original submission
    print('\nScoring:')
    results = score_submission(
        predictions_path, train_path, github_metadata_path, discard_missing=True
    )
