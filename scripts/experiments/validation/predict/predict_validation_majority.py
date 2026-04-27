"""Generate Task6 predictions using the majority-class baseline.

Predicts the most frequent stage from training data for each of T, N, M.
"""

import pandas as pd

from tnm.config import STAGES, SUBMISSION_DATA_DIR


TRAIN_PATH = SUBMISSION_DATA_DIR / '01-validation' / 'challenge' / 'tcga_tnm_train.csv'
TASK6_PATH = (
    SUBMISSION_DATA_DIR / '01-validation' / 'challenge' / 'tcga_tnm_val_no_res.csv'
)
OUTPUT_PATH = (
    SUBMISSION_DATA_DIR / '01-validation' / 'predict' / 'predictions_majority.csv'
)

PREFIX = {'t': 'T', 'n': 'N', 'm': 'M'}


def predict_stage(train: pd.DataFrame, texts: pd.Series, stage: str) -> pd.Series:
    """Predict the majority class from training data for a single stage."""
    majority = train[stage].dropna().mode()[0]
    return pd.Series(majority, index=texts.index)


def main():
    train = pd.read_csv(TRAIN_PATH)
    df = pd.read_csv(TASK6_PATH)
    print(f'Train rows: {len(train)}')
    print(f'Task6 rows: {len(df)}')

    # Predict
    for stage in STAGES:
        df[stage] = predict_stage(train, df['text'], stage)
        n_covered = df[stage].notna().sum()
        n_missing = df[stage].isna().sum()
        print(
            f'{stage.upper()} covered: {n_covered} / {len(df)} ({100 * n_covered / len(df):.1f}%), empty: {n_missing}'
        )

    # Print distribution
    for stage in STAGES:
        print(f'\n--- {stage.upper()} distribution ---')
        dist = df[stage].value_counts().sort_index()
        for label, count in dist.items():
            print(
                f'  {PREFIX[stage]}{int(label)}: {count:>5}  ({100 * count / len(df):5.1f}%)'
            )

    # Save with integer values
    out = df[['patient_filename', 'text', 't', 'n', 'm']].copy()
    for stage in STAGES:
        out[stage] = out[stage].astype('Int64')

    out.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved predictions to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
