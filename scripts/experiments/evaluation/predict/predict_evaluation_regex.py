"""Generate Task6 predictions using regex-based TNM extraction.

Extracts T, N, M stages from explicit mentions in the text (e.g. pT2, N1, ypT3a).
Leaves the cell empty when no match is found.
"""

import re

import pandas as pd

from tnm.config import STAGES, SUBMISSION_DATA_DIR


TASK6_PATH = SUBMISSION_DATA_DIR / '02-evaluation' / 'challenge' / 'Task6_test.csv'
OUTPUT_PATH = (
    SUBMISSION_DATA_DIR / '02-evaluation' / 'predict' / 'predictions_regex.csv'
)


PATTERNS = {
    't': [
        # Compact TNM string, e.g. "T3N1", "T2aN0", "T3 N1"
        (re.compile(r'T([1-4])[a-z]?\s?N[0-3]'), None),
        # Prefixed T stage (pT2, cT3, ypT1)
        (re.compile(r'\b(?:yp|p|c|y)T([1-4])'), None),
    ],
    'n': [
        # Compact TNM string, e.g. "T3N1", "pT2aN0", "T3 N1"
        (re.compile(r'T[1-4][a-z]?\s?N([0-3])'), None),
        # Prefixed N stage (pN1, cN0, ypN2)
        (re.compile(r'\b(?:yp|p|c|y)N([0-3])'), None),
    ],
    'm': [
        # Compact TNM string, e.g. "N1M0", "N1 M0"
        (re.compile(r'N[0-3][a-z]{0,2}\s?M([01])'), None),
        # Prefixed M stage (pM1, cM0)
        (re.compile(r'\b(?:yp|p|c|y)M([01])'), None),
    ],
}
PREFIX = {'t': 'T', 'n': 'N', 'm': 'M'}


def predict_stage(text: str, stage: str) -> int | None:
    """Extract stage from text via regex, or return None if no match."""
    for pattern, handler in PATTERNS[stage]:
        m = pattern.search(text)
        if m:
            if handler is None:
                return int(m.group(1))
            if callable(handler):
                return handler(m)
            return handler
    return None


def main():
    df = pd.read_csv(TASK6_PATH)
    print(f'Task6 rows: {len(df)}')

    # Predict
    for stage in STAGES:
        df[stage] = df['text'].apply(lambda text, s=stage: predict_stage(text, s))
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

    # Save with integer values where matched, empty where not
    out = df[['patient_filename', 'text', 't', 'n', 'm']].copy()
    for stage in STAGES:
        out[stage] = out[stage].astype('Int64')

    out.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved predictions to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
