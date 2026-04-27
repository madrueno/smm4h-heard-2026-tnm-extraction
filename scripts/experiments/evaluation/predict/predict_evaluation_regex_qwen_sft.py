"""Fill empty regex predictions with Qwen SFT predictions.

Uses regex-based TNM predictions as the primary source and fills empty cells
(where no regex match was found) with predictions from the Qwen SFT model.
"""

import pandas as pd

from tnm.config import STAGES, SUBMISSION_DATA_DIR


PREDICT_DIR = SUBMISSION_DATA_DIR / '02-evaluation' / 'predict'
REGEX_PATH = PREDICT_DIR / 'predictions_regex.csv'
QWEN_SFT_PATH = PREDICT_DIR / 'predictions_qwen_sft.csv'
OUTPUT_PATH = PREDICT_DIR / 'predictions_regex_qwen_sft.csv'


def main():
    regex_df = pd.read_csv(REGEX_PATH)
    qwen_sft_df = pd.read_csv(QWEN_SFT_PATH)

    print(f'Loaded {len(regex_df)} rows from {REGEX_PATH.name}')
    print(f'Loaded {len(qwen_sft_df)} rows from {QWEN_SFT_PATH.name}')

    out = regex_df[['patient_filename', 'text']].copy()

    for stage in STAGES:
        regex_col = regex_df[stage]
        qwen_col = (
            qwen_sft_df.set_index('patient_filename')
            .loc[regex_df['patient_filename'], stage]
            .values
        )

        n_missing = regex_col.isna().sum()
        filled = regex_col.fillna(pd.Series(qwen_col, index=regex_df.index))
        n_filled = filled.notna().sum() - (len(filled) - n_missing)

        print(
            f'{stage.upper()}: {n_missing} empty cells filled from qwen_sft'
            f' ({n_filled} filled, {filled.isna().sum()} remaining empty)'
        )

        out[stage] = filled.astype('Int64')

    out.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved predictions to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
