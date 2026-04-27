"""Build TCGA_T0_TNMX_filtered by merging report text with TNM labels (filtered).

This pipeline explicitly discards records with:
- ``T_stage`` in ``{T0, TX}``
- ``N_stage`` in ``{NX}``
- ``M_stage`` in ``{MX}``
"""

import re

import pandas as pd

from tnm.config import INTERIM_DATA_DIR, RAW_DATA_DIR


TEXT_PATH = RAW_DATA_DIR / 'text' / 'TCGA_Reports.csv'
LABEL_T_PATH = RAW_DATA_DIR / 'labels' / 'TCGA_T14_patients.csv'
LABEL_N_PATH = RAW_DATA_DIR / 'labels' / 'TCGA_N03_patients.csv'
LABEL_M_PATH = RAW_DATA_DIR / 'labels' / 'TCGA_M01_patients.csv'

OUTPUT_PATH = INTERIM_DATA_DIR / 'TCGA_T0_TNMX_filtered.csv'

DISCARDED_STAGE_VALUES = {'T_stage': ['TX', 'T0'], 'N_stage': ['NX'], 'M_stage': ['MX']}


def map_stage(val: str, prefix: str) -> str:
    """Map a granular stage subtype to its macro-class (e.g. T1a -> T1)."""
    if pd.isna(val):
        return val
    match = re.search(rf'({prefix}[0-4X])', str(val))
    return match.group(1) if match else val


def build_master_dataset() -> pd.DataFrame:
    """Merge text+labels and apply TNM stage filtering policy (including T0/TX discard)."""
    df_text = pd.read_csv(TEXT_PATH)
    df_text['patient_id'] = df_text['patient_filename'].apply(
        lambda x: str(x).split('.')[0]
    )

    df_t = pd.read_csv(LABEL_T_PATH)
    df_n = pd.read_csv(LABEL_N_PATH)
    df_m = pd.read_csv(LABEL_M_PATH)

    df_labels = df_t.merge(
        df_n,
        on=['case_submitter_id', 'project_id', 'case_id'],
        how='outer',
        suffixes=('', '_n'),
    )
    df_labels = df_labels.merge(
        df_m,
        on=['case_submitter_id', 'project_id', 'case_id'],
        how='outer',
        suffixes=('', '_m'),
    )

    df_labels.rename(
        columns={
            'case_submitter_id': 'patient_id',
            'ajcc_pathologic_t': 'T_stage',
            'ajcc_pathologic_n': 'N_stage',
            'ajcc_pathologic_m': 'M_stage',
        },
        inplace=True,
    )

    df_master = df_text.merge(df_labels, on='patient_id', how='inner')

    # Map subtypes to macro-classes (e.g. T1a -> T1, N2b -> N2, M1a -> M1)
    for prefix in ['T', 'N', 'M']:
        col = f'{prefix}_stage'
        df_master[col] = df_master[col].apply(lambda x: map_stage(x, prefix))

    # Apply discard policy: explicitly remove T0/TX and unknown assessability codes (NX/MX)
    for col, values in DISCARDED_STAGE_VALUES.items():
        df_master = df_master[~df_master[col].isin(values)]

    # Drop duplicated report text to prevent cross-split leakage
    df_master = df_master.drop_duplicates(subset=['text'])

    return df_master


def main():
    df_master = build_master_dataset()
    print(f'Unified reports: {len(df_master)}')

    df_master = df_master[['patient_id', 'text', 'T_stage', 'N_stage', 'M_stage']]
    df_master = df_master.rename(columns={'T_stage': 't', 'N_stage': 'n', 'M_stage': 'm'})

    df_master.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
