"""Generate Task6 predictions using the best BioClinical-ModernBERT-large model.

Extracts T, N, M stages using fine-tuned AutoModelForSequenceClassification
checkpoints trained on simple data.
"""

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tnm.config import CUSTOM_MODELS_DIR, STAGES, SUBMISSION_DATA_DIR


MODEL_PATHS = {
    't': CUSTOM_MODELS_DIR / 'bio_clinical_modernbert_simple/t/<run_dir>/best',
    'n': CUSTOM_MODELS_DIR / 'bio_clinical_modernbert_simple/n/<run_dir>/best',
    'm': CUSTOM_MODELS_DIR / 'bio_clinical_modernbert_simple/m/<run_dir>/best',
}

LABEL_MAP = {
    't': {0: 'T1', 1: 'T2', 2: 'T3', 3: 'T4'},
    'n': {0: 'N0', 1: 'N1', 2: 'N2', 3: 'N3'},
    'm': {0: 'M0', 1: 'M1'},
}

MAX_LENGTH = {'t': 4096, 'n': 4096, 'm': 4096}

PREFIX = {'t': 'T', 'n': 'N', 'm': 'M'}

TASK6_PATH = (
    SUBMISSION_DATA_DIR / '01-validation' / 'challenge' / 'tcga_tnm_val_no_res.csv'
)
OUTPUT_PATH = (
    SUBMISSION_DATA_DIR
    / '01-validation'
    / 'predict'
    / 'predictions_bioclinicalmodernbert.csv'
)


def predict_stage(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    label_map: dict,
    max_length: int,
) -> list[int]:
    """Extract stage from text via BioClinical-ModernBERT inference."""
    model.eval()
    predictions = []

    for text in tqdm(texts, desc='Predicting'):
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        pred = label_map[logits.argmax(dim=-1).item()]
        predictions.append(int(pred[1:]))

    return predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = pd.read_csv(TASK6_PATH)
    print(f'Task6 rows: {len(df)}')

    # Predict
    for stage in STAGES:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATHS[stage]
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[stage])

        df[stage] = predict_stage(
            df['text'].tolist(),
            model,
            tokenizer,
            device,
            LABEL_MAP[stage],
            MAX_LENGTH[stage],
        )
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
