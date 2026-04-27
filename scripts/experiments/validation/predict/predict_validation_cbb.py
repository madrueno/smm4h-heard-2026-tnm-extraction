"""Generate Task6 predictions using the paper Clinical-BigBird baseline.

Extracts T, N, M stages using jkefeli/CancerStage_Classifier_{T,N,M}.
"""

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BigBirdForSequenceClassification

from tnm.config import STAGES, SUBMISSION_DATA_DIR


TOKENIZER_NAME = 'yikuan8/Clinical-BigBird'
MODEL_NAMES = {
    't': 'jkefeli/CancerStage_Classifier_T',
    'n': 'jkefeli/CancerStage_Classifier_N',
    'm': 'jkefeli/CancerStage_Classifier_M',
}

NUM_LABELS = {'t': 4, 'n': 4, 'm': 2}
LABEL_MAP = {
    't': {0: 'T1', 1: 'T2', 2: 'T3', 3: 'T4'},
    'n': {0: 'N0', 1: 'N1', 2: 'N2', 3: 'N3'},
    'm': {0: 'M0', 1: 'M1'},
}
PREFIX = {'t': 'T', 'n': 'N', 'm': 'M'}

MAX_LENGTH = {'t': 2048, 'n': 2048, 'm': 1024}

TASK6_PATH = (
    SUBMISSION_DATA_DIR / '01-validation' / 'challenge' / 'tcga_tnm_val_no_res.csv'
)
OUTPUT_PATH = SUBMISSION_DATA_DIR / '01-validation' / 'predict' / 'predictions_cbb.csv'


def predict_stage(
    texts: list[str],
    model: BigBirdForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    label_map: dict,
    max_length: int,
) -> list[int]:
    """Extract stage from text via Clinical-BigBird inference."""
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

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Predict
    for stage in STAGES:
        model = BigBirdForSequenceClassification.from_pretrained(
            MODEL_NAMES[stage], num_labels=NUM_LABELS[stage]
        ).to(device)

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
