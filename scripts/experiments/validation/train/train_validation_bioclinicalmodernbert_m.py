"""Train the M-stage BioClinical-ModernBERT-large model."""

import pandas as pd

from tnm.config import SUBMISSION_DATA_DIR
from tnm.training.transformer_trainer import TransformerBasicTrainer


TRAIN_PATH = SUBMISSION_DATA_DIR / '01-validation/challenge/tcga_tnm_train.csv'

LABELS = ('M0', 'M1')
BATCH_SIZE = 8
MAX_LENGTH = 4096
LEARNING_RATE = 5e-6
NUM_EPOCHS = 3
SEED = 0


def prepare_stage_dataframe(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    df = df[['patient_filename', 'text', stage]].dropna(subset=[stage]).copy()
    df[stage] = stage.upper() + df[stage].astype(int).astype(str)
    return df


def main() -> None:
    df_train = prepare_stage_dataframe(pd.read_csv(TRAIN_PATH), 'm')

    TransformerBasicTrainer(
        stage='m',
        labels=LABELS,
        df_train=df_train,
        df_dev=None,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        seed=SEED,
        model_name='thomas-sounack/BioClinical-ModernBERT-large',
        output_name='bio_clinical_modernbert_simple',
    ).run()


if __name__ == '__main__':
    main()
