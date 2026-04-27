"""Fine-tune the T-stage Qwen3.5-27B model on simple data."""

import pandas as pd

from tnm.config import SUBMISSION_DATA_DIR
from tnm.training.llm_sft_trainer import LLMSFTFullTrainer


TRAIN_PATH = SUBMISSION_DATA_DIR / '01-validation/challenge/tcga_tnm_train.csv'

LABELS = ('T1', 'T2', 'T3', 'T4')
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 4096
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
SEED = 0


def prepare_stage_dataframe(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    df = df[['patient_filename', 'text', stage]].dropna(subset=[stage]).copy()
    df[stage] = stage.upper() + df[stage].astype(int).astype(str)
    return df


def main() -> None:
    df = prepare_stage_dataframe(pd.read_csv(TRAIN_PATH), 't')

    LLMSFTFullTrainer(
        stage='t',
        labels=LABELS,
        df=df,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        seed=SEED,
        model_name='Qwen/Qwen3.5-27B',
        output_name='qwen27b_simple',
        max_seq_length=MAX_SEQ_LENGTH,
    ).run()


if __name__ == '__main__':
    main()
