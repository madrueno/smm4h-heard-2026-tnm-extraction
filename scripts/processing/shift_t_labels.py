import pandas as pd

from tnm.config import SUBMISSION_DATA_DIR


INPUT_PATH = (
    SUBMISSION_DATA_DIR / '03-postevaluation' / 'predict' / 'predictions_qwen_sft.csv'
)

OUTPUT_PATH = (
    SUBMISSION_DATA_DIR
    / '03-postevaluation'
    / 'predict'
    / 'predictions_qwen_sft_t_shifted.csv'
)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df['t'] = df['t'] - 1
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == '__main__':
    main()
