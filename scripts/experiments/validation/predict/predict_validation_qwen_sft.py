"""Generate Task6 predictions using Qwen3.5-27B fine-tuned with SFT.

Uses the paper Qwen adapters trained on simple data and predicts T, N, M
stages for the validation split.
"""

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tnm.config import CUSTOM_MODELS_DIR, STAGES, SUBMISSION_DATA_DIR
from tnm.llm.prompts import MStagePrompt, NStagePrompt, TStagePrompt


BASE_MODEL = 'Qwen/Qwen3.5-27B'

MODEL_PATHS = {
    't': CUSTOM_MODELS_DIR / 'qwen27b_simple/t/<run_dir>/adapter',
    'n': CUSTOM_MODELS_DIR / 'qwen27b_simple/n/<run_dir>/adapter',
    'm': CUSTOM_MODELS_DIR / 'qwen27b_simple/m/<run_dir>/adapter',
}

STAGE_PROMPTS = {'t': TStagePrompt(), 'n': NStagePrompt(), 'm': MStagePrompt()}
PREFIX = {'t': 'T', 'n': 'N', 'm': 'M'}

STAGE_LABELS = {
    't': ('T1', 'T2', 'T3', 'T4'),
    'n': ('N0', 'N1', 'N2', 'N3'),
    'm': ('M0', 'M1'),
}

TASK6_PATH = (
    SUBMISSION_DATA_DIR / '01-validation' / 'challenge' / 'tcga_tnm_val_no_res.csv'
)
OUTPUT_PATH = (
    SUBMISSION_DATA_DIR / '01-validation' / 'predict' / 'predictions_qwen_sft.csv'
)


def build_prefix_allowed_fn(tokenizer, labels, prompt_len):
    """Build a constrained decoding function that forces output to valid labels."""
    label_token_ids = [
        tokenizer.encode(label, add_special_tokens=False) for label in labels
    ]

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        gen_len = input_ids.shape[0] - prompt_len
        allowed = set()
        for token_seq in label_token_ids:
            if gen_len < len(token_seq) and all(
                input_ids[prompt_len + i] == token_seq[i] for i in range(gen_len)
            ):
                allowed.add(token_seq[gen_len])
        return list(allowed) if allowed else [tokenizer.eos_token_id]

    return prefix_allowed_tokens_fn


def predict_stage(
    texts: list[str],
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt,
    labels: tuple[str, ...],
) -> list[int]:
    """Run constrained greedy generation and return numeric stage predictions."""
    model.eval()
    device = next(model.parameters()).device
    predictions = []

    for text in tqdm(texts, desc='Predicting'):
        messages = [{'role': 'user', 'content': prompt.get_prompt(text)}]
        model_inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors='pt',
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}

        prompt_len = model_inputs['input_ids'].shape[1]
        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=build_prefix_allowed_fn(
                    tokenizer, labels, prompt_len
                ),
            )

        generated = tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()

        pred = labels[0]
        for label in labels:
            if label in generated:
                pred = label
                break

        predictions.append(int(pred[1:]))

    return predictions


def main():
    df = pd.read_csv(TASK6_PATH)
    print(f'Task6 rows: {len(df)}')

    for stage in STAGES:
        adapter_path = MODEL_PATHS[stage]

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map='auto'
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)

        df[stage] = predict_stage(
            df['text'].tolist(),
            model,
            tokenizer,
            STAGE_PROMPTS[stage],
            STAGE_LABELS[stage],
        )
        n_covered = df[stage].notna().sum()
        n_missing = df[stage].isna().sum()
        print(
            f'{stage.upper()} covered: {n_covered} / {len(df)} ({100 * n_covered / len(df):.1f}%), empty: {n_missing}'
        )

        print(f'\n--- {stage.upper()} distribution ---')
        dist = df[stage].value_counts().sort_index()
        for label, count in dist.items():
            print(
                f'  {PREFIX[stage]}{int(label)}: {count:>5}  ({100 * count / len(df):5.1f}%)'
            )

        del model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = df[['patient_filename', 'text', 't', 'n', 'm']].copy()
    for stage in STAGES:
        out[stage] = out[stage].astype('Int64')

    out.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved predictions to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
