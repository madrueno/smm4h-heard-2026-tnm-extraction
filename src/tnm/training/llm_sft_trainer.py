from __future__ import annotations

import random
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from tnm.config import CUSTOM_MODELS_DIR
from tnm.llm.prompts import MStagePrompt, NStagePrompt, TStagePrompt
from tnm.llm.chat_templates import ChatTemplateOverrides


STAGE_PROMPTS = {'t': TStagePrompt(), 'n': NStagePrompt(), 'm': MStagePrompt()}


class LLMSFTFullTrainer:
    """SFT trainer that trains on the full dataset (no eval split)."""

    def __init__(
        self,
        stage: str,
        labels: tuple[str, ...],
        df: pd.DataFrame,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        num_epochs: int,
        seed: int,
        model_name: str,
        output_name: str,
        max_seq_length: int = 2048,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_4bit: bool = True,
    ) -> None:
        self.stage = stage
        self.labels = labels
        self.df = df
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        self.model_name = model_name
        self.output_name = output_name
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.load_in_4bit = load_in_4bit
        self.prompt = STAGE_PROMPTS[stage]

    def set_seed(self) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _format_sample(self, text: str, label: str) -> dict:
        return {
            'messages': [
                {'role': 'user', 'content': self.prompt.get_prompt(text)},
                {'role': 'assistant', 'content': label},
            ]
        }

    def build_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.truncation_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._ensure_assistant_mask_support()

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if self.load_in_4bit
            else None
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map='auto',
            dtype=torch.bfloat16,
        )

        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
            bias='none',
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)

    def _has_assistant_mask_support(self) -> bool:
        template = self.tokenizer.chat_template
        if isinstance(template, dict):
            template = template.get('default')
        if template is None or '{% generation %}' not in template:
            return False

        encoded = self.tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': 'Classify this report.'},
                {'role': 'assistant', 'content': self.labels[0]},
            ],
            tokenize=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        return any(encoded['assistant_masks'])

    def _ensure_assistant_mask_support(self) -> None:
        if self._has_assistant_mask_support():
            return

        override = ChatTemplateOverrides.get(self.model_name)
        if override is not None:
            self.tokenizer.chat_template = override

        if not self._has_assistant_mask_support():
            raise RuntimeError(
                f'Assistant masking is not supported for tokenizer {self.model_name}.'
            )

    def build_datasets(self) -> None:
        records = [
            self._format_sample(text, label)
            for text, label in zip(self.df['text'], self.df[self.stage])
        ]
        self.train_dataset = Dataset.from_list(records)

    def build_output_dirs(self) -> None:
        timestamp = int(datetime.now().timestamp())
        run_name = (
            f'{self.stage}_{self.output_name}_s{self.seed}_'
            f'bs{self.batch_size}_ga{self.gradient_accumulation_steps}_{timestamp}'
        )
        self.model_dir = CUSTOM_MODELS_DIR / self.output_name / self.stage / run_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def build_trainer(self) -> None:
        train_size = len(self.train_dataset)
        steps_per_epoch = max(
            1, train_size // (self.batch_size * self.gradient_accumulation_steps)
        )
        logging_steps = max(1, steps_per_epoch // 5)

        sft_config = SFTConfig(
            output_dir=str(self.model_dir),
            report_to='none',
            seed=self.seed,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            save_strategy='epoch',
            save_total_limit=1,
            logging_strategy='steps',
            logging_steps=logging_steps,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            max_length=self.max_seq_length,
            assistant_only_loss=True,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.train_dataset,
            processing_class=self.tokenizer,
        )

    def train(self) -> None:
        self.trainer.train()

    def save_outputs(self) -> None:
        adapter_dir = self.model_dir / 'adapter'
        self.trainer.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)
        print(f'Saved adapter to: {adapter_dir}')

    def run(self) -> None:
        self.set_seed()
        self.build_model()
        self.build_datasets()
        self.build_output_dirs()
        self.build_trainer()
        self.train()
        self.save_outputs()
