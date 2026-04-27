from __future__ import annotations

import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from tnm.config import CUSTOM_MODELS_DIR
from tnm.training.dataset import EncodedDataset


class TransformerBasicTrainer:
    def __init__(
        self,
        stage: str,
        labels: tuple[str, ...],
        df_train: pd.DataFrame,
        df_dev: pd.DataFrame | None,
        batch_size: int,
        max_length: int,
        learning_rate: float,
        num_epochs: int,
        seed: int,
        model_name: str,
        output_name: str,
    ) -> None:
        self.stage = stage
        self.labels = labels
        self.df_train = df_train
        self.df_dev = df_dev
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        self.model_name = model_name
        self.output_name = output_name
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}

    def set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def compute_metrics(
        self, eval_pred: tuple[np.ndarray, np.ndarray]
    ) -> dict[str, float]:
        logits, y_true = eval_pred
        y_pred = torch.tensor(logits).argmax(dim=-1).numpy()
        return {'f1': float(f1_score(y_true, y_pred, average='macro'))}

    def build_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.labels), label2id=self.label_to_id
        )

    def build_datasets(self) -> None:
        train_labels = [self.label_to_id[label] for label in self.df_train[self.stage]]

        train_encodings = self.tokenizer(
            self.df_train['text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
        )

        self.train_dataset = EncodedDataset(train_encodings, train_labels)

        if self.df_dev is not None:
            dev_labels = [self.label_to_id[label] for label in self.df_dev[self.stage]]
            dev_encodings = self.tokenizer(
                self.df_dev['text'].tolist(),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
            )
            self.dev_dataset = EncodedDataset(dev_encodings, dev_labels)
        else:
            self.dev_dataset = None

    def build_output_dirs(self) -> None:
        timestamp = int(datetime.now().timestamp())
        run_name = (
            f'{self.stage}_{self.output_name}_s{self.seed}_'
            f'bs{self.batch_size}_len{self.max_length}_{timestamp}'
        )
        self.model_dir = CUSTOM_MODELS_DIR / self.output_name / self.stage / run_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def build_trainer(self) -> None:
        train_size = len(self.train_dataset)

        if self.dev_dataset is not None:
            eval_steps = max(1, math.ceil((train_size / self.batch_size) / 10))
            training_args = TrainingArguments(
                output_dir=str(self.model_dir),
                report_to='none',
                seed=self.seed,
                num_train_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                save_strategy='steps',
                save_steps=eval_steps,
                save_total_limit=2,
                logging_strategy='steps',
                logging_steps=max(1, math.ceil((train_size / self.batch_size) / 5)),
                eval_strategy='steps',
                eval_steps=eval_steps,
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                greater_is_better=True,
                bf16=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )
        else:
            training_args = TrainingArguments(
                output_dir=str(self.model_dir),
                report_to='none',
                seed=self.seed,
                num_train_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                save_strategy='epoch',
                save_total_limit=1,
                logging_strategy='steps',
                logging_steps=max(1, math.ceil((train_size / self.batch_size) / 5)),
                eval_strategy='no',
                load_best_model_at_end=False,
                bf16=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False},
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            compute_metrics=self.compute_metrics,
        )

    def train(self) -> None:
        self.trainer.train()

    def save_outputs(self) -> None:
        self.trainer.save_model(self.model_dir / 'best')
        self.tokenizer.save_pretrained(self.model_dir / 'best')
        print(f'Saved model to: {self.model_dir / "best"}')

    def run(self) -> None:
        self.set_seed()
        self.build_model()
        self.build_datasets()
        self.build_output_dirs()
        self.build_trainer()
        self.train()
        self.save_outputs()
