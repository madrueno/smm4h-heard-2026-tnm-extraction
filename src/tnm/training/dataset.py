"""Dataset loading for TNM staging experiments."""

import torch


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict[str, list[int]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)
