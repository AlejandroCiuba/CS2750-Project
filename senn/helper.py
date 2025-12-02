from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)

import torch

import pandas as pd


class BinaryLangDataset(Dataset):

    def __init__(self, target_pairs: pd.DataFrame, key,
                  tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
                  max_len: int = 128, device="cpu"):

        super().__init__()

        self.max_len = max_len
        self.device = device
        self.key = key

        self.X = target_pairs.iloc[:, 0].to_list()
        self.y = target_pairs.iloc[:, 1].to_list()

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, ind):
        enc = self.tokenizer(
            self.X[ind],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0).to(self.device) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.key(self.y[ind]), dtype=torch.long).to(self.device)
        return item


def fetch_model(name: str, token: str) \
-> tuple[PreTrainedModel, PreTrainedTokenizerFast | PreTrainedTokenizer]:

    if name == "distilbert-base-uncased":
        return DistilBertForSequenceClassification.from_pretrained(name, token=token, num_labels=2), \
            DistilBertTokenizerFast.from_pretrained(name, token=token)
    elif name == "roberta-base":
        return RobertaForSequenceClassification.from_pretrained(name, token=token, num_labels=2), \
            RobertaTokenizerFast.from_pretrained(name, token=token)

def load_token(file: Path | str) -> str:

    try:
        with open(file=file, mode='r') as src:
            token = src.readline().strip()
    except Exception as e:
        raise e

    return token


if __name__ == "__main__":

    data = pd.read_csv("PATH")[["X", "y"]].replace(to_replace=pd.NA, value="")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    test = BinaryLangDataset(
        target_pairs=data,
        key=lambda y: 0 if y == "" else 1,
        tokenizer=tokenizer
    )

    print(test.__getitem__(0))
