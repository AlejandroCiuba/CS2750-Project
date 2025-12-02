from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
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

        self.X_enc = self.tokenizer(
            self.X,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        self.y_enc = self.key(self.y)

        # self.max_sequence_len_X = len(max(self.X_enc, key=len))
        # self.max_sequence_len_y = len(max(self.y_enc, key=len))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):

        X = self.X_enc[index] + [self.tokenizer.w2i[self.tokenizer._EOS]]
        X.extend([0 for _ in range(self.max_sequence_len_X - len(X) + 1)])  # Add one to prevent edge case since max does not consider EOS token

        y = self.y_enc[index] + [self.tokenizer.w2i[self.tokenizer._EOS]]
        y.extend([0 for _ in range(self.max_sequence_len_y - len(y) + 1)])

        return torch.LongTensor(X).to(device=self.device), torch.tensor(y).to(device=self.device)
    
    def encode(self, sents: list[str]) -> list:
        return self.tokenizer.encode(sents)
    
    def decode(self, inds: list[list]) -> list:
        return self.tokenizer.decode(inds)


def fetch_model(name: str, token: str) \
-> tuple[PreTrainedModel, PreTrainedTokenizerFast | PreTrainedTokenizer]:

    if name == "distilbert-base-uncased":
        return DistilBertForSequenceClassification.from_pretrained(name, token=token, num_labels=2),
    DistilBertTokenizerFast.from_pretrained(name, token=token)


if __name__ == "__main__":

    data = pd.read_csv("../data/SPC-CSV/SPC.csv")[["sentence", "pleonasm"]].replace(to_replace=pd.NA, value="")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    test = BinaryLangDataset(
        target_pairs=data,
        key=lambda y: 0 if y == "" else 1,
        tokenizer=tokenizer
    )
