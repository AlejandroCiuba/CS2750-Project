# The model and associated dataset type
# from nltk import word_tokenize
from torch.utils.data import (
    Dataset,
    DataLoader,
)

import math
import torch

import pandas as pd
import torch.nn as nn


class Tokenizer:
    """
    Custom tokenizer for the LangDataset class.
    """

    _SOS = "SOS"
    _EOS = "EOS"
    _UNK = "UNK"

    def __init__(self, preproc, remove_empty = True):
        """
        `remove_empty will automatically remove any word that is just the empty string if set to `True`.
        """

        self.w2i = {self._SOS: 0, self._EOS: 1, self._UNK: 2}
        self.i2w = {0: self._SOS, 1: self._EOS, 2: self._UNK}
        self.wc = {self._SOS: 1, self._EOS: 1, self._UNK: 1}
        self.ws = 3  # Unique
        self.preproc = preproc
        self.remove_empty = remove_empty

    def __len__(self) -> int:
        return self.ws

    def _tok_sent(self, sent: str):
        for word in sent.split(' '):
            word = self.preproc(word)
            if word == "" and self.remove_empty:
                continue
            if word not in self.w2i:
                self.w2i[word] = self.ws
                self.i2w[self.ws] = word
                self.wc[word] = 1
                self.ws += 1
            else:
                self.wc[word] += 1

    def tokenize(self, sents: str | list[str]):
        if isinstance(sents, str):
            self._tok_sent(sents)
        elif isinstance(sents, list):
            for sent in sents:
                self._tok_sent(sent)

    def encode(self, sents: list[list]):
        encodeds = []
        for sent in sents:
            encoded_sent = []
            for word in sent:
                if self.preproc(word) == "" and self.remove_empty:
                    continue
                elif self.preproc(word) in self.w2i:
                    encoded_sent.append(self.w2i[self.preproc(word)])
                else:
                    encoded_sent.append(self.w2i[self._UNK])
            encodeds.append(encoded_sent)
        return encodeds

    def decode(self, inds: list[list]):
        return [[self.i2w[ind] if ind < len(self.i2w) else self._UNK for ind in sent] for sent in inds]
 

class LangDataset(Dataset):

    def __init__(self, target_pairs: pd.DataFrame, tokenizer: Tokenizer, train: bool = True, device="cpu"):

        super().__init__()

        self.train = train
        self.device = device

        self.X = target_pairs.iloc[:, 0].to_list()
        self.y = target_pairs.iloc[:, 1].to_list()

        self.tokenizer = tokenizer

        if train:
            tokenizer.tokenize(self.X)
            tokenizer.tokenize(self.y)

        self.X_enc = tokenizer.encode(map(lambda x: x.split(' '), self.X))
        self.y_enc = tokenizer.encode(map(lambda x: x.split(' '), self.y))

        self.max_sequence_len_X = len(max(self.X_enc, key=len))
        self.max_sequence_len_y = len(max(self.y_enc, key=len))

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


class Encoder(nn.Module):

    def __init__(self, input_size: int = 50, embedding_size: int = 100, hidden_size: int = 40, layers: int = 1, dropout: float = 0.1, device: str = "cpu"):

        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.device = device

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        return self.gru(self.dropout(self.embedding(input)))


class Attention(nn.Module):

    def __init__(self, hidden_size: int, device: str = "cpu"):

        super().__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.K = nn.Linear(self.hidden_size, self.hidden_size)
        self.Q = nn.Linear(self.hidden_size, self.hidden_size)
        self.V = nn.Linear(self.hidden_size, 1)

        self.dim_k = self.hidden_size
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, queries, keys):

        # Luong attention
        # weights = torch.matmul(self.softmax(torch.matmul(self.Q[queries], torch.transpose(self.K[keys])) * (1 / math.sqrt(self.dim_k))), self.V).squeeze(2).unsqueeze(1)
        # context = torch.bmm(weights, keys)

        # Additive attention
        scores = self.V(self.tanh(self.Q(queries) + self.K(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = self.softmax(scores)
        context = torch.bmm(weights, keys)

        return context, weights


class Decoder(nn.Module):

    def __init__(self, output_size: int, hidden_size: int, dropout: float = 0.1, max_seq_len: int = 100, device: str = "cpu"):

        super().__init__()
        self.embedding_size = hidden_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)  # Hidden size is also the embeddings size
        self.attention = Attention(self.hidden_size)
        self.attention.to(self.device)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.logits = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input, hidden, encoder_outputs):

        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, att_weights = self.attention(query, encoder_outputs)

        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, att_weights

    def forward(self, enc_outputs, enc_hidden, target_tensor = None):

        batch_size = enc_outputs.size(0)
        dec_input = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)  # Start with the SOS token
        dec_hidden = enc_hidden

        dec_outputs = []
        atts = []

        for i in range(self.max_seq_len):

            dec_output, dec_hidden, att = self.forward_step(dec_input, dec_hidden, enc_outputs)
            dec_outputs.append(dec_output)
            atts.append(att)

            # Teacher forcing forces the correct target as the next input
            if target_tensor is not None:
                dec_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, inds = dec_output.topk(1)
                dec_input = inds.squeeze(-1).detach()

        dec_outputs = torch.cat(dec_outputs, dim=1)
        dec_outputs = self.logits(dec_outputs)
        atts = torch.cat(atts, dim=1)

        return dec_outputs, dec_hidden, atts


if __name__ == "__main__":

    import re

    pattern = re.compile(r'[^a-zA-Z\d\']+', re.IGNORECASE)
    def preproc(x: str):
        x = x.strip().lower()
        x = pattern.sub("", x)
        return x

    test = ["this is a test sentence.",
            "This is @nother 123!",]
    
    test_tok = Tokenizer(preproc)
    test_tok.tokenize(sents=test)
    print(test_tok.w2i)

    print(test_tok.decode([[1,4,2,3],[0,2,4,1]]))

    test_df = pd.read_csv("DATA-PATH")
    test_df = test_df[["X", "y"]]
    test_df.replace(to_replace=pd.NA, value="", inplace=True)
    
    test_tok = Tokenizer(preproc=preproc, remove_empty=False)
    test_tok.tokenize(test_df["X"].to_list())
    test_tok.tokenize(test_df["y"].to_list())

    print(test_df.head(1))

    dataset = LangDataset(test_df, test_tok)

    print(test_df.iloc[0, :])
    print(dataset.__getitem__(0))

    no_ans = test_df[test_df["y"] == ""].index[0]
    print(test_df.iloc[no_ans, :])
    print(dataset.__getitem__(no_ans))
    print(test_tok.w2i[""])