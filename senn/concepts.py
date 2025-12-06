# Fetch the concepst for the SENN
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import pos_tag
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast,
    RobertaModel,
    RobertaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)

import concurrent.futures
import string
import torch

"""
Each function takes in a list of length N of sentences and outputs a Tensor
of size NxK where each row is a feature vector 1xK. Additional args may be
passed in through kwargs.
"""

stop = stopwords.words('english')

def cosine_similarity_between_neighbors(sents: list[str], **kwargs) -> torch.Tensor:
    """
    `embeddings`: KeyedVector object from gensim
    `top_k`: Top k cosine similarities.
    Does not move them to the current device.
    """
    tok_sents = [word_tokenize(sent) for sent in sents]

    tops = []
    for sent in tok_sents:
        top = []
        for ind in range(len(sent)):
            if ind != len(sent) - 1:
                try:
                    top.append(kwargs['embeddings'].similarity(sent[ind], sent[ind + 1]))
                except:
                    continue

        if len(top) < kwargs['top_k']:
            top.extend([0.0 for _ in range(kwargs['top_k'] - len(top))])

        tops.append(sorted(top, reverse=True)[:kwargs['top_k']])

    return torch.tensor(tops).squeeze()


def cosine_similarity_sentence_top(sents: list[str], **kwargs) -> torch.Tensor:
    """
    embeddings: BERT model to get sentence embeddings
    tokenizer: The BERT model's tokenizer
    device: The device to have the BERT model on
    The embeddings will be on the BERT model device.
    """

    def _masking(sent: str) -> tuple[str, list[str]]:
        words_to_mask = [tok[0] for tok in pos_tag(word_tokenize(sent)) if tok[1] in ["JJ", "NN"]]
        with concurrent.futures.ThreadPoolExecutor() as exec:
            masked = exec.map(lambda tok: sent.replace(tok, ""), words_to_mask)
        return sent, list(masked)
    

    def _get_cosines(sent: tuple[str, list[str]]):

        def _get_embedding(input: str) -> torch.Tensor:
            output = kwargs["tokenizer"](
                input,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
                )
            with torch.no_grad():
                return kwargs["embeddings"](**output).last_hidden_state.mean(dim=-1)  # https://www.geeksforgeeks.org/nlp/how-to-generate-word-embedding-using-bert/

        base_emb: torch.Tensor = _get_embedding(sent[0])
        alt_embs: torch.Tensor = _get_embedding(sent[1])

        return torch.nn.functional.cosine_similarity(base_emb, alt_embs, dim=-1).max().unsqueeze(dim=-1)


    with concurrent.futures.ThreadPoolExecutor() as exec:
        future = exec.map(lambda sent: _get_cosines(_masking(sent)), sents)


    return torch.concat(tuple(future), dim=-1).unsqueeze(dim=-1)


def no_concept(sents: list[str], **kwargs) -> torch.Tensor:
    return torch.ones(kwargs['k'])


if __name__ == "__main__":

    test = [
        "Let's go to the big small world!",
        "I don't like that type of food.",
        "That's some awful disgusting smell...",]
    
    # vecs = KeyedVectors.load_word2vec_format("PATH", binary=True)
    # print(vecs.similarity("the", "this"))

    # top_k: torch.Tensor = cosine_similarity_between_neighbors(test, embeddings=vecs, top_k=3)
    # print(top_k)

    embeddings = DistilBertModel.from_pretrained("distilbert-base-uncased", num_labels=2)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    sent_sim: torch.Tensor = cosine_similarity_sentence_top(test, embeddings=embeddings, tokenizer=tokenizer, device="cpu")
    print(sent_sim)