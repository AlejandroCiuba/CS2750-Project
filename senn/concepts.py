# Fetch the concepst for the SENN
from gensim.models import KeyedVectors
from nltk import word_tokenize

import torch

"""
Each function takes in a list of length N of sentences and outputs a Tensor
of size NxK where each row is a feature vector 1xK. Additional args may be
passed in through kwargs.
"""

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


def no_concept(sents: list[str], **kwargs) -> torch.Tensor:
    return torch.ones(kwargs['k'])


if __name__ == "__main__":

    test = [
        "Let's go to the big small world!",
        "I don't like that type of food.",
        "That's some awful disgusting smell...",]
    
    vecs = KeyedVectors.load_word2vec_format("PATH", binary=True)
    print(vecs.similarity("the", "this"))

    top_k: torch.Tensor = cosine_similarity_between_neighbors(test, embeddings=vecs, top_k=3)
    print(top_k)