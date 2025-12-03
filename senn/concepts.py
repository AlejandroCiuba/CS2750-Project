# Fetch the concepst for the SENN
import spacy

import torch

"""
Each function takes in a list of length N of sentences and outputs a Tensor
of size NxK where each row is a feature vector 1xK. Additional args may be
passed in through kwargs.
"""

def concepts(sents: list[str], **kwargs) -> torch.Tensor:
    pass