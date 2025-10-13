# Generic logging setup
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path
from typing import Union

import argparse
import logging

import pandas as pd

StrPath = Union[str, Path]

def make_loggers(*args: StrPath, levels: Union[list[int], int]) \
    -> tuple[logging.Logger]:

    def make_logger(path, level, i) -> logging.Logger:

        logger = logging.getLogger(f"log-{i}")
        logger.setLevel(level)

        handler = logging.FileHandler(path)
        handler.setFormatter(fmt)

        logger.addHandler(handler)

        return logger

    # Set up logger
    # version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    bundle = zip(args, levels) if isinstance(levels, list) else zip(args, [levels] * len(args))

    return tuple([make_logger(path, level, i) for i, (path, level) in enumerate(bundle)])


def load_token(file: Path | str) -> str:

    try:
        with open(file=file, mode='r') as src:
            token = src.readline().strip()
    except Exception as e:
        raise e

    return token


def to_save(file: Path | str, contents: str | list[str], overwrite: bool = False):

    try:
        with open(file=file, mode="w" if overwrite else "a") as dest:
            if isinstance(contents, list):
                dest.writelines(contents)
            elif isinstance(contents, str):
                dest.write(f"{contents}\n")
            else:
                raise TypeError(f"contents is of type {type(contents)}, not str or list")
            
    except Exception as e:
            raise e


def full_sentence(row: pd.Series) -> list[str]:
    before, after, pleonasm = row.before, row.after, row.consensus
    if pleonasm == 'neither' or pleonasm == 'both':
        return f"{before.strip()} {after.strip()}"
    else:
        return f"{before.strip()} {pleonasm.strip()} {after.strip()}"
    
# Cannot just get sentences from the 'review' column as we need to get both parts if there are two pleonasms.
def few_shot(row: pd.Series, sample_pool: pd.DataFrame, examples: int = 4):

    samples = sample_pool.sample(examples)

    preamble = "In the sentence:"
    fewshot = """"""
    for sample in samples.itertuples():

        before, after, pleonasm = sample.before, sample.after, sample.consensus
        if pleonasm == 'neither':
            fewshot += f"'''{preamble} \"{before} {after}\", there are no pleonasms.'''\n"
        elif pleonasm == 'both':
            fewshot += f"'''{preamble} \"{before} {after}\", the pleonasms are {before.split(' ')[-1]} and {after.split(' ')[0]}.'''\n"
        else:
            fewshot += f"'''{preamble} \"{before} {pleonasm} {after}\", the pleonasm is {pleonasm}.'''\n"

    return fewshot.strip()