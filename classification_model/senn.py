# Self-Explainable Neural Network
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from logger import make_loggers
from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

import argparse
import helper
import logging
import torch

import pandas as pd


def load_token(file: Path | str) -> str:

    try:
        with open(file=file, mode='r') as src:
            token = src.readline().strip()
    except Exception as e:
        raise e

    return token


def train(args:argparse.Namespace):
    pass



def main(args: argparse.Namespace):

    # Set up loggers and API token
    log, err = make_loggers(*args.logs, levels=[logging.INFO, logging.ERROR])

    # Fetch the device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token:str = ""
    try:
        token = load_token(args.api)
    except Exception as e:
        err.error(e, exc_info=True)
        exit()

    # Set up the datasets
    data = pd.read_csv(args.data)
    train_df, test_df = None, None

    if args.fold == -1:
        train_df = data[data["fold"] != data["fold"].max()]
        test_df = data[data["fold"] == data["fold"].max()]
    else:
        train_df = data[data["fold"] != int(args.fold)]
        test_df = data[data["fold"] == int(args.fold)]

    train_df = train_df[["sentence", "pleonasm"]]
    test_df = test_df[["sentence", "pleonasm"]]

    train_df.replace(to_replace=pd.NA, value="", inplace=True)
    test_df.replace(to_replace=pd.NA, value="", inplace=True)

    RUN_INFO = f"""\nRUN: {args.metadata}\n\tMODEL: {args.name}\n\tDATASET: {args.data}\n\tTEST FOLD: {args.fold if args.fold != -1 else int(data["fold"].max())}\n\tTRAINING SIZE: {len(train_dataset)}\n\tEPOCHS: {args.epochs}\n\tBATCH SIZE: {args.batch_size}\n\tLEARNING RATE: {args.learning_rate}"""
    log.info(RUN_INFO)

    log.info(f"\n\tDEVICES: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")

    # Set up model
    model, tokenizer = helper.fetch_model(args.name, token)
    model.to(DEVICE)

    optim = AdamW(params=model.parameters, lr=args.learning_rate)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_training_steps=int(0.1 * )
    )

    train(args)



def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default="TEST RUN",
        help="Metadata used to identify the run in logs (experiment name).",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="ENCDEC",
        help="Model name.",
    )

    parser.add_argument(
        "-a",
        "--api",
        required=True,
        type=Path,
        help="Metadata used to identify the run in logs (experiment name).",
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=Path,
        help="File path to the CSV-styled corpus."
    )

    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        default=-1,
        help="Test fold, defaults to -1 to test on the last fold."
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=60,
        help="Training epochs, defaults to 60."
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per forward pass, defaults to 32."
    )

    parser.add_argument(
        "-r",
        "--learning_rate",
        type=float,
        default=1E-3,
        help="Batch size per forward pass, defaults to 32."
    )

    parser.add_argument(
        "-l",
        "--logs",
        required=True,
        type=Path,
        nargs=2,
        help="File path to the general log and error log files."
    )

    parser.add_argument(
        "-s",
        "--save",
        required=True,
        type=Path,
        help="File path to save the test output at."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="senn.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run the Self-Explainable Neural Network.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)