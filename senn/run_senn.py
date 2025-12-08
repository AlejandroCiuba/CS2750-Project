# Self-Explainable Neural Network
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from gensim.models import KeyedVectors
from logger import make_loggers
from pathlib import Path
from senn import SENN
from sklearn.metrics import classification_report
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
)
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

import argparse
import concepts
import helper
import logging
import textwrap
import torch

import pandas as pd


def train(model,
          optim: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LambdaLR,
          encoder, optim_enc: torch.optim.Optimizer, crit,
          data: DataLoader, epochs: int,
          logger: logging.Logger):

    for epoch in tqdm(range(1, epochs + 1), desc="Training..."):
        tl = 0.0
        tle = 0.0
        for batch, cvec in data:

            # Train the parameterizer
            optim.zero_grad()
            outputs = model(**batch)
            # loss = outputs.loss
            # loss.backward()
            
 

            # Train the encoder
            optim_enc.zero_grad()
            outputs, _, _ = encoder(cvec, outputs.logits)
            loss_enc = crit(outputs, batch['labels'].float())
            loss_enc.backward()
            optim.step()
            optim_enc.step()
            # sched.step()
        
        # tl += loss.item()
        tle += loss_enc.item()
        logger.info(f"{epoch}/{epochs}: ({tle / len(data):0.5f} {loss_enc.item()})")


def evaluate(model, encoder, data: DataLoader, logger: logging.Logger):
    with torch.no_grad():
        preds, true = [], []
        for batch, cvec in tqdm(data, desc="Evaluating..."):

            outputs = torch.nn.functional.sigmoid(encoder(cvec, model(**batch).logits)[0])

            pred:torch.Tensor = torch.where(outputs >= 0.5, 1.0, 0.0)  # topk produces a list [probabilities, inds]
            preds.extend(pred.cpu().numpy())
            true.extend(batch['labels'].cpu().numpy())

    logger.info(f"\n{classification_report(true, preds, digits=5)}")


def fetch_gensim(path: Path) -> KeyedVectors:
    return KeyedVectors.load_word2vec_format(path, binary=True)


def main(args: argparse.Namespace):

    # Set up loggers and API token
    log, err = make_loggers(*args.logs, levels=[logging.INFO, logging.ERROR])

    # Fetch the device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token:str = ""
    try:
        if args.api is not None:
            token = helper.load_token(args.api)
    except Exception as e:
        err.error(e, exc_info=True)
        exit()

    # Set up the datasets
    data = pd.read_csv(args.data)
    data.replace(to_replace=pd.NA, value="", inplace=True)
    train_df, test_df = None, None

    if args.fold == -1:
        train_df = data[data["fold"] != data["fold"].max()]
        test_df = data[data["fold"] == data["fold"].max()]
    else:
        train_df = data[data["fold"] != int(args.fold)]
        test_df = data[data["fold"] == int(args.fold)]

    train_df = train_df[["sentence", "pleonasm"]]
    test_df = test_df[["sentence", "pleonasm"]]

    # Set up model
    model, tokenizer = helper.fetch_model(args.name, token)
    model.to(DEVICE)

    encoder = SENN(concept_dim=3, emb_size=2, hidden=8)
    encoder.to(DEVICE)

    try:
        vecs = fetch_gensim(args.gensim)
    except Exception as e:
        err.error(e, exc_info=True)
        exit()    

    log.info("GENSIM MODEL LOADED...")

    # Set up datasets

    concept = concepts.cosine_similarity_between_neighbors
    concept_args = {
        'embeddings': vecs,
        'top_k': 3,
    }

    train_dataset = helper.BinLangDatasetSENN(
        train_df, key=lambda y: 0.0 if y == "" else 1.0, 
        tokenizer=tokenizer, device=DEVICE,
        concept=concept, concept_args=concept_args, concept_size=3,
        )
    test_dataset = helper.BinLangDatasetSENN(
        test_df, key=lambda y: 0.0 if y == "" else 1.0, 
        tokenizer=tokenizer, device=DEVICE,
        concept=concept, concept_args=concept_args, concept_size=3,
        )

    train_dl = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size)  

    optim = AdamW(params=model.parameters(), lr=args.learning_rate)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_training_steps=len(train_dl) * args.epochs,
        num_warmup_steps=0.1 * len(train_dl) * args.epochs,
    )

    optim_enc = AdamW(params=encoder.parameters(), lr=args.learning_rate)
    criterion_enc = torch.nn.BCEWithLogitsLoss()

    RUN_INFO = textwrap.dedent(
        f"""
        \nRUN: {args.metadata}
        \tMODEL: {args.name}
        \tDATASET: {args.data}
        \tTEST FOLD: {args.fold if args.fold != -1 else int(data["fold"].max())}
        \tTRAINING SIZE: {len(train_dataset)}
        \tEPOCHS: {args.epochs}
        \tBATCH SIZE: {args.batch_size}
        \tLEARNING RATE: {args.learning_rate}"""
    )
    log.info(RUN_INFO)
    log.info(f"\tDEVICES: {' '.join([str(torch.cuda.device(i) )for i in range(torch.cuda.device_count())])}")

    try:
        train(model=model, optim=optim, sched=sched,
              encoder=encoder, optim_enc=optim_enc, crit=criterion_enc,
            data=train_dl, epochs=args.epochs, logger=log)
    except Exception as e:
        err.error(e, exc_info=True)
        exit()    
    
    try:
        evaluate(model=model, encoder=encoder, data=test_dl, logger=log)
    except Exception as e:
        err.error(e, exc_info=True)
        exit()


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
        type=Path,
        default=None,
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
        "-c",
        "--concept",
        type=str,
        default="NONE",
        help="Concept to apply within the SENN; NONE will result in the addition of two more linear layers.",
    )

    parser.add_argument(
        "-g",
        "--gensim",
        type=Path,
        required=True,
        help="Path to the gensim model for word vectors.",
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
        default=1E-2,
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

    # parser.add_argument(
    #     "-s",
    #     "--save",
    #     required=True,
    #     type=Path,
    #     help="File path to save the test output at."
    # )


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