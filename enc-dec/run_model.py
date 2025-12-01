# Runs the encoder decoder model for pleonasm detection
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from enc_dec import (
    Tokenizer,
    LangDataset,
    Encoder,
    Attention,
    Decoder,
)
from logger import make_loggers
from pathlib import Path
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
)

import argparse
import logging
import re
import torch

import pandas as pd
import torch.nn as nn


def create_model(vocab_size: int, embedding_size: int, 
                 encoder_hidden_size: int, encoder_layers: int,
                 max_sequence_length: int, device: str) -> tuple[nn.Module, nn.Module]:
    return Encoder(vocab_size, embedding_size, encoder_hidden_size, encoder_layers, device=device).to(device),\
        Decoder(vocab_size, embedding_size, max_seq_len=max_sequence_length, device=device).to(device)  # Sequence length is three at max


def train(encoder: Encoder, decoder: Decoder, 
          encoder_optimizer: torch.optim.Optimizer, decoder_optimizer: torch.optim.Optimizer,
          training_data: DataLoader, criterion, epochs: int,
          logger: logging.Logger) -> float:
    
    MODEL_INFO = f"""\n\tVOCAB SIZE: {len(training_data.dataset.tokenizer)}\n\tEMBEDDING SIZE: {128}\n\tHIDDEN SIZE: {128}\n\tLAYERS: {1}\n\tMAX SEQUENCE LENGTH: {training_data.dataset.max_sequence_len_X}"""
    logger.info(MODEL_INFO)
    
    total_loss = 0.0

    for epoch in range(1, epochs + 1):

        total_batch_loss = 0.0

        for batch in training_data:

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(batch[0])
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, batch[1])

            loss = criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), batch[1].view(-1))
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_batch_loss += loss.item()

        epoch_loss = total_batch_loss / len(training_data)  # Average loss
        total_loss += epoch_loss

        # Write it down
        logger.info(f"{epoch}/{epochs}: {epoch_loss:.5f} {total_loss / epoch:.5f}")


def evaluate(encoder: Encoder, decoder: Decoder, testing_data: DataLoader, logger: logging.Logger, log_examples: bool = True):

    logger.info(f"TESTING SIZE: {len(testing_data.dataset)}")

    with torch.no_grad():

        total_acc: float = 0.0
        for i, batch in enumerate(testing_data):

            encoder_outputs, encoder_hidden = encoder(batch[0])
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

            _, inds = decoder_outputs.topk(1)
            inds:list[list[int]] = inds.squeeze().tolist()

            output: list[set[str]] = list(map(lambda sent: set(filter(lambda x: True if x != "EOS" and x != "SOS" and x != "UNK" else False, sent)), testing_data.dataset.tokenizer.decode(inds)))
            correct: list[set[str]] = list(map(lambda sent: set(filter(lambda x: True if x != "EOS" and x != "SOS" and x != "UNK" else False, sent)), testing_data.dataset.tokenizer.decode(batch[1].tolist())))

            # output[0] = correct[0].copy()
            # output[0].remove('plain')
            # output[1] = set()
            truth_vector: list[bool] = list(map(lambda x: 1 if (x[0] != set() and x[0] & x[1] != set()) or (x[0] == set() and x[1] == set()) else 0, zip(correct, output)))
            # logger.info(list(output))
            # logger.info(list(correct))
            # logger.info(truth_vector)
            logger.info(f"{i + 1}/{len(testing_data)}: {sum(truth_vector) / len(correct):.4f}")
            total_acc += (sum(truth_vector) / len(correct))

            if log_examples:
                logger.info(f"\tOUTPUT EXAMPLES: {list(output)}")
                logger.info(f"\tTARGETS:         {list(correct)}")

    logger.info(f"TOTAL ACCURACY: {total_acc / len(testing_data.dataset):.4f}")


def main(args: argparse.Namespace):

    # Set up loggers and API token
    log, err = make_loggers(*args.logs, levels=[logging.INFO, logging.ERROR])

    # Fetch the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Preprocessing
    pattern = re.compile(r'[^a-zA-Z\d\']+', re.IGNORECASE)
    def preproc(x: str):
        x = x.strip().lower()
        x = pattern.sub("", x)
        return x

    train_dataset = LangDataset(train_df, tokenizer=Tokenizer(preproc=preproc), device=device)
    test_dataset = LangDataset(test_df, tokenizer=train_dataset.tokenizer, train=False, device=device)

    train_dl = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size)

    RUN_INFO = f"""\nRUN: {args.metadata}\n\tMODEL: {args.name}\n\tDATASET: {args.data}\n\tTEST FOLD: {args.fold if args.fold != -1 else int(data["fold"].max())}\n\tTRAINING SIZE: {len(train_dataset)}\n\tEPOCHS: {args.epochs}\n\tBATCH SIZE: {args.batch_size}\n\tLEARNING RATE: {args.learning_rate}"""
    log.info(RUN_INFO)

    log.info(f"\n\tDEVICES: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")

    # Load the model
    encoder, decoder = create_model(len(train_dataset.tokenizer), 128, 128, 1, train_dataset.max_sequence_len_y + 1, device)  # Add one to include the EOS token...

    enc_opt = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

    crit = nn.NLLLoss()

    # train
    try:
        train(encoder=encoder, decoder=decoder,
            encoder_optimizer=enc_opt, decoder_optimizer=dec_opt,
            training_data=train_dl, criterion=crit,
            epochs=args.epochs, logger=log)
    except Exception as e:
        err.error(f"{e}")
        exit()

    log.info("TRAINING COMPLETE...")

    try:
        evaluate(encoder=encoder, decoder=decoder,
                testing_data=test_dl, logger=log)
    except Exception as e:
        err.error(f"{e}")
        exit()
    
    log.info("SAVING MODEL...")

    try:
        torch.save(encoder.state_dict(), args.save / f"{args.name}-{args.metadata}-{args.fold}-enc.pt")
        torch.save(decoder.state_dict(), args.save / f"{args.name}-{args.metadata}-{args.fold}-dec.pt")
        torch.save(decoder.attention.state_dict(), args.save / f"{args.name}-{args.metadata}-{args.fold}-att.pt")
    except Exception as e:
        err.error(f"{e}")
        exit()

    log.info("DONE")



def add_args(parser: argparse.ArgumentParser):
    
    parser.add_argument(
        "-m",
        "--metadata",
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
        prog="run_model.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Runs the encoder decoder model for pleonasm detection.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)