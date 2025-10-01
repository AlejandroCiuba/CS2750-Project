## Split the data into training, testing and development sets
## Alejandro Ciuba, alejandrociuba@pitt.edu
from pathlib import Path
from tqdm import tqdm

import argparse
import logging
import math

import pandas as pd


def partition(data: pd.DataFrame, folds: int, column: str, seed: int, bootstrap: bool = False) -> pd.DataFrame:

    spf: int = math.ceil(len(data) / folds)
    rem: int = len(data) // args.folds

    logging.info(f"Total Samples: {len(data)}")
    logging.info(f"Samples per first {folds - 1} folds: {spf}")
    logging.info(f"Samples per fold {folds}: {rem}")

    for i in tqdm(range(folds), desc="Sampling folds..."):

        # Get samples for the fold
        sample: pd.DataFrame = data.sample(spf if i < folds - 1 else rem, random_state=seed)
        data.drop(index=sample.index, inplace=True)

        # Add column for the fold
        sample[column] = pd.Series(data=[i for _ in range(len(sample))])
        print(sample.head(1))

        logging.info(f"After fold {i}, there are {len(data)} samples to select from.")


def main(args: argparse.Namespace):

    logging.basicConfig(
        level=logging.INFO,
        handlers= [
            logging.StreamHandler(),
        ]
    )

    data: pd.DataFrame = pd.DataFrame()
    try:

        if not args.JSON:
            data = pd.read_csv(args.data)
        else:
            data = pd.read_json(args.data, lines=True)

    except Exception as e:

        logging.error(e)
        exit()

    splits: pd.DataFrame = partition(data=data, folds=args.folds, column=args.column, seed=args.seed, bootstrap=False)

    try:

        if not args.Jsave:
            splits.to_csv(args.save)
        else:
            splits.to_json(args.save, orient='records', lines=True)

    except Exception as e:

        logging.error(e)
        exit()


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Data set.\n \n",
    )

    parser.add_argument(
        "-J",
        "--JSON",
        type=bool,
        default=False,
        help="JSON format data (with lines), otherwise assumes CSV format.\n \n",
    )

    parser.add_argument(
        "-f",
        "--folds",
        type=int,
        default=5,
        help="Number of non-overlapping folds for the data.\n \n",
    )

    parser.add_argument(
        "-c",
        "--column",
        type=str,
        default="fold",
        help="Column to save the folds in, defaults to \"fold\".\n \n",
    )

    parser.add_argument(
        "-r",
        "--seed",
        type=int,
        default=1_000,
        help="Seed to control fold randomization.\n \n",
    )

    parser.add_argument(
        "-s",
        "--save",
        type=Path,
        required=True,
        help="File name to save splits in.\n \n",
    )

    parser.add_argument(
        "-S",
        "--Jsave",
        type=bool,
        default=False,
        help="JSON format data (with lines) to save splits, otherwise assumes CSV format.\n \n",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="splits.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Split the data into training, testing, and development sets.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
    