# Evaluate the LLMs.
# NOTE: While we have the real answer in the original run_baselines.py output, we changed how we evaluated
# the models (now matching any of the pleonasms, not just consensus, counts). Therefore, we need the data
# and fold information.
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from pathlib import Path

import argparse
import json
import logging
import subprocess

import pandas as pd

def main(args: argparse.Namespace):

    logging.basicConfig(
        filename=args.save,
        level=logging.INFO,
    )

    # Get total number of lines
    entries = int(subprocess.run(f"cat {args.output} | wc -l", shell=True, executable="/bin/bash", capture_output=True).stdout.decode(encoding='ascii'))  # https://www.baeldung.com/linux/python-run-bash-command
    data = pd.read_json(args.data, lines=True)
    data = data[data["fold"] == args.fold]

    # Check each line of JSON
    count = 0
    correct = 0
    binary_correct = 0  # Whether or not it guess that there is a pleonasm
    with open(args.output, 'r') as src:
        for line in src:
            try:
                obj = json.loads(s=line)
                row: pd.Series = data.iloc[count, :]
                if row['consensus'].strip().lower() != "neither":
                    if str(obj['output']['pleonasm']).strip().lower() in [str(row['first']).strip().lower(), str(row['second']).strip().lower()]:
                        correct += 1

                if (str(obj['output']['pleonasm']).strip().lower() == "none" and "".join(obj['real']).strip().lower() == "none") \
                or (str(obj['output']['pleonasm']).strip().lower() != "none" and "".join(obj['real']).strip().lower() != "none"):
                    binary_correct += 1
                count += 1
            except Exception as e:
                logging.error(e)

    logging.info(f"Accuracy: {correct/entries:.4f}\nBinary Accuracy: {binary_correct/entries:.4f}")


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="File path to the output file (curently assumes JSON).",
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=Path,
        help="File path to the original data (currentl assumes JSON).",
    )

    parser.add_argument(
        "-f",
        "--fold",
        required=True,
        type=int,
        help="Fold to examine.",
    )

    parser.add_argument(
        "-s",
        "--save",
        required=False,
        type=Path,
        help="File path to save file.",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="eval_llms.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Evaluate the output from the LLMs.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)