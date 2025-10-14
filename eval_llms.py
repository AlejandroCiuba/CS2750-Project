# DESCRIPTION OF THE PROGRAM
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from pathlib import Path

import argparse
import json
import logging
import subprocess


def main(args: argparse.Namespace):

    logging.basicConfig(
        filename=args.save,
        level=logging.INFO,
    )

    # Get total number of lines
    entries = int(subprocess.run(f"cat {args.output} | wc -l", shell=True, executable="/bin/bash", capture_output=True).stdout.decode(encoding='ascii'))  # https://www.baeldung.com/linux/python-run-bash-command

    # Check each line of JSON
    correct = 0
    with open(args.output, 'r') as src:
        for line in src:
            try:
                obj = json.loads(s=line)
            except Exception as e:
                logging.error(e)
                continue

            if str(obj['output']['pleonasm']).strip().lower() in [str(word).strip().lower() for word in obj['real']]:
                correct += 1

    logging.info(f"Accuracy: {correct/entries:.4f}")


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="File path to the output file (curently assumes JSON).",
    )

    parser.add_argument(
        "-s",
        "--save",
        required=True,
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