# Encoder decoder model for pleonasm detection
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
import argparse
import logging


def main(args: argparse.Namespace):
    pass


def add_args(parser: argparse.ArgumentParser):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="enc-dec.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Encoder decoder model for pleonasm detection.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)