# Experiment with setting up models for pleonasm detection
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
import argparse
import logging


def main(args: argparse.Namespace):
    pass


def add_args(parser: argparse.ArgumentParser):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="test.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test to see if we can get a large model on the CRC running.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
