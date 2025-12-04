# Experiment with setting up models for pleonasm detection
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from baselines import (
    load_model,
    prompt_model
)
from gensim.models import KeyedVectors
from helper_funcs import (
    make_loggers,
    load_token,
    to_save,
    )
from make_prompts import (
    iter_prompt,
    full_sentence,
    few_shot,
    highest_cosine_similarity_between_neighbors
    )
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    )

import argparse
import datetime
import logging
import torch

import pandas as pd


def main(args: argparse.Namespace):

    # Set up loggers and API token
    log, err = make_loggers(*args.logs, levels=[logging.INFO, logging.ERROR])

    token:str = ""
    try:
        token = load_token(args.api)
    except Exception as e:
        err.error(e, exc_info=True)
        exit()

    # Set up dataset
    data = pd.read_json(args.data, lines=True)

    rest = data[data['fold'] != args.fold]
    test = data[data['fold'] == args.fold].copy()

    test['review'] = test.apply(full_sentence, axis=1)
    test['task'] = test.apply(lambda x: "Identify the pleonasm in the customer review, write NONE if there are no pleonasms.", axis=1)
    test['format'] = test.apply(lambda x: "{\"pleonasm\": \"<WORD>\"}", axis=1)

    vecs:KeyedVectors | None = None
    if args.linguistic_feature in ["highest"]:
        try:
            vecs = KeyedVectors.load_word2vec_format(args.linguistic_path, binary=True)
        except Exception as e:
            err.error(e, exec_info=True)
            exit()
        test['linguistic_feature'] = test.apply(highest_cosine_similarity_between_neighbors, axis=1, embeddings=vecs)

    if args.examples != 0:
        if args.linguistic_feature == "none":
            test['examples'] = test.apply(few_shot, axis=1, sample_pool=rest, examples=4)
        if args.linguistic_feature == "highest":
            if vecs is None:
                err.error("The option \"highest\" was given with no KeyedVector model path. Aborting...")
                exit()
            test['examples'] = test.apply(few_shot, axis=1, sample_pool=rest, examples=4, 
                                          linguistic_feature=highest_cosine_similarity_between_neighbors, lf_kwargs={"embeddings": vecs})
            del vecs

    # load the tokenizer and the model
    try:
        model, tokenizer = load_model(name=args.name, token=token)
    except Exception as e:
        err.error(e, exec_info=True)
        exit()

    RUN_INFO = f"""\nRUN: {args.metadata}\n\tMODEL: {args.name}\n\tDATASET: {args.data}\n\tX-SHOT: {args.examples if args.examples != -1 else 0}\n\tTEST FOLD: {args.fold}\n\tTEMPLATE: {args.template}\n"""
    log.info(RUN_INFO)
    log.info(f"\n\tDEVICES: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")

    log.info("Model is loaded from pretrained")

    # prepare the model input

    log.info(f"Running {args.data} on {args.name}")

    # Evaluate model
    try:
        for prompt in tqdm(iter_prompt(template=args.template, data=test), desc="Running model on corpus"):

            messages = [
                {"role": "user", "content": prompt[0]}
            ]

            if args.examples != 0:
                log.info(f"REVIEW: {prompt[1]}\nEXAMPLES: {prompt[2]}\n")
            else:
                log.info(f"REVIEW: {prompt[1]}")


            content = prompt_model(name=args.name, messages=messages, model=model, tokenizer=tokenizer)

            entry = "{\"review\": \"%s\", \"real\": [%s], \"output\": %s}" % (prompt[1], prompt[-1], content)

            to_save(args.save, entry, overwrite=False)

    except Exception as e:
        err.error(e, exc_info=True)
        exit()

    log.info(f"SAVED TO: {args.save}")
    log.info(f"TIME COMPLETED: {datetime.datetime.now()}")


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
        required=True,
        type=str,
        help="Model name.",
    )

    parser.add_argument(
        "-a",
        "--api",
        required=True,
        type=Path,
        help="File path to API access token for HuggingFace."
    )

    parser.add_argument(
        "-x",
        "--linguistic_feature",
        type=str,
        default="none",
        help="The linguistic concept to give the model; defaults to \"none\"."
    )

    parser.add_argument(
        "-v",
        "--linguistic_path",
        type=Path,
        default=None,
        help="Path to the file containing whatever is needed to get the linguistic features."
    )

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=Path,
        help="File path to the JSON-styled corpus."
    )

    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        default=-1,
        help="Test fold, defaults to -1 to test on all folds."
    )

    parser.add_argument(
        "-e",
        "--examples",
        type=int,
        default=0,
        help="If running few-shot, is the number of folds used as examples."
    )

    parser.add_argument(
        "-t",
        "--template",
        required=True,
        type=Path,
        help="File path to the text file containing the model prompt."
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
        help="File path to save LLM output at."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="run_baseline.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run SPC on the specified model.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
