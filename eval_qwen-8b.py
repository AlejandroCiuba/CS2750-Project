# Experiment with setting up models for pleonasm detection
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from helper_funcs import (
    make_loggers,
    load_token,
    to_save,
    full_sentence,
    few_shot,
    )
from make_prompts import iter_prompt
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

    if args.examples != 0:
        test['examples'] = test.apply(few_shot, axis=1, args=(rest, args.examples))

    # load the tokenizer and the model
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        token=token
    )

    RUN_INFO = f"""\nRUN: {args.metadata}\n\tMODEL: {model_name}\n\tDATASET: {args.data}\n\tX-SHOT: {args.examples if args.examples != -1 else 0}\n\tTEST FOLD: {args.fold}\n\tTEMPLATE: {args.template}\n"""
    log.info(RUN_INFO)

    log.info(f"\n\tDEVICES: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")

    log.info("Model is loaded from pretrained")

    # prepare the model input

    log.info(f"Running {args.data} on {model_name}")

    # prompt = "Identify the pleonasm in the following sentence: \"We have thoughtfully thought about your situation.\" Output your response in the following JSON format: \{\"pleonasm\": \"<PLEONASM>\"\}."
    # log.info(prompt)

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

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # parsing thinking content
            index = 0
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            entry = "{\"review\": \"%s\", \"real\": [%s], \"output\": %s}" % (prompt[1], prompt[-1], content)

            # log.info(f"thinking content: {thinking_content}")
            # log.info(f"content: {content}")
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
        nargs="*",
        default="TEST RUN",
        help="Metadata used to identify the run in logs (experiment name).",
    )

    parser.add_argument(
        "-a",
        "--api",
        required=True,
        type=Path,
        help="File path to API access token for HuggingFace."
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
        prog="eval_qwen-8b.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Run SPC on QWEN-8B.",
        epilog="Created by Alejandro Ciuba, alc307@pitt.edu",
    )

    add_args(parser)
    args = parser.parse_args()

    main(args)
