# Experiment with setting up models for pleonasm detection
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoProcessor, 
    BitsAndBytesConfig,
    Llama4ForConditionalGeneration,
    )
from logger import make_loggers

import argparse
import datetime
import logging
import torch
import transformers


def load_token(file: Path | str) -> str:

    try:
        with open(file=file, mode='r') as src:
            token = src.readline().strip()
    except Exception as e:
        raise e

    return token


def main(args: argparse.Namespace):

    log, err = make_loggers("logs/runs/test-run.log", "logs/errors/test-err.log", levels=[logging.INFO, logging.ERROR])

    token:str = ""
    try:
        token = load_token(args.api)
    except Exception as e:
        err.error(e, exc_info=True)
        exit()

    model_name = "Qwen/Qwen3-8B"

    RUN_INFO = f"""RUN: {args.metadata}\n\tMODEL: {model_name}\n\tDATASET: TEST INPUT"""
    log.info(RUN_INFO)

    log.info(f"\n\tDEVICES: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        token=token
    )
    log.info("Model is loaded from pretrained")

    # prepare the model input
    prompt = "Identify the pleonasm in the following sentence: \"We have thoughtfully thought about your situation.\" Output your response in the following JSON format: \{\"pleonasm\": \"<SENTENCE>\"\}."
    log.info(prompt)

    messages = [
        {"role": "user", "content": prompt}
    ]

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

    log.info("Outputs received")

    # parsing thinking content
    index = 0
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    log.info(f"thinking content: {thinking_content}")
    log.info(f"content: {content}")
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
