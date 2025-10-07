# Experiment with setting up models for pleonasm detection
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from pathlib import Path
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from logger import make_loggers

import argparse
import logging
import torch
import transformers


def main(args: argparse.Namespace):

    model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    messages = [
        {"role": "user", "content": "Identify the pleonasm in this sentence: \"This is a free gift.\""},
    ]

    print(messages)
    print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    print("Model is past the pipeline stage, running")

    output = pipe(messages, do_sample=False, max_new_tokens=200)

    print("Model has been inferenced")
    print(output[0]["generated_text"][-1]["content"])

    # model_id = "meta-llama/Llama-3.1-8B"

    # pipeline = transformers.pipeline(
    #     "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    # )

    # print(pipeline("""Can you tell me which word is a pleonasm in the following sentence? \"This is a free gift.\"}"""))

    # log, err = make_loggers(Path("./logs/runs/test-run.log"), Path("./logs/errors/test-err.log"),
    #                         levels=[logging.INFO, logging.ERROR])
    
    # log.info(f"METADATA: {args.metadata}")
    # logging.info(f"METADATA: {args.metadata}")

    # model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

    # processor = AutoProcessor.from_pretrained(model_id)
    # model = Llama4ForConditionalGeneration.from_pretrained(
    #     model_id,
    #     attn_implementation="flex_attention",
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    # )

    # # url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    # # url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
    # messages = \
    # [
    #     {
    #         "role": "user",
    #         "content": 
    #         [
    #             {"type": "text", 
    #              "text": 
    #              """Can you tell me which word is a pleonasm in the following sentence?
    #              \"This is a free gift.\"
    #              Output your results in the following JSON format:
    #              \{\"pleonasm\": <WORD>\}
    #              """,
    #             },
    #         ],
    #     },
    # ]

    # inputs = processor.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt",
    # ).to(model.device)

    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=256,
    # )

    # response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
    # log.info(response)
    # log.info(outputs[0])
    # log.info(type(outputs))


def add_args(parser: argparse.ArgumentParser):

    parser.add_argument(
        "-m",
        "--metadata",
        nargs="*",
        default="TEST RUN",
        help="Metadata used to identify the run in logs.",
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
