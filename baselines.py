# Calls the models we are using as baselines
# Created by Alejandro Ciuba, alejandrociuba@pitt.edu
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig,
    )

import logging
import torch

def load_model(name: str, token: Path | str, **kwargs):


    model, tokenizer = None, None
    if name == "Qwen/Qwen3-8B":

        tokenizer = AutoTokenizer.from_pretrained(name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="auto",
            token=token
            )

    elif name == "deepseek-ai/deepseek-llm-7b-chat":

        tokenizer = AutoTokenizer.from_pretrained(name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token
            )
        model.generation_config = GenerationConfig.from_pretrained(name, token=token)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    if model is None or tokenizer is None:
        raise TypeError("No matching model name.")

    return model, tokenizer


def prompt_model(name: str, messages: list, **kwargs):

    if name == "Qwen/Qwen3-8B":

        text = kwargs['tokenizer'].apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
            )

        model_inputs = kwargs['tokenizer']([text], return_tensors="pt").to(kwargs['model'].device)

        # conduct text completion
        generated_ids = kwargs['model'].generate(
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

        thinking_content = kwargs['tokenizer'].decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = kwargs['tokenizer'].decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    elif name == "deepseek-ai/deepseek-llm-7b-chat":

        input_tensor = kwargs['tokenizer'].apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = kwargs['model'].generate(input_tensor.to(kwargs['model'].device), max_new_tokens=100)

        content = kwargs['tokenizer'].decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True).strip("\n")

    return content