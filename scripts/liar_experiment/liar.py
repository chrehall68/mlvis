from typing import Dict
import datasets
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)
import accelerate
import random
from tqdm import tqdm
import pickle
import argparse
from liar_utils import *
import os

# globals
responses: Dict[str, list] = {}


def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return LABEL_MAP[entry["label"]] in decoded


def workflow(idx: int, entry: dict, model, verbose: bool = False) -> bool:
    # encode input, move it to cuda, then generate
    encoded_input = tokenizer(entry["prompt"], return_tensors="pt")
    encoded_input = {item: val.cuda() for item, val in encoded_input.items()}
    generation = model.generate(
        **encoded_input,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # log the prompt and response if verbose
    if verbose:
        print(tokenizer.batch_decode(generation)[0])

    decoded = tokenizer.decode(generation[0, -1])
    correct = was_correct(decoded, entry)

    if decoded not in responses:
        responses[decoded] = []
    responses[decoded].append(idx)

    if verbose:
        print(
            "The model was",
            "correct" if correct else "incorrect",
            " - responded",
            tokenizer.decode(generation[0, -1]),
            "and answer should have been",
            LABEL_MAP[entry["label"]],
        )
    return correct


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=["falcon", "llama", "mistral", "orca"])
parser.add_argument("shot", type=int, choices=[0, 1, 5])
parser.add_argument(
    "--device", type=int, help="cuda device to use", required=False, default=-1
)
parser.add_argument(
    "--full",
    required=False,
    type=bool,
    help="Whether to use full fp16 precision",
    default=False,
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.device != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # load full dataset
    liar = datasets.load_dataset("liar")
    train = liar["train"]
    test = liar["test"]
    val = liar["validation"]
    full_liar = datasets.concatenate_datasets([train, test, val])

    model_map = {
        "falcon": "tiiuae/falcon-7b-instruct",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "orca": "microsoft/Orca-2-7b",
    }
    model_name = model_map[args.model]

    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not args.full:
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,  # device_map="auto"
        )
    else:
        config = AutoConfig.from_pretrained(model_name)
        with accelerate.init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.bfloat16
            )
        model.tie_weights()
        dev_map = accelerate.infer_auto_device_map(
            model,
            max_memory={0: "11GB", 1: "7GB"},
            no_split_module_classes=[
                "LlamaDecoderLayer",
                "MistralDecoderLayer",
                "FalconDecoderLayer",
            ],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=dev_map
        )

    # setup
    n_examples = args.shot
    entries = random.choices(list(range(len(train))), k=n_examples)
    full_liar = full_liar.map(
        lambda e: {"prompt": to_n_shot_prompt(n_examples, e, full_liar, entries)}
    )

    # run experiment
    num_correct = 0
    for idx, entry in enumerate((prog := tqdm(full_liar))):
        if idx in entries:
            continue  # don't include items that were in the examples

        correct = workflow(idx, entry, model)
        if correct:
            num_correct += 1
        prog.set_postfix_str(f"acc: {num_correct/(idx+1):.3f}")

    # log results
    tail = "" if not args.full else "_full"
    pickle.dump(
        responses,
        open(
            f"{model_name[model_name.index('/')+1:]}_{n_examples}responses{tail}.pk",
            "wb",
        ),
    )
    with open(f"{n_examples}_shot{tail}.txt", "a") as file:
        file.write(f"{model_name} : {num_correct}/{len(full_liar)-len(entries)}\n")

    # print results up till now
    with open(f"{n_examples}_shot{tail}.txt", "r") as file:
        print(file.read())
