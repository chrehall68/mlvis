import argparse
import os
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict
import random
from tqdm import tqdm
import pickle

# seed random
random.seed(1770)

# Globals
LABEL_MAP = {0: "A", 1: "B"}
responses: Dict[str, list] = {}


# 0 : False
# 1 : Half True
# 2 : Mostly True
# 3 : True
# 4 : Barely True
# 5 : Pants on Fire
def to_binary_label(entry):
    if entry["label"] in [3, 2, 1]:
        entry["label"] = 0
    else:
        entry["label"] = 1
    return entry


def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return LABEL_MAP[entry["label"]] in decoded


def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    speaker = entry["speaker"].replace("-", " ").title()
    statement = entry["statement"].lstrip("Says ")

    prompt = f"""Please select the option that most closely describes the following claim by {speaker}:\n{statement}\n\nA) True\nB) False\n\nChoice: ("""
    return prompt


def to_n_shot_prompt(n: int, entry: Dict[str, str]) -> str:
    examples = ""
    for i in range(n):
        examples += (
            to_zero_shot_prompt(full_liar[entries[i]])
            + LABEL_MAP[full_liar[entries[i]]["label"]]
            + "\n\n"
        )
    prompt = to_zero_shot_prompt(entry)
    return examples + prompt


def workflow(idx, entry: dict, model, k: int = 0, verbose: bool = False) -> bool:
    prompt = to_n_shot_prompt(k, entry)

    # encode input, move it to cuda, then generate
    encoded_input = tokenizer(prompt, return_tensors="pt")
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
parser.add_argument("device", type=int, choices=[0, 1], help="cuda device to use")
parser.add_argument("model", type=str, choices=["falcon", "llama", "mistral", "orca"])
parser.add_argument("shot", type=int, choices=[0, 1, 5])

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    # load full dataset
    liar = datasets.load_dataset("liar")
    train = liar["train"]
    test = liar["test"]
    val = liar["validation"]
    full_liar = datasets.concatenate_datasets([train, test, val])
    full_liar = full_liar.map(to_binary_label)

    model_map = {
        "falcon": "tiiuae/falcon-7b-instruct",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "orca": "microsoft/Orca-2-7b",
    }
    model_name = model_map[args.model]

    # load model
    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,
    )

    # run experiment
    entries = random.choices(list(range(len(full_liar))), k=args.shot)

    workflow(
        0, train[random.randint(0, len(train) - 1)], model, verbose=True, k=args.shot
    )

    num_correct = 0
    responses = {}
    for idx, entry in enumerate((prog := tqdm(full_liar))):
        if idx in entries:
            continue  # don't include items that were in the examples

        correct = workflow(idx, entry, model, k=args.shot)
        if correct:
            num_correct += 1
        prog.set_postfix_str(f"acc: {num_correct/(idx+1):.3f}")

    # log results
    with open(f"{args.shot}_shot_binary.txt", "a") as file:
        file.write(f"{model_name} : {num_correct}/{len(full_liar)-len(entries)}\n")

    # print results up till now
    with open(f"{args.shot}_shot_binary.txt", "r") as file:
        print(file.read())

    # dump responses
    pickle.dump(
        responses,
        open(
            f"{model_name[model_name.index('/')+1:]}_responses_bool_{args.shot}.pk",
            "wb",
        ),
    )
