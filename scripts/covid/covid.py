import os
import datasets
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict
import random
from tqdm import tqdm
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("device", type=int, choices=[0, 1])
parser.add_argument("model", type=str, choices=["llama", "falcon", "mistral", "orca"])
parser.add_argument("shot", type=int, choices=[0, 1, 5])

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
# covid-19 section of
# @misc{patwa2020fighting,
# title={Fighting an Infodemic: COVID-19 Fake News Dataset},
# author={Parth Patwa and Shivam Sharma and Srinivas PYKL and Vineeth Guptha and Gitanjali Kumari and Md Shad Akhtar and Asif Ekbal and Amitava Das and Tanmoy Chakraborty},
# year={2020},
# eprint={2011.03327},
# archivePrefix={arXiv},
# primaryClass={cs.CL}
# }
ds = datasets.load_dataset("nanyy1025/covid_fake_news")

train = ds["train"]
test = ds["test"]
val = ds["validation"]

ds = datasets.concatenate_datasets([train, test, val])
# ## Model Loading

model_map = {
    "falcon": "tiiuae/falcon-7b-instruct",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "orca": "microsoft/Orca-2-7b",
    "gemma": "google/gemma-7b-it",
}


# change this depending on experiment
model_name = model_map[args.model]


# torch.backends.cuda.matmul.allow_tf32 = True

config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config)


LABEL_MAP = {"real": ["A", "True"], "fake": ["B", "False"]}


def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return any(map(lambda el: el.lower() in decoded.lower(), LABEL_MAP[entry["label"]]))


n_examples = args.shot


random.seed(1770)
entries = random.choices(list(range(len(ds))), k=n_examples)
MAX_LEN = 256


def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    prompt = f"""Please select the option (A or B) that most closely describes the following claim: {entry['tweet'][:MAX_LEN]}\n\n(A) True\n(B) False\n\nChoice: ("""
    return prompt


def to_n_shot_prompt(n: int, entry: Dict[str, str]) -> str:
    examples = ""
    for i in range(n):
        examples += (
            to_zero_shot_prompt(ds[entries[i]])
            + LABEL_MAP[ds[entries[i]]["label"]][0]
            + ")"
            + "\n\n"
        )
    prompt = to_zero_shot_prompt(entry)
    return examples + prompt


responses: Dict[str, list] = {}


def workflow(idx: int, entry: dict, model, k: int = 0, verbose: bool = False) -> bool:
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


ds = ds.map(lambda e: {"prompt": to_n_shot_prompt(n_examples, e)})

workflow(0, ds[random.randint(0, len(ds) - 1)], model, verbose=True, k=n_examples)


num_correct = 0
responses = {}
for idx, entry in enumerate((prog := tqdm(ds))):
    if idx in entries:
        continue  # don't include items that were in the examples

    correct = workflow(idx, entry, model, k=n_examples)
    if correct:
        num_correct += 1
    prog.set_postfix_str(f"acc: {num_correct/(idx+1):.3f}")


pickle.dump(
    responses,
    open(f"{model_name[model_name.index('/')+1:]}_{n_examples}responses.pk", "wb"),
)


# log results
with open(f"{n_examples}_shot_covid.txt", "a") as file:
    file.write(f"{model_name} : {num_correct}/{len(ds)-len(entries)}\n")


# print results up till now
with open(f"{n_examples}_shot_covid.txt", "r") as file:
    print(file.read())
