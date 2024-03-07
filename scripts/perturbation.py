import os
import datasets
import argparse
from typing import Dict
import captum.attr as attr
from captum._utils.models.linear_model import SkLearnLasso
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import random
from liar_experiment.liar_utils import *

# globals
MODEL_MAP = {
    "falcon": "tiiuae/falcon-7b-instruct",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "orca": "microsoft/Orca-2-7b",
}


def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return LABEL_MAP[entry["label"]] in decoded


# use argparse to take experiment_type (string), model_name (string, either llama, falcon, mistral, or orca),
# n_examples (int, either 0, 1, or 5), n_perturbation_samples (an arbitrary int), and n_samples (how many times to run the experiment)
parser = argparse.ArgumentParser()
parser.add_argument("device", type=int, choices=[0, 1], help="the cuda device to use")
parser.add_argument(
    "model_name",
    type=str,
    help="the model to use",
    choices=["llama", "falcon", "mistral", "orca"],
)
parser.add_argument(
    "experiment_type",
    type=str,
    help="the experiment type to run",
    choices=["lime", "shap"],
)
parser.add_argument(
    "n_examples", type=int, help="the number of examples to use", choices=[0, 1, 5]
)
parser.add_argument(
    "n_perturbation_samples",
    type=int,
    help="the number of samples to use in perturbation methods",
)
parser.add_argument(
    "n_samples", type=int, help="the number of times to run the experiment"
)


# specific functions
def softmax_results(tokens: torch.Tensor):
    with torch.no_grad():
        if tokenizer.bos_token_id is not None:  # falcon's is None
            tokens[0, 0] = tokenizer.bos_token_id
        result = model(
            torch.where(tokens != 0, tokens, tokenizer.eos_token_id).cuda(),
            attention_mask=torch.where(tokens != 0, 1, 0).cuda(),
        ).logits
        ret = torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()
        assert not ret.isnan().any()
    return ret


def get_embeds(tokens: torch.Tensor):
    with torch.no_grad():
        if hasattr(model, "model"):
            return model.model.embed_tokens(tokens.cuda())
        elif hasattr(model, "transformer"):
            return model.transformer.word_embeddings(tokens.cuda())
        raise Exception("Unknown model format")


# encode text indices into latent representations & calculate cosine similarity
def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
    original_emb = get_embeds(original_inp)
    perturbed_emb = get_embeds(perturbed_inp)
    distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=-1)
    distance[distance.isnan()] = 0
    ret = torch.exp(-1 * (distance**2) / 2).sum()
    assert not ret.isnan().any()
    return ret


# binary vector where each word is selected independently and uniformly at random
def bernoulli_perturb(text, **kwargs):
    probs = torch.ones_like(text) * 0.5
    probs[0, 0] = 0  # don't get rid of the start token
    ret = torch.bernoulli(probs).long()
    return ret


# remove absent tokens based on the intepretable representation sample
def interp_to_input(interp_sample, original_input, **kwargs):
    ret = original_input.clone()
    ret[interp_sample.bool()] = 0
    return ret


if __name__ == "__main__":
    # parse args
    args = parser.parse_args()
    model_name = args.model_name
    experiment_type = args.experiment_type.upper()
    n_examples = args.n_examples
    n_perturbation_samples = args.n_perturbation_samples
    n_samples = args.n_samples

    # setup the device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    # load dataset
    liar = datasets.load_dataset("liar")
    full_liar = datasets.concatenate_datasets(
        [liar["train"], liar["test"], liar["validation"]]
    )

    # load model
    model_name = MODEL_MAP[model_name]
    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,  # device_map="auto"
    )

    # calculate entries
    entries = random.choices(list(range(len(full_liar))), k=n_examples)

    # calculate vocab
    vocab = tokenizer.vocab
    label_tokens = {
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "F": [],
    }
    for char, idx in vocab.items():
        if char in label_tokens:
            label_tokens[char].append(idx)
    label_tokens = {char: idxs[0] for char, idxs in label_tokens.items()}

    for sample in range(n_samples):
        prompt = to_n_shot_prompt(
            n_examples, full_liar[random.randint(0, len(full_liar))], full_liar, entries
        )
        print(prompt)
        # get tokens for later
        tokens = tokenizer(prompt, return_tensors="pt").input_ids

        # calculate which label the model responds w/ (so we can evaluate integrated gradients for that label)
        with torch.no_grad():
            label = torch.argmax(model(tokens.cuda()).logits[:, -1])
        label = tokenizer.decode(label)

        # run experiment
        LIME = "LIME"
        SHAP = "SHAP"
        if experiment_type == LIME:
            attributer = attr.LimeBase(
                softmax_results,
                interpretable_model=SkLearnLasso(alpha=0.0003),
                similarity_func=exp_embedding_cosine_distance,
                perturb_func=bernoulli_perturb,
                perturb_interpretable_space=True,
                from_interp_rep_transform=interp_to_input,
                to_interp_rep_transform=None,
            )
        elif experiment_type == SHAP:
            attributer = attr.KernelShap(softmax_results)
        else:
            raise Exception("Invalid Experiment Type")
        attributions = attributer.attribute(
            tokens,
            target=label_tokens[label],
            n_samples=n_perturbation_samples,
            show_progress=True,
        )

        # verify attributions
        assert attributions.nonzero().numel() != 0
        all_tokens = tokens.squeeze(0)
        all_tokens = list(map(tokenizer.decode, all_tokens))

        # get predictions for comparison
        with torch.no_grad():
            predictions = softmax_results(tokens)

        # save attributions
        SCALE = 2 / attributions.abs().max()
        attr_vis = CustomDataRecord(
            attributions[0] * SCALE,  # word attributions
            predictions[0].max(),  # predicted probability
            tokenizer.decode(torch.argmax(predictions[0])),  # predicted class
            label,  # attr class
            predictions[0, label_tokens[label]],  # attr probability
            attributions.sum(),  # attr score
            all_tokens,  # raw input ids
            (
                abs(predictions[0, label_tokens[label]] - attributions.sum())
                if experiment_type == SHAP
                else None
            ),
        )
        html = visualize_text([attr_vis])

        save_results(experiment_type.lower(), html, attributions, model_name, sample)
