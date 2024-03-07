import os
import datasets
import captum.attr as attr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import random
from argparse import ArgumentParser
from liar_experiment.liar_utils import *


# set up argument parser
parser = ArgumentParser()
parser.add_argument("device", type=int, choices=[0, 1], help="The cuda device to use")
parser.add_argument(
    "model",
    type=str,
    choices=["falcon", "llama", "mistral", "orca"],
    help="the model to evaluate",
)
parser.add_argument(
    "shot", type=int, choices=[0, 1, 5], help="How many examples to give"
)
parser.add_argument(
    "steps",
    type=int,
    default=512,
    help="How many steps to use when approximating integrated gradients",
)
parser.add_argument("samples", type=int, help="How many different samples to take")


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    # load dataset
    liar = datasets.load_dataset("liar")
    full_liar = datasets.concatenate_datasets(
        [liar["train"], liar["test"], liar["validation"]]
    )

    # load model
    model_name = MODEL_DICT[args.model]
    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config,  # device_map="auto"
    )

    # get examples
    entries = random.choices(list(range(len(full_liar))), k=args.shot)

    # figure out which tokens correspond to A,B,C,D,E,F
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

    for sample in range(args.samples):
        # make prompt
        prompt = to_n_shot_prompt(
            args.shot, full_liar[random.randint(0, len(full_liar))], full_liar, entries
        )
        print(prompt)

        # get tokens for later
        tokens = tokenizer(prompt, return_tensors="pt").input_ids

        # calculate which label the model responds w/ (so we can evaluate integrated gradients for that label)
        with torch.no_grad():
            label = torch.argmax(model(tokens.cuda()).logits[:, -1])
        label = tokenizer.decode(label)

        # replace the normal pytorch embeddings (which only take ints) to interpretable embeddings
        # (which are compatible with the float inputs that integratedgradients gives)
        if hasattr(model, "model"):
            interpretable_emb = attr.configure_interpretable_embedding_layer(
                model.model, "embed_tokens"
            )
        elif hasattr(model, "transformer"):
            interpretable_emb = attr.configure_interpretable_embedding_layer(
                model.transformer, "word_embeddings"
            )
        else:
            print("This shouldn't happen!!")
            raise Exception(
                "Unable to convert model embeddings to interpretable embeddings"
            )

        # calculate inputs and baselines
        input_embs = interpretable_emb.indices_to_embeddings(tokens).cpu()
        baselines = torch.zeros_like(input_embs).cpu()

        # calculate integrated gradients
        ig = attr.IntegratedGradients(lambda inps: softmax_results_embeds(inps, model))
        attributions = ig.attribute(
            input_embs,
            baselines=baselines,
            target=label_tokens[label],
            n_steps=args.steps,
            internal_batch_size=2,
            return_convergence_delta=True,
        )

        # convert attributions to [len(tokens)] shape
        summarized_attributions = summarize_attributions(attributions[0])
        all_tokens = tokens.squeeze(0)
        all_tokens = list(map(tokenizer.decode, all_tokens))

        # remove the interpretable embedding layer so we can get regular predictions
        if hasattr(model, "model"):
            attr.remove_interpretable_embedding_layer(model.model, interpretable_emb)
        else:
            attr.remove_interpretable_embedding_layer(
                model.transformer, interpretable_emb
            )
        with torch.no_grad():
            predictions = softmax_results(tokens, model)

        MARGIN_OF_ERROR = 0.1  # off by no more than 10 percentage points
        if (
            torch.abs(
                (summarized_attributions.sum() - predictions[0, label_tokens[label]])
            )
            >= MARGIN_OF_ERROR
        ):
            print("we are off!!")
            print(
                "we should be getting somewhere near",
                predictions[0, label_tokens[label]],
            )
            print("instead, we get", summarized_attributions.sum())

        # make and save html
        SCALE = 2 / summarized_attributions.abs().max()
        attr_vis = CustomDataRecord(
            summarized_attributions * SCALE,  # word attributions
            predictions[0].max(),  # predicted probability
            tokenizer.decode(torch.argmax(predictions[0])),  # predicted class
            label,  # attr class
            predictions[0, label_tokens[label]],  # attr probability
            summarized_attributions.sum(),  # attr score
            all_tokens,  # raw input ids
            attributions[1],  # convergence delta
        )
        html = visualize_text([attr_vis])

        save_results("ig", html, attributions, model_name, sample)
