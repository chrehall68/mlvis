import os
import datasets
import captum
import captum.attr as attr
from captum.attr import visualization as viz
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import random
from typing import Dict, Any
from dataclasses import dataclass
from IPython.display import HTML
from argparse import ArgumentParser


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
parser.add_argument("samples", type=int, help="How many different samples to take")
parser.add_argument(
    "steps",
    type=int,
    default=512,
    help="How many steps to use when approximating integrated gradients",
)

# globals
MODEL_DICT = {
    "falcon": "tiiuae/falcon-7b-instruct",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "orca": "microsoft/Orca-2-7b",
}
LABEL_MAP = {
    0: "E",  # 0 : False
    1: "C",  # 1 : Half True
    2: "B",  # 2 : Mostly True
    3: "A",  # 3 : True
    4: "D",  # 4 : Barely True
    5: "F",  # 5 : Pants on Fire
}
random.seed(2024)


# functions
def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    speaker = entry["speaker"].replace("-", " ").title()
    try:
        statement = entry["statement"][entry["statement"].index("Says ") + 5 :]
    except:
        statement = entry["statement"]

    prompt = f"""Please select the option that most closely describes the following claim by {speaker}:\n{statement}\n\nA) True\nB) Mostly True\nC) Half True\nD) Barely True\nE) False\nF) Pants on Fire (absurd lie)\n\nChoice: ("""
    return prompt


def to_n_shot_prompt(n: int, entry: Dict[str, str], full_liar) -> str:
    examples = ""
    for i in range(n):
        examples += (
            to_zero_shot_prompt(full_liar[entries[i]])
            + LABEL_MAP[full_liar[entries[i]]["label"]]
            + "\n\n"
        )
    prompt = to_zero_shot_prompt(entry)
    return examples + prompt


def softmax_results(inputs: torch.Tensor):
    result = model(inputs.cuda()).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()


def softmax_results_embeds(embds: torch.Tensor):
    result = model(inputs_embeds=embds.cuda()).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()


def summarize_attributions(attributions):
    with torch.no_grad():
        attributions = attributions.sum(dim=-1).squeeze(0)
        return attributions


@dataclass
class CustomDataRecord:
    word_attributions: Any
    pred_prob: torch.Tensor
    pred_class: str
    attr_class: str
    attr_prob: torch.Tensor
    attr_score: Any
    raw_input_ids: Any | list[str]
    convergence_delta: Any


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-2, min(2, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def visualize_text(
    datarecords: list[CustomDataRecord], legend: bool = True
) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Convergence Delta</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    viz.format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    viz.format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.attr_class, datarecord.attr_prob
                        )
                    ),
                    viz.format_classname(
                        "{0:.2f}".format(datarecord.convergence_delta.item())
                    ),
                    viz.format_classname("{0:.2f}".format(datarecord.attr_score)),
                    viz.format_word_importances(
                        datarecord.raw_input_ids, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))

    return html


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
            args.shot, full_liar[random.randint(0, len(full_liar))], full_liar
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
        ig = attr.IntegratedGradients(softmax_results_embeds)
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
            predictions = softmax_results(tokens)

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
            print("instead, we get", summarized_attributions[label].sum())

        # make and save html
        SCALE = 2 / summarized_attributions.max()
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
        open(f"ig/{model_name[model_name.index('/')+1:]}_{sample}.html", "w").write(
            html.data
        )
