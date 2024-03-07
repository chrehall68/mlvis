import random
from typing import Dict, Any
import torch
from dataclasses import dataclass
from IPython.display import HTML
from captum.attr import visualization as viz
import os

# seed experiments
random.seed(2024)

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


# workflow functions
def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    speaker = entry["speaker"].replace("-", " ").title()
    try:
        statement = entry["statement"][entry["statement"].index("Says ") + 5 :]
    except:
        statement = entry["statement"]

    prompt = f"""Please select the option that most closely describes the following claim by {speaker}:\n{statement}\n\nA) True\nB) Mostly True\nC) Half True\nD) Barely True\nE) False\nF) Pants on Fire (absurd lie)\n\nChoice: ("""
    return prompt


def to_n_shot_prompt(
    n: int, entry: Dict[str, str], full_liar, entries: Dict[int, Dict]
) -> str:
    examples = ""
    for i in range(n):
        examples += (
            to_zero_shot_prompt(full_liar[entries[i]])
            + LABEL_MAP[full_liar[entries[i]]["label"]]
            + "\n\n"
        )
    prompt = to_zero_shot_prompt(entry)
    return examples + prompt


def softmax_results(inputs: torch.Tensor, model: torch.nn.Module):
    result = model(inputs.cuda()).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()


def softmax_results_embeds(embds: torch.Tensor, model: torch.nn.Module):
    result = model(inputs_embeds=embds.cuda()).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()


def summarize_attributions(attributions):
    with torch.no_grad():
        attributions = attributions.sum(dim=-1).squeeze(0)
        return attributions


# visualization utils
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
    # clip values to [-2,2])
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
    if datarecords[0].convergence_delta is not None:
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
    else:
        rows = [
            "<tr><th>Predicted Label</th>"
            "<th>Attribution Label</th>"
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


def save_results(
    experiment_name: str,
    html: HTML,
    attributions: torch.Tensor,
    model_name: str,
    sample_num: int,
):
    # make the directories if necessary
    html_dir = f"html/{experiment_name}/{model_name[model_name.index('/')+1:]}"
    pt_dir = f"pt/{experiment_name}/{model_name[model_name.index('/')+1:]}"
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    # write html data
    open(f"{html_dir}/{sample_num}.html", "w").write(html.data)

    # save pts as well
    torch.save(attributions, f"{pt_dir}/{sample_num}.pt")
