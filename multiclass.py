import pickle
import datasets
import scripts.liar_experiment.liar_utils as lutils
import scripts.covid.covid_utils as cutils
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, classification_report
import scripts.liar_experiment.liar_boolean as lboolean
import argparse
from typing import Callable, Dict

MODEL_NAME_MAP = {
    "Orca": "Orca-2-7b",
    "Falcon": "Falcon-7b-instruct",
    "Llama": "Llama-2-7b-chat-hf",
    "Mistral": "Mistral-7B-Instruct-v0.2",
}
CLASSES = ["A", "B", "C", "D", "E", "F"]
BINARY_CLASSES = ["A", "B"]


def get_stats(
    shot: int,
    simple_name: str,
    ds,
    extract_fn,
    model_map_fn: Callable[[int], Dict[str, str]],
    binary_mode: bool,
):
    model_map = model_map_fn(shot)
    classes = CLASSES
    if binary_mode:
        classes = BINARY_CLASSES

    responses = pickle.loads(
        open(f"./quantitative/{model_map[simple_name]}.pk", "rb").read()
    )
    n = sum(len(responses[el]) for el in responses)
    assert (n + shot) - len(ds) == 0, f"Expected len {n+shot}, but length is {len(ds)}"

    # first, put the responses in order
    labels_in_order = ["N/A" for _ in range(len(ds))]
    for response in responses:
        for idx in responses[response]:
            labels_in_order[idx] = response

    assert labels_in_order.count("N/A") == shot
    for key in responses:
        assert labels_in_order.count(key) == len(responses[key])
    # store examples (which are the indices with N/A)
    examples = []
    for i in range(shot):
        examples.append(
            labels_in_order.index("N/A", examples[-1] + 1 if len(examples) > 0 else 0)
        )
    examples.sort()

    # now, we can compare in a single pass
    y_true = [extract_fn(ds[i]) for i in range(len(ds))]
    true_positives = {cls: 0 for cls in classes}
    false_positives = {cls: 0 for cls in classes}
    cls_count = {cls: 0 for cls in classes}
    num_correct = 0

    for i in range(len(ds)):
        # ignore if this was an example
        if i in examples:
            continue

        true_label = extract_fn(ds[i])
        cls_count[true_label] += 1

        if labels_in_order[i] == true_label:
            num_correct += 1
            true_positives[true_label] += 1
        elif labels_in_order[i] in classes:
            false_positives[labels_in_order[i]] += 1

    # now clear the examples from labels in order and y_true
    for idx in reversed(examples):
        y_true.pop(idx)
        labels_in_order.pop(idx)
    assert "N/A" not in labels_in_order

    # finally, calculate scores on the cleaned y_true/labels_in_order
    precision = {cls: 0 for cls in classes}
    recall = {cls: 0 for cls in classes}
    f1 = {cls: 0 for cls in classes}
    for cls in classes:
        # calculate precision
        try:
            precision[cls] = true_positives[cls] / (
                true_positives[cls] + false_positives[cls]
            )
        except ZeroDivisionError:
            precision[cls] = -1

        # calculate recall
        recall[cls] = true_positives[cls] / (cls_count[cls])

        # calculate f1
        # 2 * tp / (2*tp + fp + fn) = 2 * tp / (tp + fp + (tp + fn)) = 2 * tp / (tp + fp + cls_count[cls])
        f1[cls] = (
            2
            * true_positives[cls]
            / (true_positives[cls] + false_positives[cls] + cls_count[cls])
        )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true": y_true,
        "labels_in_order": labels_in_order,
    }


def get_classwise_string(
    simple_name, precision, recall, f1, binary_mode: bool = False
) -> str:
    classes = CLASSES
    if binary_mode:
        classes = BINARY_CLASSES

    # format it for table in overleaf
    formatted_str = "\\midrule"

    # add precision
    for cls in classes:
        formatted_str += " & "
        if precision[cls] == -1:
            formatted_str += "N/A "
        else:
            formatted_str += f"{precision[cls]:.3f} "
    formatted_str += f" \\\\ {simple_name}"

    # add recall
    for cls in classes:
        formatted_str += f"& {recall[cls]:.3f} "
    formatted_str += " \\\\ "

    # add f1
    for cls in classes:
        formatted_str += f"& {f1[cls]:.3f}"
    formatted_str += " \\\\"

    return formatted_str


def get_macro_string(
    simple_name: str,
    y_true,
    labels_in_order,
    precision,
    recall,
    f1,
    binary_mode: bool = False,
) -> str:
    # separate string containing macro-statistics
    # ie macro-precision, MCC, and Kappa score
    classes = CLASSES
    if binary_mode:
        classes = BINARY_CLASSES

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for cls in classes:
        macro_precision += precision[cls] if precision[cls] != -1 else 0
        macro_recall += recall[cls]
        macro_f1 += f1[cls]
    macro_precision /= len(classes)
    macro_recall /= len(classes)
    macro_f1 /= len(classes)
    formatted_str = f"{simple_name} & {macro_precision:.3f} & {macro_recall:.3f} & {macro_f1:.3f} & "
    formatted_str += f"{matthews_corrcoef(y_true, labels_in_order):.3f} & {cohen_kappa_score(y_true, labels_in_order):.3f} \\\\"
    return formatted_str


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["covid", "liar"], type=str)
parser.add_argument(
    "--mode", required=False, choices=["bool", "full", "nosample", "covid"], default=""
)
parser.add_argument("--binary", required=False, type=bool, default=False)
if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset == "liar":
        ds = datasets.load_dataset("liar")
    else:
        ds = datasets.load_dataset("nanyy1025/covid_fake_news")
    ds = datasets.concatenate_datasets([ds[el] for el in ds])
    shots = [0, 1, 5]
    mode = args.mode
    binary_mode = args.binary
    data = {}
    model_map_fn = lambda shot: {
        "Orca": f"Orca-2-7b_{shot}responses{'_' + mode if mode != '' else mode}",
        "Falcon": f"falcon-7b-instruct_{shot}responses{'_' + mode if mode != '' else mode}",
        "Llama": f"Llama-2-7b-chat-hf_{shot}responses{'_' + mode if mode != '' else mode}",
        "Mistral": f"Mistral-7B-Instruct-v0.2_{shot}responses{'_' + mode if mode != '' else mode}",
    }
    if args.dataset == "liar" and not args.binary:
        extract_fn = lambda entry: lutils.LABEL_MAP[entry["label"]]
    elif args.dataset == "liar" and args.binary:
        extract_fn = lambda entry: lboolean.LABEL_MAP[
            lboolean.to_binary_label(entry)["label"]
        ]
        model_map_fn = lambda shot: {
            "Orca": f"Orca-2-7b_responses_{mode}_{shot}",
            "Falcon": f"falcon-7b-instruct_responses_{mode}_{shot}",
            "Llama": f"Llama-2-7b-chat-hf_responses_{mode}_{shot}",
            "Mistral": f"Mistral-7B-Instruct-v0.2_responses_{mode}_{shot}",
        }
    else:
        extract_fn = lambda entry: cutils.LABEL_MAP[entry["label"]][0]

    for shot in shots:
        data[shot] = {}
        for model in MODEL_NAME_MAP:
            data[shot][model] = get_stats(
                shot, model, ds, extract_fn, model_map_fn, binary_mode
            )

    # print regular stats as their own table
    if not binary_mode:
        print(
            " & \\textbf{A} & \\textbf{B} & \\textbf{C} & \\textbf{D} & \\textbf{E} & \\textbf{F} \\\\"
        )
    else:
        print(" & \\textbf{A} & \\textbf{B} \\\\")
    for shot in shots:
        print(
            f"\\midrule\n{shot}-shot {mode if mode != '' else 'nondeterministic'} \\\\"
        )
        for model in MODEL_NAME_MAP:
            d = data[shot][model]
            print(
                get_classwise_string(
                    model,
                    precision=d["precision"],
                    recall=d["recall"],
                    f1=d["f1"],
                    binary_mode=binary_mode,
                )
            )

    print()
    print(
        " & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{MCC} & \\textbf{Kappa} \\\\"
    )
    for shot in shots:
        print(
            f"\\midrule\n{shot}-shot {mode if mode != '' else 'nondeterministic'} \\\\"
        )
        for model in MODEL_NAME_MAP:
            d = data[shot][model]
            print(get_macro_string(model, **d, binary_mode=binary_mode))
