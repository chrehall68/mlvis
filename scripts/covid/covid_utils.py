from typing import Dict

# globals
LABEL_MAP = {"real": ["A", "True"], "fake": ["B", "False"]}
MAX_LEN = 256


# workflow functions
def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    prompt = f"""Please select the option (A or B) that most closely describes the following claim:\n{entry['tweet'][:MAX_LEN]}\n\n(A) True\n(B) False\n\nChoice: ("""
    return prompt


def to_n_shot_prompt(
    n: int, entry: Dict[str, str], ds, entries: Dict[int, Dict]
) -> str:
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
