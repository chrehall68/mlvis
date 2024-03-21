from typing import Dict

# globals
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
