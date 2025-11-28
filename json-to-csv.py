# Quick script to turn the JSON to a CSV
from pathlib import Path

import json

import pandas as pd


def main():

    SPC_json = Path("DATA_PATH")

    with open(SPC_json, "r") as src:
        data: pd.DataFrame = pd.read_json(SPC_json, lines=True)

    print(data.info())

    data["sentence"] = data.apply(lambda x: f"{x.before} {x['first']} {x.second} {x.after}", axis=1)
    data["pleonasm"] = data.apply(lambda x: f"{x['first']} {x.second}" if x.consensus.lower().strip() != "neither" else "", axis=1)

    data = data.drop(columns=set(data.columns).difference(set(["sentence", "pleonasm", "fold"])))  # Assumes folds have already been made with splits.py

    data.to_csv("SAVE_PATH", index=False)


if __name__ == "__main__":
    main()