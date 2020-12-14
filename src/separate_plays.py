from main import (load_dataset_from, DATASET_FILE)
import pandas as pd
import os

NEW_DATASETS_PATH = "../data/new"

if __name__ == "__main__":
    print(">>> STARTING PROGRAM")

    df = load_dataset_from(DATASET_FILE)
    
    play_list = list()
    
    for play_name in df["Play"].unique():
        play_list.append((play_name, df.loc[df["Play"] == play_name, ["Player", "PlayerLine"]]))

    for play in play_list:
        play[1].to_csv(os.path.join(NEW_DATASETS_PATH, ".".join([play[0].replace(" ", ""), "csv"])))
