import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import TextClassifier

###### FASTTEXT IMPORTS
import fasttext

DATASET_FILE = "../data/Shakespeare_data.csv"


def load_dataset_from(csv_file: str) -> pd.DataFrame:
    """ Loads data from csv_file and returns it as pandas.DataFrame object """

    print(">>> LOADING DATASET FROM FILE {filename}".format(filename=csv_file))
    if not csv_file.endswith(".csv"):
        print("File has to be CSV type file!")
        exit(1)

    try:
        data = pd.read_csv(csv_file)
        print(">>> Finished loading data!")
        return data
    except FileNotFoundError:
        print("File couldn't be found. Verify if '{f_path}' is a correct file path!".format(f_path=csv_file))
        exit(1)


if __name__ == '__main__':
    print(">>> STARTING PROGRAM")

    df = load_dataset_from(DATASET_FILE)

    # To verify if data loaded correctly:
    # print(df.head(10))

    # Extract all players lines and cast it to list
    text = df["PlayerLine"].tolist()
    # Cast test lines into sentences so it can be leverage by flair
    sentences_list = [Sentence(line) for line in text]

    classifier = TextClassifier.load('sentiment')
    classifier.predict(sentences_list)
    print(sentences_list[10])

    # model = fasttext.train_unsupervised(DATASET_FILE, model='skipgram')
    # print(len(model.words))
    # print(len(model.labels))