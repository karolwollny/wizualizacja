import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import TextClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

###### FASTTEXT IMPORTS
import fasttext

DATASET_FILE = "../data/Shakespeare_data.csv"
MASK_FILE = "../data/william-shakespeare-black-silhouette.jpg"


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

    tmp = {}
    for dp in df.values:
        l = dp.tolist()
        tmp[str(l[-2])] = " ".join([tmp[str(l[-2])], l[-1]]) if str(l[-2]) in tmp.keys() else l[-1]

    classifier = TextClassifier.load('sentiment')

    negative = {}
    positive = {}

    for player in tmp.keys():
        tmp[player] = Sentence(tmp[player])
        classifier.predict(tmp[player])
        if tmp[player].labels[0].value == "NEGATIVE":
            negative[player] = tmp[player].labels[0].score
        elif tmp[player].labels[0].value == "POSITIVE":
            positive[player] = tmp[player].labels[0].score

    wordcloud_negative = WordCloud(background_color='black',
                          width=500,
                          height=250,
                          relative_scaling=1,
                          ).generate_from_frequencies(negative)

    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.title("NEGATIVE")
    plt.axis("off")
    plt.show()

    wordcloud_positive = WordCloud(background_color='black',
                          width=500,
                          height=250,
                          relative_scaling=1,
                          ).generate_from_frequencies(positive)

    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.title("POSITIVE")
    plt.axis("off")
    plt.show()

    #text = df["PlayerLine"].tolist()
    # Cast test lines into sentences so it can be leverage by flair
    # sentences_list = [Sentence(line) for line in text[:100]]
    #
    # classifier = TextClassifier.load('sentiment')
    # classifier.predict(sentences_list)
    # print(sentences_list[10])

    # model = fasttext.train_unsupervised(DATASET_FILE, model='skipgram')
    # print(len(model.words))
    # print(len(model.labels))
