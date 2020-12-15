import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import TextClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json

###### FASTTEXT IMPORTS
import fasttext

DATASET_FILE = "../data/new/Hamlet.csv"
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

    classifier = TextClassifier.load('sentiment')

    tmp_negatives = {}
    tmp_positives = {}

    print(f'Number of players: {len(df["Player"].unique())}')

    for player_name in df["Player"].unique():
        tmp_negatives[player_name] = list()
        tmp_positives[player_name] = list()

    for dp in df.values:
        l = dp.tolist()
        sentence = Sentence(l[-1])
        classifier.predict(sentence)
        if sentence.labels[0].value == "NEGATIVE":
            tmp_negatives[l[-2]].append(sentence.labels[0].score)
        elif sentence.labels[0].value == "POSITIVE":
            tmp_positives[l[-2]].append(sentence.labels[0].score)

    negative = {}
    positive = {}

    for player in tmp_negatives.keys():
        if len(tmp_negatives[player]) > 10:
            negative[player] = sum(tmp_negatives[player]) / len(tmp_negatives[player])

    for player in tmp_positives.keys():
        if len(tmp_positives[player]) > 10:
            positive[player] = sum(tmp_positives[player]) / len(tmp_positives[player])

    print(negative)
    print(positive)

    # for player in tmp.keys():
    #     tmp[player] = Sentence(tmp[player])
    #     classifier.predict(tmp[player])
    #     if tmp[player].labels[0].value == "NEGATIVE":
    #         negative[player] = tmp[player].labels[0].score
    #     elif tmp[player].labels[0].value == "POSITIVE":
    #         positive[player] = tmp[player].labels[0].score
    #
    f = plt.figure()
    f.add_subplot(2, 1, 1)
    wordcloud_negative = WordCloud(background_color='black',
                          width=500,
                          height=250,
                          relative_scaling=1,
                          ).generate_from_frequencies(negative)

    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.title("NEGATIVE")
    plt.axis("off")
    #plt.show()

    wordcloud_positive = WordCloud(background_color='black',
                          width=500,
                          height=250,
                          relative_scaling=1,
                          ).generate_from_frequencies(positive)
    f.add_subplot(2, 1, 2)
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.title("POSITIVE")
    plt.axis("off")
    plt.show()

    positive_json = json.dumps(positive, indent=4)
    negative_json = json.dumps(negative, indent=4)



    # text = df["PlayerLine"].tolist()
    # Cast test lines into sentences so it can be leverage by flair
    # sentences_list = [Sentence(line) for line in text[:100]]
    #
    # classifier = TextClassifier.load('sentiment')
    # classifier.predict(sentences_list)
    # print(sentences_list[10])

    # model = fasttext.train_unsupervised(DATASET_FILE, model='skipgram')
    # print(len(model.words))
    # print(len(model.labels))
