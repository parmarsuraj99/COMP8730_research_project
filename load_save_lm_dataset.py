import random
import re

import gc

from datasets import load_dataset
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split

random.seed(0)
print(random.random())


def load_oscar() -> list:
    """
    Loads OSCAR dataset for Sanskrit using HuggingFace datasets

    :return: list of str
    """

    dataset = load_dataset("oscar", "unshuffled_deduplicated_sa")
    text = dataset["train"]["text"]

    text = [t.replace("\n", "") for t in text]

    return text


def load_wikipedia() -> list:
    """
    Loads WikiPedia Sanskrit dataset from TensorFlow dataset hub

    :return: list of str
    """

    # Construct a tf.data.Dataset
    ds = tfds.load("wikipedia/20200301.sa:1.0.0", split="train")

    df = tfds.as_dataframe(ds.take(len(ds)))
    text = df.text.apply(lambda x: x.decode("utf-8"))
    text = [t.replace("\n", "") for t in text]
    return text


def load_oscar_wiki_() -> list:
    """
    Calls different dataloading functions and returns combined list of texts
    Also cleans the text by removing English alphanumericals

    :return: list of str
    """

    oscar_ = load_oscar()
    oscar_ = oscar_
    print("oscar loaded")

    wiki_ = load_wikipedia()
    wiki_ = wiki_

    final_text = oscar_ + wiki_
    final_text = [re.sub("[a-zA-Z0-9]+", "", txt_) for txt_ in final_text]
    return final_text


txt = load_oscar_wiki_()

# splitting into train and evaluation splits
train_, eval_ = train_test_split(txt, test_size=0.2, random_state=0)


def save_list_to_txt(fname: str, arr: list):
    """
    Append new line ('\n') to end of string and save the list in a txt file
    :param fname: filename to load
    :param arr: array to save
    """
    with open(fname, "w") as fp:
        fp.writelines([a + "\n" for a in arr])


# saving arrays to files
save_list_to_txt("sanskrit_corpus_full.txt", txt)
save_list_to_txt("sanskrit_corpus_train.txt", train_)
save_list_to_txt("sanskrit_corpus_eval.txt", eval_)

print(txt[:100])
gc.collect()
