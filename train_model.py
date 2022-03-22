import argparse
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import DataCollatorForLanguageModeling
from tqdm.auto import tqdm
import torch
from transformers import TrainingArguments, Trainer
import random

random.seed(0)


def load_text(fname: str = "sanskrit_corpus.txt") -> list:
    """
    load text and returns lines in a list after removing new line

    :param fname: filename to load
    :return: list of str
    """
    with open(file=fname, mode="r") as fp:
        lines = fp.read().split("\n")
        return lines


def load_model_tokenizer(model_name: str, from_scratch=False) -> tuple:
    """
    load model tokenizer and model from HF hub using model_name

    :param model_name: model_name to load from HF hub
    :param from_scratch: if True, don't use pretrained weights and load from config, otherwise loda pretrained weights
    :return: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, strip_accents=False)
    if from_scratch:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_config(config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name)

    return (tokenizer, model)


def batch_encode(text: list, max_seq_len: int, batch_size=4096) -> tuple:
    """
    memory efficient encoding of sentences

    :param text: list of sentences
    :param max_seq_len: maximum sequence to tokenize
    :param batch_size: howmany sentences to tokenize at once
    :return: (input_ids, attention_mask)
    """
    encoded_sentences = []
    for i in tqdm(range(0, len(text), batch_size)):
        encoded_sent = tokenizer.batch_encode_plus(
            text[i : i + batch_size],
            max_length=max_seq_len,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True,
            # return_tensors="pt",
        )

    encoded_sentences += encoded_sent

    return (input_ids_train, attention_masks_train)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# argparse for passing command line interface
parser = argparse.ArgumentParser(description="Training data")
parser.add_argument(
    "--checkpoint",
    type=str,
    help="HF hub model name (default: 'ai4bharat/indic-bert')",
    default="ai4bharat/indic-bert",
)
parser.add_argument(
    "--from_scratch",
    type=bool,
    default=False,
    help="initialize model weights as random",
)

args = parser.parse_args()
print(args.checkpoint)

# getting command-line arguments
model_name = args.checkpoint
from_scratch = args.from_scratch

# loading model based on arguments
model_name = "ai4bharat/indic-bert"
tokenizer, model = load_model_tokenizer(
    model_name=model_name, from_scratch=from_scratch
)

# loading raw texts
text_train = load_text("sanskrit_corpus_train.txt")
text_eval = load_text("sanskrit_corpus_eval.txt")

# batch tokenizing the texts
tokenized_train = tokenizer(text_train, padding=True, truncation=True, max_length=128)
tokenized_eval = tokenizer(text_eval, padding=True, truncation=True, max_length=128)

# making a torch Dataset object for tokenized sentences
dataset_train = Dataset(tokenized_train)
dataset_eval = Dataset(tokenized_eval)

# Data collator does the masking of input tokens while training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

# https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/trainer#transformers.TrainingArguments
# model training arguments can be changed
training_args = TrainingArguments(
    output_dir=f"./results_scratch_{str(from_scratch)}",  # helps separating folders for two models
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    do_train=True,
    do_eval=True,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# training started
trainer.train()
