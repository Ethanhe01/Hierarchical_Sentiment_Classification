import gzip
import argparse
import logging
import json
import pickle as pkl
import spacy
import itertools
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
from gensim.models import word2vec
import numpy as np

from tqdm import tqdm
from random import randint,shuffle
from collections import Counter

MAX_LENGTH = 200000


def count_lines(file):
    count = 0
    for _ in file:
        count += 1
    file.seek(0)
    return count


def data_generator(data):
    with gzip.open(args.input,"r") as f:
        for x in tqdm(f,desc="Reviews",total=count_lines(f)):
            x = x.decode('utf-8').strip()
            d = json.loads(x)
            yield d


def to_array_comp(doc):
        return [[w.orth_ for w in s] for s in doc.sents]


def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser,to_array_comp)


def n_text(text):
    if text is None or len(text)==0:
        text = '.'
    tlen = len(text)
    if tlen > MAX_LENGTH:
        print(text, tlen)
        text = text[:MAX_LENGTH]
        tlen = MAX_LENGTH
    punc = string.punctuation
    if text[-1] not in punc:
        text = text + '.'
    return text


def proc_para(Id, text):
    sens = sent_tokenize(text)
    sens_words = []
    plen = len(sens)
    for s in sens:
        ws = word_tokenize(s)
        ws = [w.lower() for w in ws]
        ws.append('<ssss>')
        sens_words.append(ws)
    return sens_words, plen


def build_dataset(args):

    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    nlp = spacy.load('en', create_pipeline=custom_pipeline)
    gen_a,gen_b,gen_c = itertools.tee(data_generator(args.input),3)
    print("begin processing data")
    data = [
        (
            z["reviewerID"],
            z["asin"],
            tok,
            z["overall"],
            tok_sum,
            z['helpful'][0],
            z['reviewTime']
        )
        for z,tok,tok_sum in zip(
            tqdm((z for z in gen_a),desc="reading file"),
            nlp.pipe((n_text(x["reviewText"]) for x in gen_b), batch_size=10, n_threads=20),
            nlp.pipe((n_text(x["summary"]) for x in gen_c), batch_size=10, n_threads=20)
        )
    ]
    print("finish processing data")

    print(data[0])
    shuffle(data)

    splits = [randint(0,args.nb_splits-1) for _ in range(0,len(data))]
    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    return {"data":data,"splits":splits,"rows":("user_id","item_id","review","rating","summary","helpful","reviewTime")}


def main(args):
    ds = build_dataset(args)
    pkl.dump(ds,open(args.output,"wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="sentences.pkl")
    parser.add_argument("--nb_splits",type=int, default=5)
    args = parser.parse_args()

    main(args)