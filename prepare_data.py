import os
import gzip
import argparse
import logging
import json
import pickle as pkl
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


def proc_para(text):
    sens = sent_tokenize(text)
    sens_words = []
    # plen = len(sens)
    for s in sens:
        ws = word_tokenize(s)
        if len(ws) == 0:
            continue
        ws = [w.lower() for w in ws]
        sens_words.append(ws)
    return sens_words


def build_dataset(args):

    print("Building dataset from : {}".format(args.input))
    print("-> Building {} random splits".format(args.nb_splits))

    gen_a,gen_b,gen_c = itertools.tee(data_generator(args.input),3)
    print("begin processing data")
    data = [
        (
            z["reviewerID"],
            z["asin"],
            proc_para(z['reviewText']),
            z["overall"],
            proc_para(z['summary']),
            z['helpful'][0],
            z['reviewTime']
        )
        for z in tqdm(gen_a,desc="reading file")
        if len(z['reviewText']) > 0 and len(z['summary']) > 0 and len(proc_para(z['reviewText'])) > 0 and len(proc_para(z['summary'])) > 0
    ]
    print("finish processing data")

    print(data[0])
    shuffle(data)

    splits = [randint(0,args.nb_splits-1) for _ in range(0,len(data))]
    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    return {"data":data,"splits":splits,"rows":("user_id","item_id","review","rating","summary","helpful","reviewTime")}


def not_empty(s):
    return s and s.strip()


def get_sentences(path):
    d = pkl.load(open(path, 'rb'))
    docs = d['data']
    idx = 0
    for i in docs:
        for j in i[2]:
            lk = list(filter(not_empty, list(j)))
            if idx < 20:
                print(lk)
            idx += 1
            yield(lk)


def train_word2vec(args, subname):
    p_name = args.output.replace('.pkl','')
    model_path = p_name+'_word2vec_model_wmc{}'.format(args.word_min_count)
    if os.path.exists(model_path):
        model = gensim.models.Word2Vec.load(model_path)
    else:
        sentences = list(get_sentences(args.output))
        model = word2vec.Word2Vec(sentences, min_count=args.word_min_count, size=200, workers=10)
        model.save(model_path)

    vocab = model.wv.vocab
    all_wordvec = []
    of_word = open(p_name+'_'+'wordlist_wmc{}.txt'.format(args.word_min_count), 'w')
    of_emb = open(p_name+'_'+'embinit_wmc{}.save'.format(args.word_min_count), 'wb')
    for word in tqdm(vocab, desc='record word_list'):
        print(word, file=of_word)
        all_wordvec.append(model[word])
    pkl.dump(all_wordvec, of_emb, protocol=pkl.HIGHEST_PROTOCOL)
    of_word.close()
    of_emb.close()
    with open(p_name+'_emb_wmc{}'.format(args.word_min_count), 'w') as of:
        print(len(vocab), 200, file=of)
        for word, vec in tqdm(zip(vocab, all_wordvec), desc='record emb'):
            print(word, end=' ', file=of)
            for v in vec:
                print(v, end=' ', file=of)
            print('', end='\n', file=of)


def main(args):
    if not args.only_word2vec:
        ds = build_dataset(args)
        pkl.dump(ds,open(args.output,"wb"))
    train_word2vec(args, '')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="prepared_data.pkl")
    parser.add_argument("--word-min-count", type=int, default=1)
    parser.add_argument("--nb_splits",type=int, default=5)
    parser.add_argument("--only-word2vec",action='store_true',default=False)
    args = parser.parse_args()

    main(args)
