#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import gensim
from dmmath.math_env.sympy_helper import get_all_op2info
import nltk
from collections import OrderedDict, defaultdict
import pickle as pkl
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cdist
from tqdm import tqdm
import argparse

STOP = set(nltk.corpus.stopwords.words("english"))


class Sentence:
    def __init__(self, sentence):
        self.raw = sentence
        normalized_sentence = sentence.replace("‘", "'").replace("’", "'")
        if sentence:
            self.tokens = [x.lower() for x in nltk.word_tokenize(normalized_sentence) if x.isalpha()]
        else:
            self.tokens = []
        self.tokens_without_stop = [t for t in self.tokens if t not in STOP]


all_words = set([])

op2doc = OrderedDict()
op2info = get_all_op2info()
ops = sorted(list(op2info.keys()))
print('all legal ops number: ', len(ops))
for op in ops:
    sent = Sentence(op2info[op].get('doc', 'none'))
    op2doc[op] = sent
    for token in sent.tokens:
        all_words.add(token)

with open('data/word2freq.txt', 'r') as f:
    lines = f.readlines()
word2freq = {}
for line in lines:
    word, freq = line.strip().split()
    word2freq[word] = int(freq)
    all_words.add(word)

small_word2vec_path = './data/dmmath_word2vec.pkl'

if not os.path.exists(small_word2vec_path):
    PATH_TO_WORD2VEC = "./data/GoogleNews-vectors-negative300.bin"
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC, binary=True, limit=1000000)
    small_word2vec = {}
    for word in all_words:
        if word in word2vec:
            embed = word2vec[word]
            small_word2vec[word] = embed
        else:
            print(word)
    with open(small_word2vec_path, 'wb') as f:
        pkl.dump(small_word2vec, f)
    word2vec = small_word2vec
else:
    with open(small_word2vec_path, 'rb') as f:
        word2vec = pkl.load(f)
print('the number words of having embedding vec: ', len(word2vec))
new_word2freq = {}
for word in word2freq:
    if word in word2vec:
        new_word2freq[word] = word2freq[word]
word2freq1 = new_word2freq

word2freq2 = defaultdict(lambda: 0)
for doc in op2doc.values():
    for token in doc.tokens:
        if token in word2vec:
            word2freq2[token] += 1

for op in ops:
    sent = op2doc[op]
    assert sent.tokens and [x in word2vec for x in sent.tokens]


def remove_first_principal_component(X):
    svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
    svd.fit(X)
    pc = svd.components_
    XX = X - X.dot(pc.transpose()) * pc
    return XX


def get_embedding(sentences, model, freqs, use_stoplist=False, a=0.001):
    total_freq = sum(freqs.values())
    embeddings = []
    for sent in sentences:
        tokens = sent.tokens_without_stop if use_stoplist else sent.tokens
        tokens = [token for token in tokens if token in model]
        assert tokens
        weights = [a / (a + freqs.get(token, 0) / total_freq) for token in tokens]
        embedding = np.average([model[token] for token in tokens], axis=0, weights=weights)
        embeddings.append(embedding)
    return np.array(embeddings)


def run_sif_benchmark(sentences1, sentences2, model, freqs1={}, freqs2={}, use_stoplist=False, a=0.001):
    embeddings1 = get_embedding(sentences1, model, freqs1, use_stoplist=use_stoplist, a=a)
    embeddings2 = get_embedding(sentences2, model, freqs2, use_stoplist=use_stoplist, a=a)

    # embeddings1 = remove_first_principal_component(embeddings1)
    # embeddings2 = remove_first_principal_component(embeddings2)
    sims = 1 - cdist(embeddings1, embeddings2, 'cosine')
    return sims


parser = argparse.ArgumentParser(
    description='leverage the similarity of question description and sympy api name to warm up the search procedure')
parser.add_argument('--data', type=str, help='data path')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        lines = f.readlines()
    q2anno = {}
    qcnt = 0
    for idx in tqdm(range(len(lines) // 2), total=len(lines) // 2):
        q = lines[2 * idx].strip()
        a = lines[2 * idx + 1].strip()
        qs = Sentence(q)
        q2anno[qs] = a

    ques_sents = list(q2anno.keys())
    doc_sents = [op2doc[op] for op in ops]

    sims = run_sif_benchmark(ques_sents, doc_sents, word2vec, freqs1=word2freq1, freqs2=word2freq2, use_stoplist=False,
                             a=0.001)
    for qidx, qs in tqdm(enumerate(ques_sents), total=len(ques_sents)):
        op_sim_list = sorted(zip(ops, sims[qidx, :]), key=lambda x: x[1], reverse=True)
        include_ops = [op for op in ops if op in ' '.join(qs.tokens)]
        op_sim_list = [(x, 1.0) for x in include_ops] + op_sim_list
        chosen_ops = [x[0] for x in op_sim_list[:20]]
        for op, sim in op_sim_list[:50]:
            print(op, sim)
        print()
