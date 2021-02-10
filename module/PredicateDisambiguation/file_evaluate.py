import json
import argparse
import spacy
import Levenshtein
import numpy as np

def dis_lemma(lemma):
    '''process lemmas that are not in frame files using edit distance'''
    if lemma not in all_lemmas:
        distances = [Levenshtein.distance(lemma, l) for l in all_lemmas]
        i = np.argmin(distances)
        return all_lemmas[i]
    else:
        return lemma

def lemmatize(sent, predicates, dis=False):
    '''
    lemmatization using spaCy
    Args:
        sent: word list
        predicates: predicate index list
        dis: whether to use edit distance for alignment
    '''
    sent = [s.lower() for s in sent]
    sent1 = spacy.tokens.doc.Doc(nlp.vocab, sent)
    for name, proc in nlp.pipeline:
        sent1 = proc(sent1)
    plemmas = [sent1[p].lemma_ for p in predicates]
    if dis:
        plemmas = [dis_lemma(l) for l in plemmas]
    return plemmas

def f(a, b):
    a = np.array(a)
    b = np.array(b)
    t = [i == j for i, j in zip(a, b)]
    s = sum(t)/len(t)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path')
    parser.add_argument("--data_path")
    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm')
    frames = json.load(open(args.frames_path))
    all_lemmas = sorted(list(set([k.split('.')[0] for k in frames])))
    data = json.load(open(args.data_path))
    lemma0 = []
    lemma1 = []
    lemma2 = []
    for d in data:
        sent = d['sentence']
        predicates = d['predicates']
        glemmas = d['lemmas']
        plemmas = lemmatize(sent, predicates, dis=True)
        for i in range(len(d['predicates'])):
            pre_str = sent[d['predicates'][i]]
            x0, y0 = d['plemma_ids'][i].split('.')
            x1, y1 = d['plemma_ids1'][i].split('.')
            # 注意，x0是gold lemma，x1是predict lemma，但是如果gold lemma也不在frames文件中的话，那么很可能x0也会为...
            assert x0 == 'X' or x0 == glemmas[i]
            x0 = glemmas[i]
            x1 = plemmas[i]
            lemma0.append(x0+'.'+d['frameset_ids'][i])
            lemma1.append(x0+'.'+y0)
            lemma2.append(x1+'.'+y1)
    print('prediction lemma score: %.4f'%f(lemma0, lemma1))
    print('gold lemma score: %.4f'%f(lemma0, lemma2))
