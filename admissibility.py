#!/usr/bin/python3
# -*- coding: utf-8 -*-

import re
import article6
import pandas as pd
from sqlalchemy import create_engine
from collections import Counter

from gensim.models import Word2Vec, KeyedVectors
from sklearn.manifold import TSNE

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class NoFactSectionError(Exception):
    pass


class NoComplainSectionError(Exception):
    pass


class NoDecisionError(Exception):
    pass


#0 - admissible
#1 - partly admissible
#2 - inadmissible
#3 - struck out of the list
def admissibility_anal(desc):
    if not desc:
        raise NoDecisionError
    if 'Admissible' in desc:
        return 0
    elif 'Partly admissible' in desc:
        return 1
    elif 'Inadmissible' in desc:
        return 2
    else:
        return 3


#0 - admissible/partly admissible
#1 - no
def admissibility_anal_simple(desc):
    if not desc:
        raise NoDecisionError
    if 'Admissible' in desc or 'Partly admissible' in desc:
        return 0
    else:
        return 1


def extract_decision(text, simple=True):
    try:
        for pattern in ('the Court unanimously', 'the Commission', 'the Court by a majority', 'Now, therefore, the Commission'):
            if pattern in text:
                desc = text.split(pattern)
                break

        if simple:
            return admissibility_anal_simple(desc)
        else:
            return admissibility_anal(desc)
    except:
        return None


factre = re.compile(r'(PROCEDURE|FACTS|Summary of the facts|Summary of the relevant facts)(.*?)?(COMPLAINT|THE LAW|FINDING|PROCEEDINGS)', re.DOTALL)
def extract_fact(text):
    try:
        return factre.search(text)[2]
    except:
        raise NoFactSectionError


#0 – The right to a court
#1 – Independent and impartial tribunal established by law
#2 – Fairness of proceedings
#3 – Trial within a reasonable time
#4 – Notification of the accusation – Article 6(3)(a)
#5 – Adequate time and facilities to prepare defence – Article 6(3)(b)
#6 – The right to defend oneself or to legal representation – Article 6(3)(c)
#7 – The right to examine witnesses – Article 6(3)(d)
#8 – Free assistance of an interpreter – Article 6(3)(e)
def clustering(text):
    try:
        complaints = re.findall(r'(?:COMPLAINT|THE LAW)(.*)', text, re.DOTALL)[0]
        # complaints = text.split('COMPLAINT')[1]
    except:
        # raise NoComplainSectionError
        return -2
    if 'Article 6 § 1' in complaints:
        if 'right of access to court' in complaints:
            return 0
        elif 'independent and impartial tribunal' in complaints:
            return 1
        elif 'fairness of proceedings'in complaints:
            return 2
        elif 'unreasonable length of the proceedings' in complaints:
            return 3
        else:
            return -1
    elif 'Article 6 § 2' in complaints:
        return 4
    # else:
        # print(text)

def load_data(dbname='echr_art6.sqlite'):
    engine = create_engine('sqlite:///'+dbname, echo=True)
    data = pd.read_sql('Decisions', engine)

    X = []
    Y = []
    types = []
    unused = []

    nocomplain = []
    for idx, case in data.iterrows():
        # print(case)
        try:
            if case['text']:
                fact = extract_fact(case['text'])
                # decision = admissibility_anal(case['conclusion'])
                decision = admissibility_anal_simple(case['conclusion'])
                X.append(fact)
                Y.append(decision)
                types.append(clustering(case['text']))

                if clustering(case['text']) == -2:
                    nocomplain.append((case['appno'], case['docname']))

        except NoFactSectionError:
            unused.append((case['appno'], case['docname']))
        except NoDecisionError:
            decision = extract_decision(case['text'])
            if decision:
                X.append(fact)
                Y.append(decision)
                types.append(clustering(case['text']))
            else:
                print(case['appno'], case['docname'])

    print('Total: ', len(X))
    print('Unused: ', len(unused))
    print(unused[:20])

    print(nocomplain[:20])
    print(Counter(types).most_common())

    return X, Y


def train_embeddings(data):
    model_sg = Word2Vec(data, size=200, workers=14, sg=1, window=5)
    model_sg.save('./admissibility_model_sg')
    model_cbow = Word2Vec(data, size=200, workers=14, sg=0, window=10)
    model_cbow.save('./admissibility_model_cbow')


def load_balanced_data(dbname='echr_art6.sqlite'):
    engine = create_engine('sqlite:///'+dbname, echo=True)
    data = pd.read_sql('Decisions', engine)

    ytemp = []
    for idx, case in data.iterrows():
        try:
            if case['text']:
                # decision = admissibility_anal(case['conclusion'])
                decision = admissibility_anal_simple(case['conclusion'])
                ytemp.append(decision)
        except NoFactSectionError:
            # unused.append((case['appno'], case['docname']))
            pass
        except NoDecisionError:
            # print(case['appno'], case['docname'])
            pass

    limit = Counter(ytemp).most_common()[-1][1]
    counts = {}

    X = []
    Y = []
    types = []
    unused = []

    for idx, case in data.iterrows():
        try:
            if case['text']:
                fact = extract_fact(case['text'])
                # decision = admissibility_anal(case['conclusion'])
                decision = admissibility_anal_simple(case['conclusion'])
                if counts.setdefault(decision, 0) <= limit:
                    X.append(fact)
                    Y.append(decision)
                    types.append(clustering(case['text']))
                    counts[decision] += 1
        except NoFactSectionError:
            unused.append((case['appno'], case['docname']))
        except NoDecisionError:
            decision = extract_decision(case['text'])
            if decision:
                if counts.setdefault(decision, 0) <= limit:
                    X.append(fact)
                    Y.append(decision)
                    types.append(clustering(case['text']))
                    counts[decision] += 1
            else:
                print(case['appno'], case['docname'])

    print('Unused: ', len(unused))
    print(unused[:20])
    print(Counter(types).most_common())
    return X, Y


def get_types(X):
    pass


def predict(X, Y):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

    print(len(Xtrain), len(Ytrain), len(Xtest), len(Ytest))

    m_ga = KeyedVectors.load_word2vec_format('/Users/willskywalker/Documents/Workplace/GoogleNews-vectors-negative300.bin.gz', binary=True)
    m_g = dict(zip(m_ga.wv.index2word, m_ga.wv.vectors))
    m_sg = KeyedVectors.load_word2vec_format('masha_model_cbow_w2v')
    # m_sg = Word2Vec.load('admissibility_model_sg')
    w2v_sg = dict(zip(m_sg.wv.index2word, m_sg.wv.vectors))
    m_cbow = Word2Vec.load('model_cbow_5')
    # m_cbow = Word2Vec.load('admissibility_model_cbow')
    w2v_cbow = dict(zip(m_cbow.wv.index2word, m_cbow.wv.vectors))

    def dummy(doc):
        return doc

    etree_count = Pipeline([
        ("word2vec vectorizer", TfidfVectorizer(analyzer='word',
                                                preprocessor=dummy,
                                                tokenizer=dummy,
                                                ngram_range=(2, 4))),
        ("extra trees", LinearSVC())])
    etree_w2v = Pipeline([
        # ("word2vec vectorizer", article6.MeanEmbeddingVectorizer(w2v_sg)),
        ("word2vec vectorizer", article6.MeanEmbeddingVectorizer(m_g)),
        # ("word2vec vectorizer", article6.MeanEmbeddingVectorizer(w2v_cbow)),
        ("extra trees", LinearSVC())])
    etree_w2v_tfidf = Pipeline([
        # ("word2vec vectorizer", article6.TfidfEmbeddingVectorizer(w2v_sg)),
        ("word2vec vectorizer", article6.TfidfEmbeddingVectorizer(m_g)),
        # ("word2vec vectorizer", article6.TfidfEmbeddingVectorizer(w2v_cbow)),
        ("extra trees", LinearSVC())])

    etree_count.fit(Xtrain, Ytrain)
    Ypredict = etree_count.predict(Xtest)
    print(confusion_matrix(Ytest, Ypredict))
    print(classification_report(Ytest, Ypredict))

    etree_w2v.fit(Xtrain, Ytrain)
    Ypredict = etree_w2v.predict(Xtest)
    print(confusion_matrix(Ytest, Ypredict))
    print(classification_report(Ytest, Ypredict))

    etree_w2v_tfidf.fit(Xtrain, Ytrain)
    Ypredict = etree_w2v_tfidf.predict(Xtest)
    print(confusion_matrix(Ytest, Ypredict))
    print(classification_report(Ytest, Ypredict))

    # print(cross_validate(etree_count, X, Y, scoring=['precision_micro', 'recall_micro', 'f1_micro']))
    # print(cross_validate(etree_w2v, X, Y, scoring=['precision_micro', 'recall_micro', 'f1_micro']))
    # print(cross_validate(etree_w2v_tfidf, X, Y, scoring=['precision_micro', 'recall_micro', 'f1_micro']))

    return X, Y, Xtrain, Xtest, Ytrain, Ytest, Ypredict


def main():
    X, Y = load_data()
    # train_embeddings(X)
    # X, Y = load_balanced_data()
    # predict(X, Y)


if __name__ == '__main__':
    main()
