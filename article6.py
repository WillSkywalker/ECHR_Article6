 #!/usr/bin/python3
 # -*- coding: utf-8 -*-x

import glob, os  # packages to work with folders
import pandas as pd  # package to work with CSV tables
import numpy as np
import matplotlib.pyplot as plt
import re  # regular expressions
from functools import partial
from nltk import word_tokenize, sent_tokenize, bigrams
# import tika
# from tika import parser
import pdftotext
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


DIRECTORY = '/Users/willskywalker/Documents/Workplace/HUDOCcrawler/'


def make_eng_txt(article, doctype, docname):
    filename = os.path.join(DIRECTORY, 'docs/', doctype, str(article)+'/', docname.replace('/', '_'))
    print(filename)
    # print(type(filename))
    if os.path.exists(filename+'.txt'):
        with open(filename+'.txt') as f:
            text = f.read()
    else:
        with open(filename+'.pdf', "rb") as f:
            try:
                pdf = pdftotext.PDF(f)
                text = "\n\n".join(pdf)
                with open(filename[:-4]+'.txt', 'w') as g:
                    g.write(text)
            except pdftotext.Error:
                return []
    # text = data['content']

    text = text.split('\n')
    lines = []
    for line in text:
        if line != '' and line != '\t*':
            for word in line.split():
                # lines.append(word_tokenize(word))
                lines.append(word)
    # lines = [word_tokenize(i) for word in i.split() for i in text if i != '' and i != '\t*']  # remove empty lines
    return lines


def tokenize_line(line):
    return [word_tokenize(word) for word in line.split()]


def load_documents_w2v(article, lang='ENG'):
    collections = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'COMMUNICATEDCASES', lang)))
    decisions = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'DECISIONS', lang)))
    judgements = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'JUDGMENTS', lang)))
    a = collections[~collections['appno'].isin(judgements['appno'].tolist())]
    b = judgements[~judgements['appno'].isin(collections['appno'].tolist())]
    c = decisions[(~decisions['appno'].isin(collections['appno'].tolist())) & (~decisions['appno'].isin(judgements['appno'].tolist()))]
    # trainset = pd.concat([a, b, c])
    doc = []

    for docname in a['docname'].tolist():
        doc.extend(make_eng_txt(article, 'COMMUNICATEDCASES', docname))
    for docname in b['docname'].tolist():
        doc.extend(make_eng_txt(article, 'JUDGMENTS', docname))
    for docname in c['docname'].tolist():
        doc.extend(make_eng_txt(article, 'DECISIONS', docname))
    return doc


def train_embeddings(dataset):
    pass


def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,200), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.most_similar_cosmul(word, topn=10)
    #close_words = ['albania', 'andorra', 'armenia', 'austria', 'azerbaijan', 'belgium', 'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'denmark', 'estonia', 'finland', 'france', 'georgia', 'germany', 'greece', 'hungary', 'iceland', 'ireland', 'italy', 'latvia', 'lithuania', 'luxembourg', 'malta', 'monaco', 'netherlands', 'norway', 'poland', 'portugal', 'moldova', 'romania', 'russia', 'russia', 'marino', 'serbia', 'slovak', 'slovenia', 'spain', 'sweden', 'switzerland', 'macedonia', 'kingdom', 'turkey', 'ukraine',  'albania', 'andorra', 'armenia', 'austria', 'azerbaijan', 'belgium', 'bosnia', 'bulgaria', 'croatia', 'cyprus', 'czech', 'denmark', 'estonia', 'finland', 'france', 'germany', 'hungary']
    close_words = list(set(close_words))
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


def update_database(dbname, article=6, lang='ENG'):
    engine = create_engine('sqlite:///'+dbname, echo=True)
    collections = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'COMMUNICATEDCASES', lang)))
    decisions = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'DECISIONS', lang)))
    judgements = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'JUDGMENTS', lang)))

    def get_text(doctype):
        def func(docname):
            return make_eng_txt(article, doctype, docname)
        return func

    collections['text'] = list(map(get_text('COMMUNICATEDCASES'), collections['docname'].tolist()))
    decisions['text'] = list(map(get_text('DECISIONS'), decisions['docname'].tolist()))
    judgements['text'] = list(map(get_text('JUDGMENTS'), judgements['docname'].tolist()))

    print(len(judgements['text']))
    collections.to_sql('CommunicatedCases', engine)
    decisions.to_sql('Decisions', engine)
    judgements.to_sql('Judgments', engine)

#0 - no violation
#1 - violation
#3 - both 
#2 - not ruled on
def decision_anal(desc):
    if 'No violation of Article 6' in desc:
        if 'Violation of Article 6' in desc:
            return 3
        else:
            return 0
    elif 'Violation of Article 3' in desc:
        return 1
    else:
        return 2

def train_model(article=6, lang='ENG'):
    collections = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'COMMUNICATEDCASES', lang)))
    decisions = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'DECISIONS', lang)))
    judgements = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, 'JUDGMENTS', lang)))
    j_c = set(collections['appno']) & set(judgements['appno'])
    # print(j_c)
    # print(len(j_c))
    X_filelist = collections[collections['appno'].isin(j_c)]['docname']
    X = []
    Y = []
    for appno in j_c:
        docname = collections[collections['appno'] == appno]['docname'].iloc[0]
        print(docname)
        X.append(make_eng_txt(article, 'COMMUNICATEDCASES', docname))
        Y.append(decision_anal(judgements[judgements['appno'] == appno].iloc[0]['conclusion']))

    # print(len(X))
    # print(Y)
    # print(len(set(Y)))
    # print(set(Y))
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

    m_sg = Word2Vec.load('model_sg')
    w2v_sg = dict(zip(m_sg.wv.index2word, m_sg.wv.vectors))
    m_cbow = Word2Vec.load('model_cbow')
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
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v_sg)),
        ("extra trees", LinearSVC())])
    etree_w2v_tfidf = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v_sg)),
        ("extra trees", LinearSVC())])

    print(len(Xtrain), len(Ytrain), len(Xtest), len(Ytest))
    etree_w2v_tfidf.fit(Xtrain, Ytrain)
    Ypredict = etree_w2v_tfidf.predict(Xtest)
    print(confusion_matrix(Ytest, Ypredict))
    print(classification_report(Ytest, Ypredict))

    print(cross_validate(etree_count, X, Y, scoring=['precision_micro', 'recall_micro', 'f1_micro']))
    print(cross_validate(etree_w2v, X, Y, scoring=['precision_micro', 'recall_micro', 'f1_micro']))
    print(cross_validate(etree_w2v_tfidf, X, Y, scoring=['precision_micro', 'recall_micro', 'f1_micro']))

    return X, Y


from collections import defaultdict


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        return self

    def transform(self, X):  # mean
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(list(word2vec.values())[0])

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def make_pipeline(w2v):
    etree_w2v = Pipeline([
        ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
        ("extra trees", LinearSVC())])
    etree_w2v_tfidf = Pipeline([
        ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
        ("extra trees", LinearSVC())])
    for art in [3, 5, 6, 8]:
        print('Article', art)
        Xtrain, Ytrain, Xtest, Ytest = train_per_article(art)
        Ypredict_w2v = cross_val_predict(etree_w2v, Xtrain, Ytrain, cv=10)
        Ypredict_w2v_tfidf = cross_val_predict(etree_w2v_tfidf, Xtrain, Ytrain, cv=10)
        print('Mean vector:\n')
        evaluate(Ytrain, Ypredict_w2v) #0.8934010152284264
        print('TfIdf:\n')
        evaluate(Ytrain, Ypredict_w2v_tfidf) #0.8944162436548223
        print('test')
        etree_w2v.fit(Xtrain, Ytrain)
        etree_w2v_tfidf.fit(Xtrain, Ytrain)
        print('Mean vector:\n')
        Ypredict_w2v_test = etree_w2v.predict(Xtest)
        evaluate(Ytest, Ypredict_w2v_test)
        print('TfIdf:\n')
        Ypredict_w2v_tfidf_test = etree_w2v_tfidf.predict(Xtest)
        evaluate(Ytest, Ypredict_w2v_tfidf_test)


def evaluate(Ytest, Ypredict): #evaluate the model (accuracy, precision, recall, f-score, confusion matrix)
    print('Accuracy:', accuracy_score(Ytest, Ypredict) )
    print('\nClassification report:\n', classification_report(Ytest, Ypredict))
    print('\nConfusion matrix:\n', confusion_matrix(Ytest, Ypredict), '\n\n_______________________\n\n')
    print('F1-score (weighted):', f1_score(Ytest, Ypredict, average='weighted'))
    print('F1-score (macro):', f1_score(Ytest, Ypredict, average='macro'))

def main():
    # print(make_eng_txt('/Users/willskywalker/Documents/Workplace/HUDOCcrawler/docs/DECISIONS/6/A v. NORWAY.pdf'))
    # w2vset = load_documents_w2v(6)
    # model_sg = Word2Vec(w2vset, size=200, workers=14, sg=1, window=5)
    # model_sg.save('./model_sg')
    # model_cbow = Word2Vec(w2vset, size=200, workers=14, sg=0, window=10)
    # model_cbow.save('./model_cbow')
    # w2v_sg = dict(zip(model_sg.wv.index2word, model_sg.wv.vectors))
    # print(model_sg.similar_by_word('judge'))


    m_sg = Word2Vec.load('model_sg')
    w2v_sg = dict(zip(m_sg.wv.index2word, m_sg.wv.vectors))
    m_cbow = Word2Vec.load('model_cbow')
    w2v_cbow = dict(zip(m_cbow.wv.index2word, m_cbow.wv.vectors))

    # display_closestwords_tsnescatterplot(m_sg, 'Russia')
    # update_database('echr_art6.sqlite')

    train_model()



if __name__ == '__main__':
    main()
