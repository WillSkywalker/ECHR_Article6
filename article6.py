 #!/usr/bin/python3
 # -*- coding: utf-8 -*-x

import glob, os  # packages to work with folders
import pandas as pd  # package to work with CSV tables
import re  # regular expressions
from nltk import word_tokenize, sent_tokenize, bigrams
from langdetect import detect
import tika
from tika import parser

DIRECTORY = '/Users/willskywalker/Documents/Workplace/HUDOCcrawler/'


def make_eng_txt(filename):
    data = parser.from_file(filename)
    text = data['content']
    print(detect(text))

    text = text.split('\n')
    lines = [i for i in text if i != '' and i != '\t*']  # remove empty lines
    return lines


def load_documents(article, collection, lang='ENG'):
    metadata = pd.read_csv(os.path.join(DIRECTORY, 'Article%d_%s_%s.csv' % (article, collection, lang)))



def main():
    print(make_eng_txt('/Users/willskywalker/Documents/Workplace/HUDOCcrawler/docs/DECISIONS/6/A v. NORWAY.pdf'))
    load_documents('/Users/willskywalker/Documents/Workplace/HUDOCcrawler/docs/')


if __name__ == '__main__':
    main()
