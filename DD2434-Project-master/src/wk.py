#!/usr/bin/env python
'''
Standard word kernel

From the article:
WK is a linear kernel that measures the similarity between documents that are
indexed by words with tfidf weighting scheme.

The entries of the feature vectors are weighted using a variant of tfidf,
log(1+tf) * log(n/df).

tf: term frequency - number of times the term occurs in a document.
df: document frequency - number of documents that contains the term.
n: total number of documents.

The documents are normalised so that each document has equal length.
'''

from collections import Counter
from math import log
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from preprocessing import get_classes
from preprocessing import process_directory
import random
from utils import filter_dict

def create_doc_word_matrix(texts, N):

    corpus = []
    n = len(texts)
    df = Counter()
    document_index = {}

    for text in texts.values():
        words = text.split(' ')
        corpus.extend(words)
        df.update(words)

    most_common_words = Counter(corpus).most_common(N)
    vocabulary = set([word for word, _ in most_common_words])
    word_index = {word: index for index, word in enumerate(vocabulary)}
    matrix = lil_matrix((n, N))

    for i, d in enumerate(texts.items()):
        x = 0
        document_id, text = d
        words = text.split(' ')
        tf = Counter([word for word in words if word in vocabulary])
        for word in tf:
            x = x + 1
            j = word_index[word]
            matrix[i, j] = tfidf(tf[word], df[word], n)
        document_index[document_id] = i

    return document_index, normalize(matrix.tocsr())


def tfidf(tf, df, n):
    """
    Computes the log-tfidf value.

    Computes log(1 + tf) * log(n / df)

    Parameters
    ----------
    tf : int
        Term frequency, the  number of times the term occurs in a document.
    df : int
        Document term frequency, the number of documents that contains the term.
    n : int
        The total number of documents.

    Returns
    -------
    float
        log(1 + tf) * log(n / df)

    """
    return log(1 + tf) * log(n / df)


def make_data(index,
              texts,
              classes,
              n_samples,
              n_features,
              category_filter=None):

    random.shuffle(index)
    index = index[0:n_samples]
    texts = filter_dict(index, texts)
    classes = filter_dict(index, classes)
    document_index, X = create_doc_word_matrix(texts, n_features)
    label_index, y = get_classes(classes, document_index, category_filter=category_filter)

    return document_index, label_index, X, y


def train(index,
          texts,
          classes,
          n_samples=500,
          n_features=1000,
          category_filter=None):

    document_index, label_index, X, y = make_data(
        index, texts, classes, n_samples, n_features, category_filter)

    clf = OneVsRestClassifier(SVC(kernel='linear'))

    return clf.fit(X, y)


def test(clf,
         index,
         texts,
         classes,
         n_samples=100,
         n_features=1000,
         category_filter=None):

    document_index, label_index, X, y = make_data(
        index, texts, classes, n_samples, n_features, category_filter)

    y_predict = clf.predict(X)

    return y, y_predict


def main():
    # read all data
    train_index, test_index, titles, texts, classes = process_directory()

    # params
    categories = ['earn', 'acq', 'ship', 'corn']
    n_train_samples = 5000
    n_test_samples = 100
    n_features = 3000

    classifier = train(
        train_index,
        texts,
        classes,
        n_samples=n_train_samples,
        n_features=n_features,
        category_filter=categories)

    return test(
        classifier,
        test_index,
        texts,
        classes,
        n_samples=n_test_samples,
        n_features=n_features,
        category_filter=categories)

if __name__ == '__main__':
    main()
