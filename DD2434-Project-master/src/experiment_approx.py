import time
import numpy as np

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support

import pickle

from preprocessing import process_directory, data, process_file
from tfidf import train_target
from approximation import gram_matrix_approx, gram_matrix_approx_parallel, form_S


def save_gram(gram_matrix, str_type, n):
    print('saving gram matrix ...')
    name = "gram_" + "480" + str_type + str(n) + ".dat"
    gram_matrix.dump(name)


def save_results(f1_scores, precisions, recalls, supports):
    name = 'f1_scores_480.dat'
    f1_scores.dump(name)
    name = 'precisions_480.dat'
    precisions.dump(name)
    name = 'recalls_480.dat'
    recalls.dump(name)
    name = 'supports_480.dat'
    supports.dump(name)


def getData(train_ids, test_ids, texts, classes):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in train_ids:
        X_train.append(texts[i])
        y_train.append(classes[i])

    for i in test_ids:
        X_test.append(texts[i])
        y_test.append(classes[i])

    return X_train, y_train, X_test, y_test


# """ Running the experiment from section 7 in the report,
# but with a smaller number of documents bc no time to run it all """
def main():
    n = [4,5]
    num_features = 500
    categories = ['earn', 'acq', 'money-fx', 'grain', 'crude']

    _, _, _, texts, classes = process_directory(approx=1)

    with open('train_Mathilda', 'rb') as f:
        train_ids = pickle.load(f)
    with open('test_Mathilda', 'rb') as f:
        test_ids = pickle.load(f)

    train_ids = train_ids['ids']
    test_ids = test_ids['ids']

    X_train, y_train, X_test, y_test = getData(train_ids, test_ids, texts, classes)

    y_train, mlb = train_target(y_train)
    y_test = mlb.transform(y_test)
    mlbclasses = mlb.classes


    print("training set size = ", len(X_train), len(y_train), "test set size = ", len(X_test), len(y_test))

    f1_scores = np.zeros((len(n), len(categories)))
    precisions = np.zeros((len(n), len(categories)))
    recalls = np.zeros((len(n), len(categories)))
    supports = np.zeros((len(n), len(categories)))

    for index_n, n_ in enumerate(n):

        S = form_S(X_train, n_, num_features)  # form set of k most common n-grams in the data set. assumes n' = n
        print('building gram matrix ...')
        start = time.time()
        gram_train = gram_matrix_approx(n_, S, X_train, X_train)
        end = time.time()
        print('\nelapsed time: ', (end - start) / 60, 'minutes')

        save_gram(gram_train, 'train', n_)

        classifier = OneVsRestClassifier(SVC(kernel='precomputed'))
        classifier.fit(gram_train, y_train)

        print('building gram matrix')
        start = time.time()
        gram_test = gram_matrix_approx_parallel(n_, S, X_test, X_train)
        end = time.time()
        print('\nelapsed time: ', (end - start) / 60, 'minutes')

        save_gram(gram_test, 'test', n_)
        y_pred = classifier.predict(gram_test)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        print(precision, recall, f1, support)

        for index_l, l in enumerate(categories):
            i = mlbclasses.index(l)
            f1_scores[index_n][index_l] = f1[i]
            precisions[index_n][index_l] = precision[i]
            recalls[index_n][index_l] = recall[i]
            supports[index_n][index_l] = support[i]

    save_results(f1_scores, precisions, recalls, supports)

if __name__ == '__main__':
    main()
