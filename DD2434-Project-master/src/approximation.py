####rint
####		Approximated version of the kernel


# import pyximport
#
# pyximport.install()
# from ssk_cython_single import kernel

import time
import numpy as np
import itertools
from math import sqrt

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import MultiLabelBinarizer as multilabelbinarizer
from sklearn.metrics import precision_recall_fscore_support

from multiprocessing import Pool, cpu_count

from SSK import kernel
from preprocessing import process_directory, process_file, preprocess_regex, data
from Gram_matrix import GramCalc
from tfidf import train_target, evaluate

# from functools import lru_cache


""" frobenius inner product between two gram matrices """


def frobenius(K1, K2):
    if (len(K1) != len(K2) or len(K1[0]) != len(K2[0])):
        print('invalid dimension for frobenius inner product')
        return -1;

    res = 0
    for i in range(len(K1)):
        for j in range(len(K1[0])):
            res += K1[i][j] * K2[i][j]
    return res;


""" Gram matrix for the sample S using approximated kernel
should run in O(mnn'lt+lm^2) where 
m = the number of documents in data set,
n = length of substrings considered,
n' = length of entries in S,
l = number of entries in S
t = average length of document in data set.
This constitues a saving when compared to SSK gram matrix if
l < nt^2 and n'l < mt"""


def gram_matrix_approx(n, S, X, Y):
    Gram = np.zeros((len(X), len(Y)))

    if (X == Y):
        # print('X == Y')

        ###	non parallel
        partial_kernels = np.zeros((len(X), len(S)))
        for ids, s_i in enumerate(S):
            for idx, x in enumerate(X):
                #				print(kernel(x, s_i, n))
                partial_kernels[idx][ids] = kernel(x, s_i, n);

        # print('done w partial kernels')
        for idx, s in enumerate(X):
            for jdx, t in enumerate(Y):
                xx = np.dot(partial_kernels[idx], partial_kernels[idx])
                yy = np.dot(partial_kernels[jdx], partial_kernels[jdx])
                xy = np.dot(partial_kernels[idx], partial_kernels[jdx])
                if (xx * yy == 0):  # to avoid NaN
                    Gram[idx, jdx] = 0
                else:
                    Gram[idx, jdx] = xy / sqrt(xx * yy)  # not sure if this is correct
    else:
        print('X != Y')

        partial_kernels_X = np.zeros((len(X), len(S)))
        partial_kernels_Y = np.zeros((len(Y), len(S)))

        for ids, s_i in enumerate(S):
            for idx, x in enumerate(X):
                #				print('one calculation')
                partial_kernels_X[idx][ids] = kernel(x, s_i, n);
            for jdx, y in enumerate(Y):
                partial_kernels_Y[jdx][ids] = kernel(y, s_i, n);
        print('done w partial kernels')

        for idx, s in enumerate(X):
            for jdx, t in enumerate(Y):
                xx = np.dot(partial_kernels_X[idx], partial_kernels_X[idx])
                yy = np.dot(partial_kernels_Y[jdx], partial_kernels_Y[jdx])
                xy = np.dot(partial_kernels_X[idx], partial_kernels_Y[jdx])
                if (xx * yy == 0):  # to avoid NaN
                    Gram[idx, jdx] = 0
                else:
                    Gram[idx, jdx] = xy / sqrt(xx * yy)

    return Gram


def calculate_partial_kernels(dataset, S, n):
    partial_kernels = np.zeros((len(dataset), len(S)))
    for ids, s_i in enumerate(S):
        for idx, x in enumerate(dataset):
            partial_kernels[idx][ids] = kernel(x, s_i, n);
    #			print('one calculation')

    return partial_kernels;


#	only appropriate to use when X and Y are not the same
# @lru_cache(maxsize=24)
def gram_matrix_approx_parallel(n, S, X, Y):
    Gram = np.zeros((len(X), len(Y)))

    #	start two threads computing partial kernels for the two data sets
    pool = Pool(1)
    [partial_kernels_X, partial_kernels_Y] = pool.starmap(calculate_partial_kernels, [(X, S, n), (Y, S, n)])
    pool.close()
    pool.join()
    print('done w partial kernels')
    for idx, s in enumerate(X):
        for jdx, t in enumerate(Y):
            xx = np.dot(partial_kernels_X[idx], partial_kernels_X[idx])
            yy = np.dot(partial_kernels_Y[jdx], partial_kernels_Y[jdx])
            xy = np.dot(partial_kernels_X[idx], partial_kernels_Y[jdx])
            if (xx * yy == 0):  # to avoid NaN
                Gram[idx, jdx] = 0
            else:
                Gram[idx, jdx] = xy / sqrt(xx * yy)

    return Gram


def create_gram_matrix(n, S, X, Y, kerneltype):
    if (kerneltype is None):
        GC_train = GramCalc(X, Y, n, kernel=kernel, symmetric=True)
        Gram_train_matrix = GC_train.calculate(parallel=True)
        return Gram_train_matrix
    elif (kerneltype == 'approx'):
        return gram_matrix_approx(n, S, X, Y)
    else:
        return -1  # not supported


""" The (empirical) alignment of a kernel k1 with a kernel k2 with
respect to the sample S
n = length of substrings considered
S = sample of n-grams
X, Y = data sets
kerneltypes = [kerneltype for k1, kerneltype for k2],
where the possible kerneltypes are None = SSK and 'approx' = approximated kernel """


def gram_matrix_alignment(n, S, X, Y, kerneltypes):
    print('building gram matrix ...')
    start = time.time()
    K1 = create_gram_matrix(n, S, X, Y, kerneltypes[0])
    end = time.time()
    print('\nelapsed time: ', (end - start) / 60, 'min')

    print('building gram matrix ...')
    start = time.time()
    K2 = create_gram_matrix(n, S, X, Y, kerneltypes[1])
    end = time.time()
    print('\nelapsed time: ', (end - start) / 60, 'min')

    print(K1, 'normal kernel')
    print(K2, 'approximated kernel')

    K12 = frobenius(K1, K2)
    K11 = frobenius(K1, K1)
    K22 = frobenius(K2, K2)
    res = K12 / sqrt(K11 * K22)

    return res;


""" Extracts all n-grams from a data set """


def ngram_extraction(X, n):
    ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))
    term_doc_matrix = ngram_vectorizer.fit_transform(X)
    names = ngram_vectorizer.get_feature_names()

    return term_doc_matrix, names;


""" We choose the k (contiguous) strings of length n which occur 
most frequently in the dataset and this forms our set S̃. """


def form_S(X, n, k):
    term_doc_matrix, ngrams = ngram_extraction(X, n)
    feature_vectors = term_doc_matrix.toarray()

    total = feature_vectors.sum(axis=0)
    sorted_tot = np.argsort(total)[::-1]  # sorted in descending order
    most_common = [ngrams[sorted_tot[i]] for i in range(k)]

    return most_common;


#	testing similarity measure
def test():
    n = 3
    k = 20
    # lil data set
    X = [
        "The support vector machine (SVM) is a powerful learning algorithm, e.g., for classification and clustering tasks, that works even for complex data structures such as strings, trees, lists and general graphs.",
        "It is based on the usage of a kernel function for measuring scalar products between data units.",
        "For analyzing string data Lodhi et al. (J Mach Learn Res 2:419–444, 2002) have introduced a String Subsequence kernel (SSK)"]

    X = [preprocess_regex(x) for x in X]
    term_doc_matrix, names = ngram_extraction(X, n)
    max_ngrams = len(names);
    kerneltypes = [None, 'approx']

    S = form_S(X, n, max_ngrams)  # form set of k most common n-grams in the data set
    for k in range(10, max_ngrams, 10):
        currentS = S[0:k];
        #	measure of similarity between gram matrix made by SSK and by approximated version of the kernel
        similarity = gram_matrix_alignment(n, currentS, X, X, kerneltypes);
        print('k = ', k, 'similarity between gram matrices = ', similarity)


#	get number of occurrences of ten classes in the reuters training set and test set
def get_num_of_classes():
    relevant_classes = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']
    train_ids, test_ids, _, texts, classes = process_directory(approx=1)
    train_classes = []
    test_classes = []

    for i in train_ids:
        train_classes.append(classes[i]);
    for i in test_ids:
        test_classes.append(classes[i]);

    train_classes_flat = []
    test_classes_flat = []
    for i in train_classes:
        for j in i:
            train_classes_flat.append(j);

    for i in test_classes:
        for j in i:
            test_classes_flat.append(j);

    train_nums = np.zeros(len(relevant_classes))
    test_nums = np.zeros(len(relevant_classes))

    for idx, i in enumerate(relevant_classes):
        train_nums[idx] = train_classes_flat.count(i)
        test_nums[idx] = test_classes_flat.count(i)

    print(train_nums, 'train', test_nums, 'test')


# preliminary experiment w svm
def experiment():
    n = 3
    num_features = 100
    nprime = 3
    l = 100

    X_train = [
        "The support vector machine (SVM) is a powerful learning algorithm, e.g., for classification and clustering tasks, that works even for complex data structures such as strings, trees, lists and general graphs.",
        "It is based on the usage of a kernel function for measuring scalar products between data units.",
        "For analyzing string data Lodhi et al. (J Mach Learn Res 2:419–444, 2002) have introduced a String Subsequence kernel (SSK)",
        "Kernels exist in popcorn and computer science. We will focus on the computer science version of a kernel."]
    X_train = [preprocess_regex(x) for x in X_train]
    y_train = [i for i in range(len(X_train))]

    lenX = [len(x) for x in X_train]
    t = np.average(lenX)
    m = len(X_train)

    print("t = ", t, " m = ", m, " n' = ", nprime, " l = ", l, " n = ", n)
    print(l, " < ", n * t ** 2, 'and', nprime * l, ' < ', m * t, '?')

    X_test = ["It is based on the usage of a kernel function for measuring scalar products between data units.",
              "Kernels exist in corn-pop and computer scifi. We will focus on the computer myspace version of a kernel.",
              "For analyzing string data Lodhi et al. have introduced a String Subsequence popcorn-kernel (SSPK)",
              "The support vector machine (SVM) is a fearful learning algorithm, e.g., for fication and flustering tasks, that borks even for complex data structures such a strings, tree, list and general ."]
    X_test = [preprocess_regex(x) for x in X_test]
    y_test = [1, 3, 2, 0]

    S = form_S(X_train, nprime, num_features)  # form set of k most common n-grams in the data set

    print('building gram matrix')
    gram_train = gram_matrix_approx(n, S, X_train, X_train)
    sim = gram_matrix_alignment(n, S, X_train, X_train, [None, 'approx']);
    print('matrix alignment = ', sim)
    print('done')

    classifier = OneVsRestClassifier(SVC(kernel='precomputed'))
    classifier.fit(gram_train, y_train)

    print('building gram matrix')
    gram_test = create_gram_matrix(n, S, X_train, X_test, kerneltype='approx')
    print('done')
    y_pred = classifier.predict(gram_test)

    print(y_test)
    print(y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred)

    print(f1, 'f1')
    return f1;


#	term_doc_matrix = ngram_extraction(X_train, 3)
#	# not sure why its not the same here. might be a preprocessing thing,
#	# but they seem to mention that they only do removal of stopwords which wouldnt explain why i get MORE trigrams than them
#	print('num unique trigrams = ', term_doc_matrix.shape[1], 'should be 8727 according to the report')

# running svm on two files
def experiment3():
    n = 3  # length of substrings considered
    num_features = 100  # number of elements in S
    n_train_samples = 100
    n_test_samples = 40

    categories = ['earn', 'acq', 'ship', 'corn']
    _, test_ids, _, texts_, classes_ = process_file('../data/reut2-021.sgm', categories)
    train_ids, _, _, texts, classes = process_file('../data/reut2-005.sgm', categories)
    print(len(train_ids), len(test_ids), len(classes))
    texts = {**texts_, **texts}
    classes = {**classes_, **classes}

    X_train, y_train, X_test, y_test = data(train_ids, test_ids, \
                                            texts, classes, n_train_samples, n_test_samples)

    y_train, mlb = train_target(y_train, filter_classes=categories)
    y_test = mlb.transform(y_test)

    print(len(X_train), len(y_train), len(X_test), len(y_test))

    ###
    #	Just a little thing to check if the approximation is "worth it"
    lenX = [len(x) for x in X_train]
    t = np.average(lenX)
    m = len(X_train)
    nprime = n
    l = num_features
    print("t = ", t, " m = ", m, " n' = ", nprime, " l = ", l, " n = ", n)
    print(l, " < ", n * t ** 2, 'and', nprime * l, ' < ', m * t, '?')
    ###

    S = form_S(X_train, nprime, num_features)  # form set of k most common n-grams in the data set

    print('building gram matrix parallel')
    start = time.time()
    gram_train_parallel = gram_matrix_approx_parallel(n, S, X_train, X_train);
    end = time.time()
    print('\nelapsed time: ', (end - start) / 60, 'min')
    print(gram_train_parallel, 'training gram matrix')

    print('building gram matrix non-parallel')
    start = time.time()
    gram_train = gram_matrix_approx(n, S, X_train, X_train)
    end = time.time()
    print('\nelapsed time: ', (end - start) / 60, 'min')
    print(gram_train, 'training gram matrix')

    print('parallel and non-parallel equal?', np.array_equal(gram_train, gram_train_parallel))
    print('does gram_train contain NaN?', np.isnan(gram_train).any())

    print('classifier')
    classifier = OneVsRestClassifier(SVC(kernel='precomputed'))
    classifier.fit(gram_train, y_train)

    print('building gram matrix parallel')
    start = time.time()
    gram_test_parallel = gram_matrix_approx_parallel(n, S, X_test, X_train);
    end = time.time()
    print('\nelapsed time: ', (end - start) / 60, 'minutes')
    print(gram_test_parallel, 'testing gram matrix')

    print('building gram matrix non-parallel')
    start = time.time()
    gram_test = gram_matrix_approx(n, S, X_test, X_train)
    end = time.time()
    print('\nelapsed time: ', (end - start) / 60, 'minutes')
    print(gram_test, 'testing gram matrix')

    print('parallel and non-parallel equal?', np.array_equal(gram_test, gram_test_parallel))
    print('does gram train contain NaN?', np.isnan(gram_test).any())

    print('making prediction')
    y_pred = classifier.predict(gram_test)

    _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    classes = ['earn', 'acq', 'ship', 'corn']
    for l in classes:
        i = np.where(mlb.transform([(l,)]) == 1)[1][0]
        print('f1 ', l, f1[i])

    return f1;


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    experiment3()
