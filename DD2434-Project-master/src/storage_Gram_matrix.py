import pyximport
pyximport.install()
import ssk_cython
from math import sqrt
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
from postprocessing import evaluate, print_results, evaluate_single_run
from pp_class import DataClass
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
# from ssk_cython import kernel

class GramCalc:
    """class to hold information and calculate Gram Matrix efficiently"""

    stored_normalization = None

    def __init__(self, S, T, N, symmetric=True, lam=0.5):
        self.S = S
        self.T = T

        self.N = N + 1
        self.lam = lam
        self.mat = np.zeros((self.N, len(S), len(T)))
        self.normalized_mat = np.zeros((self.N, len(S), len(T)))

        self.train_normalization = np.zeros((self.N, len(S)))
        self.test_normalization = np.zeros((self.N, len(S)))

        self.symmetric = symmetric

    def calculate(self):
        """perform all calculations"""
        self.build_mat_parallel()

        if self.symmetric:
            for n in range(self.N):
                self.train_normalization[n] = self.mat[n].diagonal()
            self.store_normalization_vars(self.train_normalization)

        else:
            self.train_normalization = self.get_stored_normalization()

        self.build_normalized()
        return np.nan_to_num(self.normalized_mat, copy=False)

    def generate_string_combos(self):
        """generate all string combinations required to build gram matrix
        as well as all norm combos"""
        mat_combos = []
        mat_coords = []

        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):
                if self.symmetric and row > col:
                    pass
                else:
                    mat_combos.append([s, t])
                    mat_coords.append([row, col])

        if not self.symmetric:
            # need to calculate normalization values seperately
            for idx, s in enumerate(self.S):
                mat_combos.append([s, s])
                mat_coords.append([idx, -1])

        # sort according to longest string
        zipped = zip(mat_combos, mat_coords)
        zipped_sorted = sorted(
            zipped, key=lambda x: len(x[0][0]) * len(x[0][1]), reverse=True)
        separated = list(zip(*zipped_sorted))
        mat_combos = list(separated[0])
        mat_coords = list(separated[1])

        return mat_combos, mat_coords

    def build_mat_parallel(self):
        mat_combos, mat_coords = self.generate_string_combos()

        outputs = self.parallelize(mat_combos)

        for i in range(len(mat_combos)):
            for n in range(self.N):
                c = mat_coords[i]

                # assymetric case
                # normalization values are stored in negative index
                if c[1] < 0:
                    self.test_normalization[n, c[0]] = outputs[i][n]

                else:
                    self.mat[n, c[0], c[1]] = outputs[i][n]

        if self.symmetric:
            for n in range(self.N):
                self.mat[n] = self.symmetrize(self.mat[n])

    def parallelize(self, string_vector):
        pool = Pool(cpu_count())
        outputs = pool.map(self.redirect_to_kernel, string_vector, chunksize=1)
        pool.close()
        pool.join()
        return outputs

    def redirect_to_kernel(self, sc):
        # ret = kernel(sc[0], sc[1], self.N)


        ko = KernelOperations(sc[0], sc[1], self.N)
        ret = ko.run_all_kernels()
        del ko

        return ret

    def build_normalized(self):
        """build normalized gram matrix from precomputed kernel values"""
        for n in range(self.N):
            for row, s in enumerate(self.S):
                for col, t in enumerate(self.T):

                    if self.symmetric and row > col:
                        pass

                    elif self.symmetric and row == col:
                        self.normalized_mat[n, row, col] = 1

                    else:
                        self.normalized_mat[n, row, col] = self.normalize(
                            n, row, col)

        if self.symmetric:
            for n in range(self.N):
                self.normalized_mat[n] = self.symmetrize(
                    self.normalized_mat[n])

    def normalize(self, n, row, col):
        """normalize gram matrix element"""
        if self.symmetric:
            return self.mat[n, row, col] / sqrt(self.train_normalization[
                n, row] * self.train_normalization[n, col])

        else:
            return self.mat[n, row, col] / sqrt(self.test_normalization[
                n, row] * self.train_normalization[n, col])

    @classmethod
    def store_normalization_vars(cls, vars):
        """store computed normalization values"""
        cls.stored_normalization = vars

    def get_stored_normalization(self):
        return self.stored_normalization

    @staticmethod
    def symmetrize(matrix):
        return matrix + matrix.T - np.diag(matrix.diagonal())


def train_target(train_classes, class_set):
    """make classes into vector for training"""
    mlb = MultiLabelBinarizer(classes=list(class_set))
    y_train = mlb.fit_transform(train_classes)

    return y_train, mlb


def main():
    # max substring length
    N = 7

    D_train, D_test = get_data()

    print("calculating train gram matrix...")
    GC_train = GramCalc(D_train.texts, D_train.texts, N, symmetric=True)
    Gram_train = GC_train.calculate()
    np.save('Gram_train', Gram_train)
    labels_train, mlb = train_target(D_train.classes, D_train.get_class_set())
    print("done calculating train gram matrix")

    print("calculating test gram matrix...")
    labels_test = mlb.transform(D_test.classes)
    GC_test = GramCalc(D_test.texts, D_train.texts, N, symmetric=False)
    Gram_test = GC_test.calculate()
    np.save('Gram_test', Gram_test)
    print("done calculating test gram matrix")

    for n in range(1, N + 1):
        print("fitting...")
        clf = OneVsRestClassifier(SVC(kernel='precomputed'))
        clf.fit(Gram_train[n], labels_train)
        print("done fitting")

        print("predicting...")
        labels_pred = clf.predict(Gram_test[n])
        print("for n = ", n)
        print("evaluation:")
        p, r, f1, s = evaluate(labels_test, labels_pred, mlb,
                               D_train.get_class_set())
        results, headers = evaluate_single_run(D_train.get_class_set(), p, r,
                                               f1, s)
        print(print_results(results, headers))


def big_test(D_train, D_test, N=7):
    # max substring length

    print("calculating train gram matrix...")
    GC_train = GramCalc(D_train.texts, D_train.texts, N, symmetric=True)
    Gram_train = GC_train.calculate()
    np.save('Gram_train_eskil', Gram_train)
    labels_train, mlb = train_target(D_train.classes, D_train.get_class_set())
    print("done calculating train gram matrix")

    print("calculating test gram matrix...")
    labels_test = mlb.transform(D_test.classes)
    GC_test = GramCalc(D_test.texts, D_train.texts, N, symmetric=False)
    Gram_test = GC_test.calculate()
    np.save('Gram_test_banan', Gram_test)
    print("done calculating test gram matrix")

    for n in range(1, N + 1):
        print("fitting...")
        clf = OneVsRestClassifier(SVC(kernel='precomputed'))
        clf.fit(Gram_train[n], labels_train)
        print("done fitting")

        print("predicting...")
        labels_pred = clf.predict(Gram_test[n])
        print("for n = ", n)
        print("evaluation:")
        p, r, f1, s = evaluate(labels_test, labels_pred, mlb,
                               D_train.get_class_set())
        results, headers = evaluate_single_run(D_train.get_class_set(), p, r,
                                               f1, s)
        print(print_results(results, headers))


def get_data():
    # # general params
    # n_min_length = 0
    # n_max_length = 1500
    # filter_classes = ['earn', 'acq', 'crude', 'corn']
    #
    # # train set
    # subset = 'train'
    # n_samples = 190
    # min_n_classes = {'earn': 76, 'acq': 57, 'crude': 38, 'corn': 19}
    # D_train = DataClass(subset, n_samples, n_min_length, n_max_length, filter_classes, min_n_classes)
    # D_train.build()
    # D_train.save("train_file")
    #
    # # test set
    # subset = 'test'
    # n_samples = 45
    # min_n_classes = {'earn': 20, 'acq': 12, 'crude': 8, 'corn': 5}
    # D_test = DataClass(subset, n_samples, n_min_length, n_max_length, filter_classes, min_n_classes)
    # D_test.build()
    # D_test.save("test_file")

    # general params
    n_min_length = 0

    n_max_length = 100
    filter_classes = ['earn', 'acq']

    # train set
    subset = 'train'
    n_samples = 10
    min_n_classes = {'earn': 3, 'acq': 4, 'crude': 3}
    D_train = DataClass(subset, n_samples, n_min_length, n_max_length, filter_classes, min_n_classes)

    D_train.build()
    D_train.save('some_specific_name_that_stands_out_train')
    int_train_texts = []
    for text in D_train.texts:
        int_train_texts.append(ssk_cython.get_array(text))
    D_train.texts = int_train_texts

    # test set
    subset = 'test'

    n_samples = 3
    min_n_classes = {'earn': 1, 'acq': 1, 'crude': 1}
    D_test = DataClass(subset, n_samples, n_min_length, n_max_length, filter_classes, min_n_classes)

    D_test.build()
    D_test.save('just_so_you_dont_lose_it_test')

    int_test_texts = []
    for text in D_test.texts:
        int_test_texts.append(ssk_cython.get_array(text))
    D_test.texts = int_test_texts

    print(len(D_train.texts))
    print(len(D_test.texts))

    return D_train, D_test


def small_test():
    train_texts = [
        'grain reserve holdings breakdown us agriculture department ',
        'brazil coffee exports disrupted strike dayold strike brazilian seamen affecting coffee',
        'us grain analysts see lower corn soy planting grain analysts surveyed american',
        'union shippers agree cut ny port costs new york ',
        'grain certificate redemptions '
    ]
    # 'us exporters report tonnes corn switched unknown ussr'
    # 'midwest cash grain slow country movement cash grain dealers reported slow country movement corn ',
    # 'brazil seamen continue strike despite court hundreds marines alert key brazilian ports seamen decided remain indefinite strike even higher labour court saturday ruled illegal union leaders said halt first national strike seamen years started february union leaders said would return work unless got pct pay ']

    n = 5
    int_train_texts = []
    for text in train_texts:
        int_train_texts.append(ssk_cython.get_array(text))

    import time

    start = time.time()

    # build Gram matrix
    GC_train = GramCalc(int_train_texts, int_train_texts, n, symmetric=True)
    Gram_train_matrix = GC_train.calculate()
    print("in main")
    print("Gram train matrix")
    for i in Gram_train_matrix:
        print(i)
        print()

    stop = time.time()

    print("elapsed time: ", stop -start)

    print("\n")

    start = time.time()

    test_texts = ['grain certificate redemptions put mln']
    int_test_texts = []
    for text in test_texts:
        int_test_texts.append(ssk_cython.get_array(text))
    GC_test = GramCalc(int_test_texts, int_train_texts, n, symmetric=False)
    Gram_test_matrix = GC_test.calculate()

    print("Gram test matrix")
    for i in Gram_test_matrix:
        print(i)
        print()


    stop = time.time()

    print("elapsed time: ", stop - start)


if __name__ == '__main__':
    small_test()
