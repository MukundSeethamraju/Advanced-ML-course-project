from math import sqrt
from multiprocessing import Pool
from multiprocessing import cpu_count
import time

from preprocessing import process_file
from utils import *
from SSK import kernel

class GramCalc:
    """class to hold information and calculate Gram Matrix efficiently"""

    stored_normalization = None

    def __init__(self, S, T, n, kernel, symmetric=True):
        self.n = n
        self.kernel = kernel

        self.S = S
        self.T = T

        self.mat = np.zeros((len(S), len(T)))
        self.normalized_mat = np.zeros((len(S), len(T)))

        self.train_normalization = np.zeros(len(S))
        self.test_normalization = np.zeros(len(S))

        self.symmetric = symmetric

    @classmethod
    def store_normalization_vars(cls, vars):
        """store computed normalization values"""
        cls.stored_normalization = vars

    def get_stored_normalization(cls):
        return cls.stored_normalization

    def calculate(self, parallel=True):
        """perform all calculations"""

        if parallel:
            self.build_mat_parallel()
        else:
            self.build_mat()

        if self.symmetric:
            self.train_normalization = self.mat.diagonal()
            self.store_normalization_vars(self.train_normalization)
        else:
            self.train_normalization = self.get_stored_normalization()

        self.build_normalized()
        return np.nan_to_num(self.normalized_mat, copy=False)

    def build_mat_parallel(self):
        mat_combos, mat_coords = self.generate_string_combos()

        outputs = self.parallelize(mat_combos)

        for i in range(len(mat_combos)):
            c = mat_coords[i]
            # assymetric case
            # normalization values are stored in negative index
            if c[1] < 0:
                self.test_normalization[c[0]] = outputs[i]
            else:
                self.mat[c[0], c[1]] = outputs[i]

        if self.symmetric:
            self.mat = self.symmetrize(self.mat)

    def parallelize(self, string_vector):
        pool = Pool(cpu_count())
        outputs = pool.map(self.redirect_to_kernel, string_vector, chunksize=1)
        pool.close()
        pool.join()
        return outputs

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
        zipped_sorted = sorted(zipped, key=lambda x: len(x[0][0]) * len(x[0][1]), reverse=True)
        separated = list(zip(*zipped_sorted))
        mat_combos = list(separated[0])
        mat_coords = list(separated[1])

        return mat_combos, mat_coords

    def redirect_to_kernel(self, sc):
        ret = self.kernel(sc[0], sc[1], self.n)
        return ret

    def build_mat(self):
        """precompute kernel on all required combinations"""
        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):

                if self.symmetric and row > col:
                    pass

                else:
                    self.mat[row, col] = self.kernel(s, t, self.n)

        if self.symmetric:
            self.mat = self.symmetrize(self.mat)
        else:
            for idx, s in enumerate(self.S):
                self.test_normalization[idx] = self.kernel(s, s, self.n)

    def build_normalized(self):
        """build normalized gram matrix from precomputed kernel values"""
        for row, s in enumerate(self.S):
            for col, t in enumerate(self.T):

                if self.symmetric and row > col:
                    pass

                elif self.symmetric and row == col:
                    self.normalized_mat[row, col] = 1

                else:
                    self.normalized_mat[row, col] = self.normalize(row, col)

        if self.symmetric:
            self.normalized_mat = self.symmetrize(self.normalized_mat)

    def normalize(self, row, col):
        """normalize gram matrix element"""
        if self.symmetric:
            return self.mat[row, col] / sqrt(self.train_normalization[row] * self.train_normalization[col])

        else:
            return self.mat[row, col] / sqrt(self.test_normalization[row] * self.train_normalization[col])

    @staticmethod
    def symmetrize(matrix):
        return matrix + matrix.T - np.diag(matrix.diagonal())


def main():
    # n_train_samples = 10
    # n_test_samples = 10
    #
    # # what classes to look at
    # filter_classes = ["ship", "corn"]
    #
    # # train_ids, test_ids, _, texts, classes = process_file('../data/reut2-002.sgm')
    # train_ids, test_ids, _, texts, classes = process_directory()
    #
    # train_texts, train_classes, test_texts, test_classes = data(
    #     train_ids, test_ids, texts, classes, n_train_samples, n_test_samples, filter_classes)
    #
    # # length of subsequences
    # n = 2
    #
    # print(train_texts)
    # print(train_classes)
    #
    # # build Gram matrix
    # GC_train = GramCalc(train_texts, train_texts, n, kernel=kernel)
    # Gram_train_matrix = GC_train.calculate()
    #
    # print(Gram_train_matrix)
    #
    # # make classes into vector and multilabel binarizer, list of classes tranformed to binary vector
    # y_train, mlb = train_target(train_classes, filter_classes)
    # y_test = mlb.transform(test_classes)
    #
    # print(test_texts)
    # print(test_classes)
    #
    # GC_test = GramCalc(test_texts, train_texts, n, kernel=kernel)
    # Gram_test_matrix = GC_test.calculate()
    #
    # classifier = OneVsRestClassifier(SVC(kernel='precomputed'))
    # classifier.fit(Gram_train_matrix, y_train)
    #
    # y_pred = classifier.predict(Gram_test_matrix)
    #
    # print(y_pred)
    # print(mlb.inverse_transform(y_pred))

    n = 3

    # train_texts = ["re", 'oo']

    train_texts =  ['grain reserve holdings breakdown us agriculture department ',
                   'brazil coffee exports disrupted strike dayold strike brazilian seamen affecting coffee'
                   'us grain analysts see lower corn soy planting grain analysts surveyed american',
                   'grain certificate redemptions us exporters report tonnes corn switched unknown ussr union shippers agree cut ny port costs new york ',
                   'midwest cash grain slow country movement cash grain dealers reported slow country movement corn ',
                   'brazil seamen continue strike despite court hundreds marines alert key brazilian ports seamen decided remain indefinite strike even higher labour court saturday ruled illegal union leaders said halt first national strike seamen years started february union leaders said would return work unless got pct pay ']

    start = time.time()

    # build Gram matrix
    GC_train = GramCalc(train_texts, train_texts, n, kernel=kernel, symmetric=True)
    Gram_train_matrix = GC_train.calculate(parallel=True)
    print("in main")
    print("Gram train matrix", Gram_train_matrix)
    stop = time.time()

    print("Gram train matrix, elapsed time: ", stop-start)

    test_texts = ['grain certificate redemptions put mln']

    start = time.time()
    GC_test = GramCalc(test_texts, train_texts, n, kernel=kernel, symmetric=False)
    Gram_test_matrix = GC_test.calculate(parallel=True)
    print("Gram test matrix", Gram_test_matrix)
    stop = time.time()

    print("Gram test matrix, elapsed time: ", stop-start)


    # evaluate(y_test, y_pred, mlb, filter_classes)
def get_train_texts(n_samples, min_length, max_length):
    _, _, _, texts, _ = process_file('../data/reut2-000.sgm')
    texts = [text for text in texts.values() if min_length < len(text) < max_length]
    train_texts = texts[:n_samples]
    return sorted(train_texts, key=len, reverse=True)


def test(train_texts):
    n = 2
    GC_train = GramCalc(
        train_texts, train_texts, n, kernel=kernel, symmetric=True)
    Gram_train_matrix = GC_train.calculate(parallel=True)
    return Gram_train_matrix


if __name__ == '__main__':
    main()
