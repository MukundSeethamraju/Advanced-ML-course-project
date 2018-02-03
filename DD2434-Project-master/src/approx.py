from experiment_approx import gram_matrix_approx_parallel, form_S, save_gram, getData
from tfidf import train_target
from preprocessing import process_directory
import pickle


def calc_gram_test(X_test, X_train, S, n=3):
    gram_test = gram_matrix_approx_parallel(n, S, X_test, X_train)
    save_gram(gram_test, 'gramtest_{}'.format(str(n)), n)
    return gram_test


def calc_gram_train(X_train, S, n=3):
    gram_train = gram_matrix_approx_parallel(n, S, X_train, X_train)
    save_gram(gram_train, 'gramtrain_{}'.format(str(n)), n)
    return gram_train


def get_data(train_filename='../data/datasets/train_Mathilda',
             test_filename='../data/datasets/test_Mathilda',
             n=3,
             num_features=500,
             categories=['earn', 'acq', 'money-fx', 'grain', 'crude']):

    _, _, _, texts, classes = process_directory(approx=1)

    with open('../data/datasets/train_Mathilda', 'rb') as f:
        train_ids = pickle.load(f)
    with open('../data/datasets/test_Mathilda', 'rb') as f:
        test_ids = pickle.load(f)

    train_ids = train_ids['ids']
    test_ids = test_ids['ids']

    X_train, y_train, X_test, y_test = getData(train_ids, test_ids, texts,
                                               classes)

    y_train, mlb = train_target(y_train)
    y_test = mlb.transform(y_test)
    S = form_S(X_train, n, num_features)
    return X_train, y_train, X_test, y_test, S, mlb


if __name__ == '__main__':
    pass
