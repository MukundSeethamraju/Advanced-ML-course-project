#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from preprocessing import get_all_classes
from pp_class import DataClass
from postprocessing import evaluate, evaluate_multiple_runs, print_results, evaluate_single_run


def train_target(train_classes, filter_classes=[]):
    """make classes into vector for training"""

    if filter_classes:
        class_set = set(filter_classes)
    else:
        class_set = get_all_classes()

    mlb = MultiLabelBinarizer(classes=list(class_set))
    y_train = mlb.fit_transform(train_classes)

    return y_train, mlb


def train(train_text_list,
          train_classes_list,
          n_features=3000,
          filter_classes=[]):

    vectorizer = TfidfVectorizer(
        max_features=n_features,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True)

    X_train = vectorizer.fit_transform(train_text_list)
    y_train, mlb = train_target(train_classes_list, filter_classes)

    classifier = OneVsRestClassifier(SVC(kernel='linear'))
    classifier.fit(X_train, y_train)
    return classifier, vectorizer, mlb


def test(test_text_list, test_classes_list, classifier, vectorizer,
         multilabelbinarizer):

    X_test = vectorizer.transform(test_text_list)
    y_test = multilabelbinarizer.transform(test_classes_list)
    y_pred = classifier.predict(X_test)
    return y_test, y_pred


def main(n_runs, n_train_samples, n_test_samples, n_features, filter_classes):

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for run in range(n_runs):

        train_data = DataClass('train', n_train_samples, 0, 1500,
                               filter_classes, {})
        train_data.build()
        test_data = DataClass('test', n_test_samples, 0, 1500, filter_classes,
                              {})
        test_data.build()

        classifier, vectorizer, mlb = train(
            train_data.texts, train_data.classes, n_features, filter_classes)

        y_test, y_pred = test(test_data.texts, test_data.classes, classifier,
                              vectorizer, mlb)

        p, r, f1, s = evaluate(y_test, y_pred, mlb, filter_classes)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)
        supports.append(s)

    if n_runs == 1:
        return evaluate_single_run(filter_classes, p, r, f1, s)
    else:
        return evaluate_multiple_runs(filter_classes, precisions, recalls,
                                      f1_scores, supports)


def subset_evaluation(n_runs=1):
    r, h = main(n_runs, 380, 90, 3000, ['earn', 'acq', 'crude', 'corn'])
    print(print_results(r, h))
    return r, h


def complete_evaluation():
    r, h = main(10, 10000, 4000, 3000, [
        'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest',
        'ship', 'wheat', 'corn'
    ])

    print(print_results(r, h))
    return r, h


def verification():
    train_data = DataClass('train', 480, 0, 1000, [], {})
    train_data.load('../data/datasets/train_960')
    test_data = DataClass('test', 480, 0, 1000, [], {})
    test_data.load('../data/datasets/test_320')

    clf, vec, mlb = train(train_data.texts, train_data.classes, 1000, [])

    y_test, y_pred = test(test_data.texts, test_data.classes, clf, vec,
                          mlb)

    p, r, f1, s = evaluate(y_test, y_pred, mlb, ['acq', 'crude', 'earn', 'grain', 'money-fx'])
    r, h = evaluate_single_run(['acq', 'crude', 'earn', 'grain', 'money-fx'], p, r, f1, s)
    print(print_results(r, h))

if __name__ == '__main__':
    #r, h = subset_evaluation()
    verification()
