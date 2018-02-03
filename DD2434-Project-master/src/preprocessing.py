#! /usr/bin/env python
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
import re
import os
import pickle

from utils import shuffle_lists, flatten


def data(train,
         test,
         texts,
         classes,
         n_train_samples=10000,
         n_test_samples=4000,
         filter_classes=[],
         min_n_test_classes=[],
         min_n_train_classes=[]):
    """
    Processes the reuters data to lists of text and classes.

    Reads the Reuters corpus from files,
    filters the data by the documents classes, shuffles it and then returns
    n_train_samples and n_test_samples of data.

    Parameters
    ----------
    n_train_samples : int
        Number of training samples to be returned.
    n_test_samples : int
        Number of test samples to be returned.
    filter_classes : list(str)
        A list of the classes that should be included in the dataset.

    Returns
    -------
    train_texts : list(str) [n_train_samples]
        List of the document texts of the train subset.
    train_classes : list(str) [n_train_samples]
        List of the classes for each document of the train subset.
    test_texts : list(str) [n_test_samples]
        List of the document texts of the test subset.
    test_classes : list(str) [n_train_samples]
        List of the classes for each document of the test subset.
    """

    train = set(train)
    test = set(test)

    train_texts = []
    train_classes = []

    test_texts = []
    test_classes = []

    for document_id, text in texts.items():
        if document_id in train:
            train_texts.append(text)
            train_classes.append(classes[document_id])
        elif document_id in test:
            test_texts.append(text)
            test_classes.append(classes[document_id])

    if filter_classes:

        if min_n_train_classes:
            part_train_texts = []
            part_train_classes = []
            rest_train_texts = []
            rest_train_classes = []
            N = 0
            for min_n, cl in zip(min_n_train_classes, filter_classes):
                train_texts_cl, train_classes_cl = filter_by_class(
                    train_texts, train_classes, [cl])
                train_texts_cl, train_classes_cl = shuffle_lists(
                    train_texts_cl, train_classes_cl)
                part_train_texts.extend(train_texts_cl[:min_n])
                part_train_classes.extend(train_classes_cl[:min_n])
                rest_train_classes.extend(train_classes_cl[min_n:])
                rest_train_texts.extend(train_texts_cl[min_n:])
                N += min_n
            if N < n_train_samples and rest_train_texts and rest_train_classes:
                rest_train_texts, rest_train_classes = shuffle_lists(
                    rest_train_texts, rest_train_classes)
                part_train_texts.extend(rest_train_texts[:n_train_samples - N])
                part_train_classes.extend(
                    rest_train_classes[:n_train_samples - N])

        else:
            train_texts, train_classes = filter_by_class(
                train_texts, train_classes, filter_classes)

        test_texts, test_classes = filter_by_class(test_texts, test_classes,
                                                   filter_classes)

    train_texts, train_classes, test_texts, test_classes = shuffle_lists(
        train_texts, train_classes, test_texts, test_classes)

    train_texts = train_texts[:n_train_samples]
    train_classes = train_classes[:n_train_samples]

    test_texts = test_texts[:n_test_samples]
    test_classes = test_classes[:n_test_samples]

    return train_texts, train_classes, test_texts, test_classes


def new_data(document_ids,
             texts,
             classes,
             n_samples=None,
             n_min_length=None,
             n_max_length=None,
             filter_classes=[],
             min_n_classes=[],
             only_filter_classes=True):
    """
    Processes the reuters data to lists of text and classes.

    Filters the data by document ids, the documents classes, shuffles it and then returns
    either the n_samples or as many as possible.

    Parameters
    ----------
    document_ids : list(int)
        List of document ids.
    texts : dict( int : list(str))
        Dict where the key is a document id and the value is the corresponding document.
    classes : dict( int : list(str))
        Dict where the key is a document id and the value is the corresponding list of the classes.
    n_samples : int (optional)
        Number of documents to be returned.
    n_min_length : int (optional)
        Minimum length of each document.
    n_max_length : int (optional
        Maximum length of each document.
    filter_classes : list(str)
        A list of the classes that should be included in the dataset.
    min_n_classes : list (int)
        A list where each element is equal to the minimum documents of the corresponding
        class in filter_classes.

    Returns
    -------
    picked_texts : list(str) [n_samples]
        List of the document texts of the data subset.
    picked_classes : list(str) [n_samples]
        List of the classes for each document of the data subset.
    picked_ids : list(int) [n_samples]
        List of the document ids of the data subset.
    """

    document_ids = set(document_ids)

    sample_texts = []
    sample_classes = []
    sample_ids = []

    # Get the documents that have an id that is in the document_ids parameter
    for document_id, text in texts.items():
        if document_id in document_ids:
            if n_min_length is not None and n_max_length is not None:
                if n_min_length < len(text) < n_max_length:
                    sample_texts.append(text)
                    sample_classes.append(classes[document_id])
                    sample_ids.append(document_id)

            elif n_min_length is not None:
                if len(text) > n_min_length:
                    sample_texts.append(text)
                    sample_classes.append(classes[document_id])
                    sample_ids.append(document_id)

            elif n_max_length is not None:
                if len(text) < n_max_length:
                    sample_texts.append(text)
                    sample_classes.append(classes[document_id])
                    sample_ids.append(document_id)

            else:
                sample_texts.append(text)
                sample_classes.append(classes[document_id])
                sample_ids.append(document_id)

    sample_texts, sample_classes, sample_ids = shuffle_lists(
        sample_texts, sample_classes, sample_ids)

    if filter_classes:

        if min_n_classes:

            # Sort filter_classes by size, smallest first
            counted_classes = Counter(list(
                flatten(sample_classes))).most_common()
            counted_classes.reverse()
            sorted_filter_classes = [
                c for c, _ in counted_classes if c in filter_classes
            ]

            picked_texts = []
            picked_classes = []
            picked_ids = []
            rest_texts = []
            rest_classes = []
            rest_ids = []
            N = 0

            for min_n, cl in zip(min_n_classes, sorted_filter_classes):
                class_texts, class_classes, class_ids = filter_by_class(
                    sample_texts, sample_classes, sample_ids, [cl],
                    only_filter_classes)
                # Check if document has already been added
                picked_ids_set = set(picked_ids)
                allready_added_document_indices = [
                    i for i, id in enumerate(class_ids) if id in picked_ids_set
                ]
                if allready_added_document_indices:
                    allready_added_document_indices = set(
                        allready_added_document_indices)
                    class_texts = [
                        t for i, t in enumerate(class_texts)
                        if i not in allready_added_document_indices
                    ]
                    class_classes = [
                        c for i, c in enumerate(class_classes)
                        if i not in allready_added_document_indices
                    ]
                    class_ids = [
                        ci for i, ci in enumerate(class_ids)
                        if i not in allready_added_document_indices
                    ]

                picked_texts.extend(class_texts[:min_n])
                picked_classes.extend(class_classes[:min_n])
                picked_ids.extend(class_ids[:min_n])
                rest_classes.extend(class_classes[min_n:])
                rest_texts.extend(class_texts[min_n:])
                rest_ids.extend(class_ids[min_n:])

            N = len(picked_texts)

            # Check if there are enough samples
            if n_samples is not None and N < n_samples and rest_texts and rest_classes:
                picked_texts.extend(rest_texts[:n_samples - N])
                picked_classes.extend(rest_classes[:n_samples - N])
                picked_ids.extend(rest_ids[:n_samples - N])

        else:
            picked_texts, picked_classes, picked_ids = filter_by_class(
                sample_texts, sample_classes, sample_ids, filter_classes,
                only_filter_classes)

    elif n_samples is not None:
        picked_texts = sample_texts[:n_samples]
        picked_classes = sample_classes[:n_samples]
        picked_ids = sample_ids[:n_samples]

    else:
        picked_texts = sample_texts
        picked_classes = sample_classes
        picked_ids = sample_ids

    picked_texts, picked_classes, picked_ids = shuffle_lists(
        picked_texts, picked_classes, picked_ids)

    return picked_texts, picked_classes, picked_ids


def filter_by_class(texts,
                    classes,
                    ids,
                    filter_classes,
                    only_filter_classes=True):
    """
    Removes all documents which classes are not in the filter_classes parameter.

    Parameters
    ----------
    texts : list(str)
        List of document texts.
    classes : list(list(str))
        List of classes for each document.
    filter_classes : list(str)
        List of the classes that should be included.
    only_filter_classes : bool (optional)
        Reduces the new_classes so that it only contains classes from filter_classes

    Returns
    -------
    new_texts : list(str)
        The filtered document list.
    new_classes : list(list(str))
        The filtered document classes list.

    """
    new_texts = []
    new_classes = []
    new_ids = []
    for i, cls in enumerate(classes):
        if only_filter_classes:
            filtered_classes = []
            for cl in cls:
                if cl in filter_classes:
                    filtered_classes.append(cl)
            if filtered_classes:
                new_texts.append(texts[i])
                new_classes.append(filtered_classes)
                new_ids.append(ids[i])
        else:
            if any(cl in cls for cl in filter_classes):
                new_texts.append(texts[i])
                new_classes.append(cls)
                new_ids.append(ids[i])

    return new_texts, new_classes, new_ids


def preprocess_regex(text, approx=None):
    """
    Filters a string.

    Decapitalizes, removes stopwords and all punctuation from the string.

    Parameters
    ----------
    text : string
    Returns
    -------
    string
        The preprocessed string.

    """
    
    if(approx is None):
	    p1 = re.compile(r'([^a-zA-Z\s\'])')  # Removes everything except a-Z, and '
	    p2 = re.compile(r'(\'(\w+)\')')  # Remove quotes
	    text = p2.sub(r'\g<2>', p1.sub(r'', text).lower())
	    words = text.split()
	    word_filter = set(stopwords.words('english'))
	    filtered_words = [
		   word.replace('\'', '') for word in words if word not in word_filter
	    ]
    else:
#	    print('preprocessing with approx')
	    words = text.split()
	    word_filter = set(stopwords.words('english'))
	    filtered_words = [word for word in words if word not in word_filter]
    
    return ' '.join(filtered_words)


def process_directory(path='../data/', class_filter=None, approx=None):
    """
    Reads and preprocesses the Reuters-21578 dataset.

    The dataset is split using the "ModApte"-split.

    Parameters
    ----------
    path : string (optional)
        Path to dataset directory
    class_filter : list(str) (optional)
        A list of which categories the documents should have.

    Returns
    -------
    train : list(int)
        List of document-IDs which have the parameter "TRAIN".
    test : list(int)
        List of document-IDS which have the parameter "TEST".
    titles : dict(int : str)
        Map of document-IDS and the titles of the documents.
    texts : dict(int : str)
        Map of document-IDS and the texts of the documents.
    classes: dict(int : list(str))
        Map of document-IDS and the class or classes of the documents.
    """
    filenames = [
        filename for filename in os.listdir(path)
        if filename.startswith('reut2-')
    ]

    train = []
    test = []
    titles = {}
    texts = {}
    classes = {}

    print('approx is none?', (approx is None))
    for filename in filenames:
        print(filename)
        _train, _test, _titles, _texts, _classes = process_file(
            path + filename, class_filter, approx)
        
        train.extend(_train)
        test.extend(_test)
        titles.update(_titles)
        texts.update(_texts)
        classes.update(_classes)

    return train, test, titles, texts, classes


def process_file(filename, class_filter=None, approx=None):
    """
    Reads and preprocesses a file of the Reuters-21578 dataset.

    The dataset is split using the "ModApte"-split.

    Parameters
    ----------
    filename : string (optional)
        Filename of the sgml file.
    class_filter : list(str) (optional)
        A list of which categories the documents should have.

    Returns
    -------
    train : list(int)
        List of document-IDs which have the parameter "TRAIN".
    test : list(int)
        List of document-IDS which have the parameter "TEST".
    titles : dict(int : str)
        Map of document-IDS and the titles of the documents.
    texts : dict(int : str)
        Map of document-IDS and the texts of the documents.
    classes: dict(int : list(str))
        Map of document-IDS and the class or classes of the documents.
    """
    train = []
    test = []
    titles = {}
    texts = {}
    classes = {}

    print('approx is none?', (approx is None))
    with open(filename, 'r') as sgml_file:
        corpus = BeautifulSoup(sgml_file.read(), 'html.parser')

        for document in corpus('reuters'):

            # Check if document is "ModApte"
            # According to the README (VIII.B.)
            # Training: lewissplit=train, topics=yes
            # Testing: lewissplit=test, topics=yes
            if document['topics'] == 'YES' and (
                    document['lewissplit'] == 'TRAIN'
                    or document['lewissplit'] == 'TEST'):
                document_id = int(document['newid'])
                cls = []

                for topic in document.topics.contents:
                    if class_filter is not None:
                        if any(cl in topic.contents for cl in class_filter):
                            cls.extend(topic.contents)
                    else:
                        cls.extend(topic.contents)
                classes[document_id] = cls
                if document.title is None:
                    title = ''
                else:
                    title = document.title.contents[0]

                titles[document_id] = title

                if document.body is None:
                    body = ''
                else:
                    body = document.body.contents[0]
                    text = title + ' ' + body
                    texts[document_id] = preprocess_regex(text, approx)

                texts[document_id] = preprocess_regex(body, approx)
#                print(texts[document_id])

                if document['lewissplit'] == 'TRAIN':
                    train.append(document_id)
                else:
                    test.append(document_id)

    return train, test, titles, texts, classes


def process_file_2(filename, category_filter=None):
    """
    Reads and preprocesses a file of the Reuters-21578 dataset.

    The dataset is split using the "ModApte"-split.

    Parameters
    ----------
    filename : string (optional)
        Filename of the sgml file.
    category_filter : list(str) (optional)
        A list of which target the documents should have.

    Returns
    -------
    train : list(int)
        List of document-IDs which have the parameter "TRAIN".
    test : list(int)
        List of document-IDS which have the parameter "TEST".
    titles : dict(int : str)
        Map of document-IDS and the titles of the documents.
    texts : dict(int : str)
        Map of document-IDS and the texts of the documents.
    classes: dict(int : list(str))
        Map of document-IDS and the class or classes of the documents.
    """
    train_index_id_map = {}
    train_data = []
    train_target = []

    test_index_id_map = {}
    test_data = []
    test_target = []

    with open(filename, 'r') as sgml_file:
        corpus = BeautifulSoup(sgml_file.read(), 'html.parser')

    for document in corpus('reuters'):

        # Check if document is "ModApte"
        # According to the README (VIII.B.)
        # Training: lewissplit=train, topics=yes
        # Testing: lewissplit=test, topics=yes
        if document['topics'] == 'YES' and (document['lewissplit'] == 'TRAIN'
                                            or
                                            document['lewissplit'] == 'TEST'):
            document_id = int(document['newid'])
            target = []
            data = ''

            for topic in document.topics.contents:
                if category_filter is not None:
                    if any(category in topic.contents
                           for category in category_filter):
                        target.extend(topic.contents)
                else:
                    target.extend(topic.contents)
            if target:
                if document.title is None:
                    title = ''
                else:
                    title = document.title.contents[0]

            if document.body is None:
                body = ''
            else:
                body = document.body.contents[0]

            data = title + ' ' + body
            data = preprocess_regex(data)

            if document['lewissplit'] == 'TRAIN':
                train_index_id_map[len(train_data)] = document_id
                train_data.append(data)
                train_target.append(target)
            else:
                test_index_id_map[len(test_data)] = document_id
                test_data.append(data)
                test_target.append(target)
    data = {}
    test = {}
    train = {}
    test['map'] = test_index_id_map
    test['data'] = test_data
    test['target'] = test_target
    train['map'] = train_index_id_map
    train['data'] = train_data
    train['target'] = train_target
    data['test'] = test
    data['train'] = train
    return data


def get_all_classes(filename='../data/all-topics-strings.lc.txt'):
    """
    Get all classes from the Reuters-21578 dataset.

    Parameters
    ----------
    filename : str
        Filename of file containing a text file where each row is a unique class.

    Returns
    -------
    list(str)
        List of classes

    """
    with open(filename, 'r') as label_file:
        labels = label_file.read().split('\n')
    return labels


def get_classes(classes,
                document_index,
                filename='../data/all-topics-strings.lc.txt',
                category_filter=None):
    """
    Get classes from the Reuters-21578 dataset as a binary vector for each document.

    Parameters
    ----------
    classes : dict(int : list(str))
        Dict that maps the document ID to the list of classes of that document.
    document_index : dict(int : int)
        Dict that maps the document ID to the matrix row index used.
    category_filter : list(str) (optional)
        List of classes which classes will be used.

    Returns
    -------
    label_index : dict(int : str)
        Dict that maps the matrix row index of the class matrix to the label.
    y : array of shape [n_documents, n_classes]
        The class labels of each document as a binary row vector.
    """
    if category_filter is not None:
        labels = category_filter
    else:
        labels = get_all_classes(filename)
    label_index = {label: index for index, label in enumerate(labels)}
    y = np.zeros((len(classes), len(label_index)))
    label_filter = set(labels)
    for document_id, document_labels in classes.items():
        if document_labels is not None:
            i = document_index[document_id]
            for label in document_labels:
                if label in label_filter:
                    j = label_index[label]
                    y[i, j] = 1
    # Reverse dict
    label_index = {v: k for k, v in label_index.items()}
    return label_index, y


def create_subset(filename, subset, n_samples, n_min_length, n_max_length,
                  filter_classes, min_n_classes,approx=None):
    trains, tests, _, all_texts, all_classes = process_directory(approx=approx)
    if subset == 'train':
        all_ids = trains
    elif subset == 'test':
        all_ids = tests
    else:
        all_ids = trains + tests

    texts, classes, ids = new_data(
        all_ids,
        all_texts,
        all_classes,
        n_samples,
        n_min_length,
        n_max_length,
        filter_classes,
        min_n_classes,
        only_filter_classes=False)
    description = 'subset: {}, n_samples: {}, min_length: {}, max_length: {}, filter_classes: {}, min_classes: {}'.format(
        subset, n_samples, n_min_length, n_max_length, filter_classes,
        min_n_classes)
    data = {'description': description, 'ids': ids}

    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return texts, classes, ids


def load_subset(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    _, _, _, all_texts, all_classes = process_directory()
    texts = []
    classes = []
    ids = data['ids']
    for id in ids:
        texts.append(all_texts[id])
        classes.append(all_classes[id])
    return texts, classes, ids


def main():
    create_subset('train_subset.pkl', 'train', 380, 0, 1500,
                  ['earn', 'acq', 'crude', 'corn'], [152, 114, 76, 38])
    create_subset('test_subset.pkl', 'test', 90, 0, 1500,
                  ['earn', 'acq', 'crude', 'corn'], [40, 25, 15, 10])


if __name__ == '__main__':
    main()
