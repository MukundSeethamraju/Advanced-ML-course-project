#! /usr/bin/env python
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import os
import pickle

from utils import shuffle_lists, flatten


class DataClass:

    def __init__(self, subset, n_samples, n_min_length, n_max_length, filter_classes, min_n_classes):
        self.ids = []
        self.texts = {}
        self.classes = {}

        # train or test
        self.subset = subset

        self.n_samples = n_samples
        self.n_min_length = n_min_length
        self.n_max_length = n_max_length

        # list of desired classes
        self.filter_classes = filter_classes

        # dict of desired minimum number fo each class
        self.min_n_classes = min_n_classes

        self.path = '../data/'
        self.filenames = [
            filename for filename in os.listdir(self.path)
            if filename.startswith('reut2-')]

    def __str__(self):
        return 'subset: {}, n_samples: {}, min_length: {}, max_length: {}, filter_classes: {}, min_classes: {}'.format(
        self.subset, self.n_samples, self.n_min_length, self.n_max_length, self.filter_classes,
            self.min_n_classes)

    def build(self):
        self.process_directory()
        self.filter()

    def get_class_set(self):
        ret = set(flatten(self.classes))
        return ret

    def save(self, filename):
        data = {'description': self.__str__(), 'ids': self.ids}
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

        self.process_directory()

        texts = []
        classes = []
        _ids = data['ids']
        for id in _ids:
            texts.append(self.texts[id])
            classes.append(self.classes[id])

        self.ids = _ids
        self.texts = texts
        self.classes = classes

    def process_directory(self):
        """
        Reads and preprocesses the Reuters-21578 dataset.

        The dataset is split using the "ModApte"-split."""

        for filename in self.filenames:
            print(filename)

            _ids, _texts, _classes = self.process_file(self.path + filename)

            self.ids.extend(_ids)
            self.texts.update(_texts)
            self.classes.update(_classes)

    def process_file(self, file_path, single_class = True):
        """
        Reads and preprocesses a file of the Reuters-21578 dataset.

        The dataset is split using the "ModApte"-split.

        Parameters
        ----------
        filename : string (optional)
            Filename of the sgml file.

        Returns
        -------
        ids : all ids

        texts : dict(int : str)
            Map of document-IDS and the texts of the documents.
        classes: dict(int : list(str))
            Map of document-IDS and the class or classes of the documents.
        """
        ids = []
        texts = {}
        classes = {}

        with open(file_path, 'r') as sgml_file:
            corpus = BeautifulSoup(sgml_file.read(), 'html.parser')

            for document in corpus('reuters'):

                # Check if document is "ModApte"
                # According to the README (VIII.B.)
                # Training: lewissplit=train, topics=yes
                # Testing: lewissplit=test, topics=yes
                if document['topics'] == 'YES' and \
                        document['lewissplit'].lower() == self.subset:


                    document_id = int(document['newid'])
                    cls = []

                    for topic in document.topics.contents:
                        cls.extend(topic.contents)

                    if document.title is None:
                        title = ''
                    else:
                        title = document.title.contents[0]

                    if document.body is None:
                        body = ''
                    else:
                        body = document.body.contents[0]
                        text = title + ' ' + body
                        texts[document_id] = self.preprocess_regex(text)

                    if not len(cls) > 1 and single_class:
                        texts[document_id] = self.preprocess_regex(body)
                        classes[document_id] = cls
                        ids.append(document_id)

                    elif not single_class:
                        texts[document_id] = self.preprocess_regex(body)
                        classes[document_id] = cls
                        ids.append(document_id)

        return ids, texts, classes

    def filter(self):
        """
        Processes the reuters data to lists of text and classes.

        Filters the data by document ids, the documents classes, shuffles it and then returns
        either the n_samples or as many as possible.

        Returns
        -------
        picked_texts : list(str) [n_samples]
            List of the document texts of the data subset.
        picked_classes : list(str) [n_samples]
            List of the classes for each document of the data subset.
        picked_ids : list(int) [n_samples]
            List of the document ids of the data subset.
        """

        document_ids_set = set(self.ids)

        sample_texts = []
        sample_classes = []
        sample_ids = []

        # Get the documents that have an id that is in the document_ids parameter
        for document_id, text in self.texts.items():
            if document_id in document_ids_set:
                if self.n_min_length < len(text) < self.n_max_length:
                    sample_texts.append(text)
                    sample_classes.append(self.classes[document_id])
                    sample_ids.append(document_id)

        sample_texts, sample_classes, sample_ids = shuffle_lists(sample_texts, sample_classes, sample_ids)

        if self.filter_classes:
            if self.min_n_classes:

                # Sort filter_classes by size, smallest first
                # dict of number of documents for each class
                counted_classes = Counter(list(flatten(sample_classes))).most_common()

                counted_classes.reverse()

                sorted_filter_classes = [c for c, _ in counted_classes if c in self.filter_classes]

                picked_texts = []
                picked_classes = []
                picked_ids = []
                rest_texts = []
                rest_classes = []
                rest_ids = []
                N = 0

                for cl in sorted_filter_classes:

                    min_n = self.min_n_classes[cl]

                    # all texts classes and ids belonging to current class
                    class_texts, class_classes, class_ids = self.filter_by_class(sample_texts, sample_classes, sample_ids, [cl])

                    # Check if document has already been added
                    picked_ids_set = set(picked_ids)

                    allready_added_document_indices = [
                        i for i, id in enumerate(class_ids) if id in picked_ids_set]

                    if allready_added_document_indices:
                        allready_added_document_indices = set(allready_added_document_indices)

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
                if N < self.n_samples and rest_texts and rest_classes:
                    picked_texts.extend(rest_texts[:self.n_samples - N])
                    picked_classes.extend(rest_classes[:self.n_samples - N])
                    picked_ids.extend(rest_ids[:self.n_samples - N])

            else:
                picked_texts, picked_classes, picked_ids = self.filter_by_class(
                    sample_texts, sample_classes, sample_ids, self.filter_classes)
                picked_texts = picked_texts[:self.n_samples]
                picked_classes = picked_classes[:self.n_samples]
                picked_ids = picked_ids[:self.n_samples]

        else:
            picked_texts = sample_texts[:self.n_samples]
            picked_classes = sample_classes[:self.n_samples]
            picked_ids = sample_ids[:self.n_samples]

        self.texts, self.classes, self.ids = shuffle_lists(picked_texts, picked_classes, picked_ids)

    @ staticmethod
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

    @ staticmethod
    def preprocess_regex(text):
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
        p1 = re.compile(r'([^a-zA-Z\s\'])')  # Removes everything except a-Z, and '
        p2 = re.compile(r'(\'(\w+)\')')  # Remove quotes
        text = p2.sub(r'\g<2>', p1.sub(r'', text).lower())
        words = text.split()
        word_filter = set(stopwords.words('english'))
        filtered_words = [
            word.replace('\'', '') for word in words if word not in word_filter
        ]
        return ' '.join(filtered_words)

def main():
    # params
    subset = 'train'
    n_samples = 380
    n_min_length = 0
    n_max_length = 1500
    min_n_classes = {'earn' : 152, 'acq': 114, 'crude': 76, 'corn':38}
    filter_classes = ['earn', 'acq', 'crude', 'corn']

    D = DataClass(subset, n_samples, n_min_length, n_max_length, filter_classes, min_n_classes)

    D.load('bajs')

    print(D.ids)


if __name__ == '__main__':
    main()
