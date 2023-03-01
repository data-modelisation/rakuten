import numpy as np
import math
import tensorflow as tf
import os
import copy
from collections.abc import Iterable

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

class CommonGenerator(tf.keras.utils.Sequence):

    def __init__(self,
        *args,
        **kwargs,
        ):

        self.batch_size = kwargs.get("batch_size")
        self.encoder = kwargs.get("encoder")
        self.csv_texts = kwargs.get("csv_texts")
        self.csv_labels = kwargs.get("csv_labels")
        self.preprocessor = kwargs.get("preprocessor")
        self.preprocessor_fit = kwargs.get("preprocessor_fit")
        self.encoder = kwargs.get("encoder")
        self.encoded = kwargs.get("encoded")

    def fit_preprocess(self,):
        if self.preprocessor_fit:
            self.features = self.preprocessor.fit_transform(self.features)
        else:
            self.features = self.preprocessor.transform(self.features)

    def encode_targets(self,):
        if not self.encoded:
            self.encoder = LabelEncoder()
            self.targets = self.encoder.fit_transform(self.labels)
        else:
            self.targets = self.encoder.transform(self.labels)

        self.__set_class_weigth__()

    def __splitter__(self, split_indexes, is_batch=True):
        
        splitted_generator = copy.deepcopy(self)
        if is_batch:
            indexes = list(self.flatten([self.__get_batch_indexes__(idx_1D) for idx_1D in split_indexes]))
        else:
            indexes = split_indexes
        
        splitted_generator.targets = splitted_generator.targets[indexes]
        splitted_generator.features = splitted_generator.features[indexes]

        return splitted_generator

    def split(self, split_indexes=[], is_batch=True):

        return [self.__splitter__(indexes, is_batch=is_batch) for indexes in split_indexes]


    def __set_class_weigth__(self,):
        values = self.targets
        classes = np.unique(values)
        weights = compute_class_weight('balanced', classes=classes, y=values)
        self.class_weight = {k: v for k, v in zip(classes, weights)}

    def __len__(self):
        return math.ceil(len(self.targets) / self.batch_size)

    def __get_batch_indexes__(self, idx):
        max_idx_possible = min(len(self.targets), (idx + 1) * self.batch_size)
        return np.arange(start=idx * self.batch_size, stop=max_idx_possible, dtype=int)

    def flatten(self, xs):
        
        for x in xs:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from self.flatten(x)
            else:
                yield x

    def decode(self, values):
        return self.encoder.inverse_transform(values)
        
    def __getitem__(self, batch_idx):

        indexes = self.__get_batch_indexes__(batch_idx)

        features = np.array(self.features[indexes])
        targets = np.array(self.targets[indexes])

        return features, targets
