import numpy as np
import pandas as pd

from src.tools.text import pipeline_loader, pipeline_preprocess
from src.generators.generator import CommonGenerator

class TextGenerator(CommonGenerator):

    def __init__(self,
        stem=True, 
        clean=True, 
        max_words=100, 
        max_len=100,
        **kwargs
        ):

        super().__init__(**kwargs)

        self.stem = stem
        self.clean = clean
        self.max_words = max_words
        self.max_len = max_len

        self.features, self.labels = self.load()
        self.fit_preprocess()
        self.encode_targets()

    def load(self):

        labels = pd.read_csv(self.csv_labels).prdtypecode
        texts = pd.read_csv(self.csv_texts)
        text_loader = pipeline_loader()
        return text_loader.fit_transform(texts).values, labels.values

