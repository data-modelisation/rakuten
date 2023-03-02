import tensorflow as tf
import numpy as np

from src.generators.generator import CommonGenerator

class FusionGenerator(CommonGenerator):

    def __init__(self,
        generators,
        **kwargs

    ):
        super().__init__(**kwargs)
        
        self.generators = generators
        self.targets = self.generators[0].targets
        self.features = np.zeros_like(self.targets)
        self.class_weight = self.generators[0].class_weight
        self.encoder = self.generators[0].encoder
        self.batch_size = self.generators[0].batch_size

    def __getitem__(self, batch_idx):

        indexes = self.__get_batch_indexes__(batch_idx)
        targets = np.array(self.targets[indexes])

        extracted = [generator[batch_idx] for generator in self.generators]
        extracted_features = [extract[0] for extract in extracted]

        return extracted_features, targets