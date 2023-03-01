import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform


from src.generators.generator import CommonGenerator
import copy

class ImageGenerator(CommonGenerator):

    def __init__(self,
        root_dir="",
        target_shape=[100, 100, 3],
        crop=True,
        **kwargs
        ):

        super().__init__(**kwargs)

        self.root_dir = root_dir
        self.target_shape = target_shape
        self.crop = crop
        self.features, self.labels = self.load()
        self.encode_targets()

    def load(self):

        labels = pd.read_csv(self.csv_labels).prdtypecode

        texts = pd.read_csv(self.csv_texts)
        links = self.root_dir + "image_" + texts.imageid.map(str) + "_product_" + texts.productid.map(str) + ".jpg"

        return links.values, labels.values

        
    def __getitem__(self, batch_idx):

        indexes = self.__get_batch_indexes__(batch_idx)
        links = self.features[indexes]
        images = [io.imread(link) for link in links]

        if self.crop:
            images = [image[50:450, 50:450] for image in images]

        images = [transform.resize(image, self.target_shape[:2]) for image in images]
        images = np.array(images)
        targets = np.array(self.targets[indexes])

        return images, targets

    def show_mask_variance(self,):

        images = next(iter(copy.deepcopy(self)))

        vt = VarianceThreshold(threshold=.075)
        vt.fit_transform(images)
        mask= vt.get_support()
        plt.imshow(mask.reshape((500, 500)))
        plt.show()