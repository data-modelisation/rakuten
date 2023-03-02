import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        self.links, self.labels = self.load()
        self.encode_targets()
        self.generator=ImageDataGenerator(
            rescale=1/255
        )

    def __splitter__(self, split_indexes, type_="train", is_batch=True):
        
        splitted_generator = copy.deepcopy(self.generator)
        
        if type_ == "train":
            kwargs = {
                "width_shift_range":0.1,
                "height_shift_range":0.1,
                "shear_range":0.1,
                "zoom_range":0.1,
                "fill_mode":'nearest',
                "horizontal_flip":True,
                "vertical_flip":False,
            }
        else:
            kwargs = {}
        df = pd.DataFrame.from_dict(
            {
                "links" : self.links[split_indexes],
                "labels" : self.labels[split_indexes]
            }
            ).astype({"links": str, "labels":str})

        return splitted_generator.flow_from_dataframe(
            dataframe=df,
            x_col="links",
            y_col="labels",
            target_size=self.target_shape[:2],
            batch_size=self.batch_size,
            class_mode="sparse",
            shuffle=False,
            **kwargs)
            

    def load(self):

        labels = pd.read_csv(self.csv_labels).prdtypecode

        texts = pd.read_csv(self.csv_texts)
        links = self.root_dir + "image_" + texts.imageid.map(str) + "_product_" + texts.productid.map(str) + ".jpg"

        return links.values, labels.values

    def show_mask_variance(self,threshold=.075):

        images = next(iter(copy.deepcopy(self)))
        mean_images = images[0].mean(axis=3).reshape(32, -1)
        vt = VarianceThreshold(threshold=.075)
        vt.fit_transform(mean_images)
        mask= vt.get_support()
        plt.imshow(mask.reshape((224, 224)), cmap="gray")
        plt.savefig("notebooks/images/mask.png")
        