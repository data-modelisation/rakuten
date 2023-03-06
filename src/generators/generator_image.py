import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg

from src.generators.generator import CommonGenerator
import copy

class ImageGenerator(CommonGenerator):

    def __init__(self,
        root_dir="",
        target_shape=[100, 100, 3],
        crop=True,
        samples=None,
        **kwargs
        ):

        super().__init__(**kwargs)

        self.samples=samples
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.crop = crop
        self.features, self.labels = self.load()
        

        print(f"Nombre d'images traitées : {len(self.features)}")
        self.encode_targets()
        print(f"Nombre de targets traitées : {len(self.targets)}")

    def flow(self, type_="train"):

        if type_ == "train":
            kwargs = {
                # "width_shift_range":0.1,
                # "height_shift_range":0.1,
                # "shear_range":0.1,
                # "zoom_range":0.1,
                # "fill_mode":'nearest',
                # "horizontal_flip":True,
                # "vertical_flip":False,
            } 
        else:
            kwargs={}

        df = pd.DataFrame.from_dict(
            {
                "links" : self.features,
                "labels" : self.targets,
            }
            ).astype({"links": str, "labels":str})

        return ImageDataGenerator(
                    rescale=1/255,
                    preprocessing_function=self.preprocessing_function
                ).flow_from_dataframe(
                    dataframe=df,
                    x_col="links",
                    y_col="labels",
                    target_size=self.target_shape[:2],
                    batch_size=self.batch_size,
                    class_mode="sparse",
                    shuffle=False,
                    **kwargs
                )
       
                
    def load(self):

        labels = pd.read_csv(self.csv_labels).prdtypecode
        texts = pd.read_csv(self.csv_texts)

        if self.samples:
            texts = texts.head(self.samples)
            labels = labels[:self.samples]

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
        

    def show_images_per_category(self, num_images=5):


        unique_labels = np.unique(self.labels)
        num_labels = len(unique_labels)

        fig, axs = plt.subplots(num_images, len(unique_labels), 
            figsize=(num_labels, num_images))

        for idx_label, label in enumerate(unique_labels):
            mask = self.labels == label
            seleted_features = self.features[mask][:num_images]
            

            for idx_image, seleted_feature in enumerate(seleted_features): 
                image = mpimg.imread(seleted_feature)
                axs[idx_image, idx_label].imshow(image)
                axs[idx_image, idx_label].axis('off')

            title = axs[idx_image, idx_label].set_title(f"Label {label}", loc='center', y=5.5, fontdict={"fontsize":8, "fontweight":"bold"})


        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig("notebooks/images/images_category.png")

