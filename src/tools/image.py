import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import logging
import pandas as pd
import string
import seaborn as sns
import multiprocessing as mp

import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import commons
from src.models.models_image import cnn_simple

def build_pipeline_model(name="cnn_simple", input_dim=()):
    print(input_dim)
    if name == "cnn_simple":
        model = cnn_simple(input_dim)

    return model

def flow_generators(df_train_image, df_test_image, target_shape,batch_size):
    #Chargement des generateurs
    Image_train, Image_test = get_image_generators()

    #Flow des generateurs
    train_generator = Image_train.flow_from_dataframe(
        dataframe=df_train_image,
        x_col="links",
        y_col="label",
        shear_range=.1,
        rotation_range=10,
        zoom_range=.1,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip=True,
        target_size=target_shape[:2],
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False)

    test_generator = Image_test.flow_from_dataframe(
        dataframe=df_test_image,
        x_col="links",
        y_col="label",
        target_size=target_shape[:2],
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False)

    return train_generator, test_generator

def get_image_generators():
    Image_train = ImageDataGenerator(
        rescale=1/255
    )#, preprocessing_function=image_tools.togray)
    Image_test = ImageDataGenerator(
        rescale=1/255
    )#, preprocessing_function=image_tools.togray)
    
    return Image_train, Image_test

#Crop image (x, y, 3)
def crop_image(im, crop_x_ratio = .2, crop_y_ratio = .2):

    img_width, img_height = im.shape[0], im.shape[1]
    crop_x_side = int(crop_x_ratio * img_width // 2)
    crop_y_side = int(crop_y_ratio * img_height // 2)

    cropped_image=im[
        crop_x_side:img_width-crop_x_side, 
        crop_x_side:img_width-crop_x_side, 
        :]
    
    return cropped_image

def togray(im):
    return skimage.color.rgb2gray(im).reshape(-1,1)
#Lecture et reduction de la taille d'une image
def read_and_resize(link, width=100, height=100, resize_=True, gray=True, rescale=True, flatten=False):
    im = imread(link)

    if resize_:
        im = resize(im, (width, height, 3))

    if rescale:
        im /= 255

    if flatten:
        im = im.reshape(-1, 3)
    
    return skimage.color.rgb2gray(im) if gray else im


#Lecture de toutes les images
def read_images(links, **kwargs):
    images = np.array([read_and_resize(link, **kwargs) for link in links])
    return images

#Transformeur pour passer de RGB à Gray
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
        return np.array([skimage.color.rgb2gray(img) for img in X])

#Transformeur pour recupere des données sur les canaux de l'image
class ChannelAnalyzer(BaseEstimator, TransformerMixin):
    def __init__(self, folder="./"):
        self.folder = folder
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        X_w_ratios = get_white_ratio(X)
        X_rgb_means = get_channel_means(X)
        #TODO merger
        return X_copy

# Construction du pipeline pour le preprocessing
def build_pipeline_preprocessor():

    preprocessor = Pipeline(steps=[    
        #('analyzer', ChannelAnalyzer()),
        ("gray", RGB2GrayTransformer()),
        ('reduct', PCA(n_components = .9)),
        ])

    return preprocessor


#Moyenne des canaux
def get_channel_means(im:np.array):
    return im.mean(axis=1)

# Ratio de blanc dans l'image
def get_white_ratio(im:np.array):
    mask_white = im[:,:,:] == [1, 1, 1]
    return mask_white.sum(axis=(1,2)) / (im.shape[1] * im.shape[2])

