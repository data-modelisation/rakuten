import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import numpy as np
import logging
from pytesseract import pytesseract
import pandas as pd
import string
import seaborn as sns
import multiprocessing as mp

import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from . import commons

#Lecture et reduction de la taille d'une image
def read_and_resize(link, width=100, height=100):
    im = imread(link)
    im = resize(im, (width, height, 3))
    return im.reshape(-1,3)

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

# Construction du pipeline pour le modèle image
def build_pipeline_model():

    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ("classifier", SVC(kernel='rbf') )
    ])

    return model

#Moyenne des canaux
def get_channel_means(im:np.array):
    return im.mean(axis=1)

# Ratio de blanc dans l'image
def get_white_ratio(im:np.array):
    mask_white = im[:,:,:] == [255, 255, 255]
    return mask_white.sum(axis=(1,2)) / (im.shape[1] * im.shape[2])

# Extraction de texte depuis une image
def get_text_from_image(im:Image):
    return pytesseract.image_to_string(im)