import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import numpy as np
import logging
from pytesseract import pytesseract
import pandas as pd
import string
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from . import commons
import multiprocessing as mp

NUM_SAMPLES = 1000


#Lecture de fichier image
@commons.timeit
def read_image_files(pid:pd.Series, iid=pd.Series, folder="", ):

    names = "/image_" + iid.map(str) + "_product_" + pid.map(str) + ".jpg"
    links = folder + names

    df_image = pd.DataFrame.from_dict({
        "links" : links,
    })

    return df_image

# Feature-engineering du texte
@commons.timeit
def apply_feature_engineering(df:pd.DataFrame, folder=""):

    #Parallisation de l'extraction d'infos
    pool = mp.Pool(mp.cpu_count())
    infos = [pool.apply(get_image_infos, folder+link) for link in df.links.values]
    results = np.array(infos)
    pool.close() 
    
    #Le taux de blanc est la 1ere colonne de results
    df["w_ratios"] = results[:, 0]

    # Les trois moyennes des canaux sont les colonnes suivantes
    df[["r_means", "b_means", "g_means"]] = np.array(results[:, 1:])
    df[""]
    return df

#Pour une photo, on récupère ses infos
def get_image_infos(*args):
    link = ''.join(args)
    im = Image.open(link)
    w_ratios = get_white_ratio(im)
    rgb_means = get_channel_means(im)
    return w_ratios, rgb_means[0], rgb_means[1], rgb_means[2]

#Moyenne des canaux
def get_channel_means(im:Image):
    
    im_array = np.array(im)
    return im_array.mean(axis=(0,1))

# Ratio de blanc dans l'image
def get_white_ratio(im:Image):
    mask_white = np.array(im) == [255, 255, 255]
    return mask_white.sum() / (im.width * im.height * 3)

# Extraction de texte depuis une image
def get_text_from_image(im:Image):
    return pytesseract.image_to_string(im)











