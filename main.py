
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd
import pickle
import joblib
import numpy as np
from pathlib import Path
import os
import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced, geometric_mean_score

from src.generators.generator import DataGenerator
# from src.generators.generator_text import TextGenerator
# from src.generators.generator_image import ImageGenerator
from src.models.models_text import *
from src.models.models_image import *
from src.models.models_fusion import *

BATCH_SIZE = 64
EPOCHS_TEXT= 30
EPOCHS_IMAGE = 30
EPOCHS_FUSION = 50
NUM_FOLDS = 3
NUM_TARGETS = 84916
TARGET_SHAPE = [224, 224, 3]
TEST_SPLIT= .16
RANDOM_STATE = 123

if __name__ == "__main__":
    
    data_generator = DataGenerator(
        # csv_texts="data/raw/X_train_update.csv",
        # csv_labels="data/raw/Y_train_CVw08PX.csv",
        # root_dir="data/raw/images/image_train/",
        csv_data="data/raw/data_translated_1_stemmed_1.csv",
        samplings=NUM_TARGETS,
        random_state=RANDOM_STATE,
        translate=False,
        stem=False,
        exploration=False,
    )
    data_train, data_test = data_generator.split(test_size=TEST_SPLIT)

   
    model_image = ModelImage_CNN_Lenet(
        suffix = "_224",
        epochs = EPOCHS_TEXT,
        batch_size = BATCH_SIZE,
        target_shape = TARGET_SHAPE,
        load=False,
        class_weight=data_generator.class_weight
    )
    model_image.fit(
        data_train,
        validation_data = data_test,
        crossval=True,
    )

    # model_text = ModelText_Neural_Embedding(
    #     suffix="",
    #     epochs=EPOCHS_TEXT,
    #     batch_size=BATCH_SIZE,
    #     load=True,
    #     class_weight=data_generator.class_weight
    # )
    # model_text.fit(
    #     data_train,
    #     validation_data=data_test,
    #     crossval=True,
    # )

    # model_fusion = ModelFusion(
    #     suffix = model_text.model_name+"_"+model_image.model_name+"_",
    #     epochs = EPOCHS_FUSION,
    #     batch_size = BATCH_SIZE,
    #     models = [model_image, model_text], #Order imports
    #     models_concat_layer_num = [-2, -2],
    #     load=False,
    #     class_weight=data_generator.class_weight
    # )
    
    # print("Fusion")
    # model_fusion.load_models().fit(
    #     data_generator,
    #     validation_data = data_test,
    #     crossval=True,
    # )